// SPDX-FileCopyrightText: 2026 Mika Tammi
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    Algorithm, CHUNK_SIZE, ContentId, Digest, FileDigests, FileHasher, GXHASH, HashTable,
    chunk_count, fingerprint,
};
use crate::manifest::{FileTable, Manifest};

/// Deterministic bytes: two chunks and a 16 KiB tail.
fn pattern() -> Vec<u8> {
    let len = usize::try_from(2 * CHUNK_SIZE + 16 * 1024).unwrap();
    (0..len)
        .map(|i| u8::try_from((i * 31 + i / 251) % 251).unwrap())
        .collect()
}

fn hex(algorithm: Algorithm, bytes: &[u8]) -> String {
    algorithm.digest(bytes).unwrap().to_hex()
}

// Published test vectors of the standard algorithms.
#[test]
fn known_vectors() {
    assert_eq!(
        hex(Algorithm::Sha512, b""),
        "cf83e1357eefb8bdf1542850d66d8007d620e4050b5715dc83f4a921d36ce9ce\
         47d0d13c5d85f2b0ff8318d2877eec2f63b931bd47417a81a538327af927da3e"
    );
    assert_eq!(
        hex(Algorithm::Sha3, b""),
        "a69f73cca23a9ac5c8b567dc185a756e97c982164fe25859e0d1dcc1475c80a6\
         15b2123af1f5f94c11e3e9402c3ac558f500199d95b6d3e301758586281dcd26"
    );
    assert_eq!(
        hex(Algorithm::Blake3, b""),
        "af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262"
    );
    assert_eq!(
        hex(Algorithm::Sha256, b""),
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
    assert_eq!(
        hex(Algorithm::Xxh3, b""),
        "99aa06d3014798d86001c324468d497f"
    );
    assert_eq!(
        hex(Algorithm::Sha256, b"abc"),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
}

// Golden vectors of the fast checks, the same on every system: gxhash must
// agree between its AES-NI and its NEON implementation. xxHash's own
// `xxhsum -H2` gives the same XXH3-128 digests, so they are stored in its
// canonical byte order.
#[test]
fn fast_check_golden_vectors() {
    let pattern = pattern();
    assert_eq!(
        hex(Algorithm::Xxh3, b"FuryModel"),
        "5d25a484abc680588475c0d55456c18a"
    );
    assert_eq!(
        hex(Algorithm::Xxh3, &pattern),
        "597e318cf56cf536b6eedfae8b04d7eb"
    );
    if Algorithm::Gxhash3.is_supported() {
        assert_eq!(
            hex(Algorithm::Gxhash3, b""),
            "e3c6d8815d54d8b4eed96c4396ffe83f"
        );
        assert_eq!(
            hex(Algorithm::Gxhash3, b"FuryModel"),
            "011eda42ce79d722c5f86ccc724f9aad"
        );
        assert_eq!(
            hex(Algorithm::Gxhash3, &pattern),
            "98bd56119fc27ba58e402e4e25ace0bc"
        );
    } else {
        assert_eq!(Algorithm::Gxhash3.digest(b""), None);
    }
}

// The flake's checks say whether their build should have gxhash3-128; the
// portable ones must not, the others must, on every system, and the CPUs
// they run on have AES instructions, so it is supported where it is built.
#[test]
fn gxhash_is_built_where_expected() {
    let expected = match option_env!("FURYMODEL_GXHASH_EXPECTED") {
        Some("1") => true,
        Some("0") => false,
        _ => return,
    };
    assert_eq!(GXHASH, expected, "whether this build has gxhash3-128");
    assert_eq!(
        Algorithm::Gxhash3.is_supported(),
        expected,
        "whether gxhash3-128 is supported here"
    );
}

#[test]
fn ids() {
    for algorithm in Algorithm::ALL {
        assert_eq!(Algorithm::from_id(algorithm.id()), Some(algorithm));
        assert_eq!(
            algorithm.table_path(),
            format!("hashes/{}.bin", algorithm.id())
        );
        if algorithm.is_supported() {
            assert_eq!(
                algorithm.digest(b"").unwrap().as_bytes().len(),
                algorithm.digest_size()
            );
        }
    }
    assert_eq!(Algorithm::from_id("md5"), None);
}

#[test]
fn picks_the_fastest_chunked_table() {
    let all = Algorithm::ALL;
    assert_eq!(
        Algorithm::fastest_chunked(all, |_| true),
        Some(Algorithm::Gxhash3)
    );
    assert_eq!(
        Algorithm::fastest_chunked(all, |algorithm| algorithm != Algorithm::Gxhash3),
        Some(Algorithm::Xxh3)
    );
    assert_eq!(
        Algorithm::fastest_chunked([Algorithm::Sha512, Algorithm::Blake3], |_| true),
        None
    );
}

fn supported() -> Vec<Algorithm> {
    Algorithm::ALL
        .into_iter()
        .filter(|algorithm| algorithm.is_supported())
        .collect()
}

#[test]
fn chunked_file_digest_is_the_digest_of_the_chunk_digests() {
    let pattern = pattern();
    let mut hasher = FileHasher::new(&supported()).unwrap();
    hasher.update(&pattern);
    let digests = hasher.finish();
    assert_eq!(chunk_count(pattern.len() as u64), 3);
    let chunk = usize::try_from(CHUNK_SIZE).unwrap();
    for digests in &digests {
        let algorithm = digests.algorithm;
        if algorithm.is_chunked() {
            let expected: Vec<Digest> = pattern
                .chunks(chunk)
                .map(|bytes| algorithm.digest(bytes).unwrap())
                .collect();
            assert_eq!(digests.chunks, expected);
            let concatenated: Vec<u8> = expected
                .iter()
                .flat_map(|digest| digest.as_bytes().to_vec())
                .collect();
            assert_eq!(digests.file, algorithm.digest(&concatenated).unwrap());
        } else {
            assert_eq!(digests.chunks, []);
            assert_eq!(digests.file, algorithm.digest(&pattern).unwrap());
        }
    }
    assert_eq!(digests.len(), supported().len());
}

#[test]
fn hasher_doesnt_depend_on_how_bytes_are_fed() {
    let pattern = pattern();
    let whole = {
        let mut hasher = FileHasher::new(&supported()).unwrap();
        hasher.update(&pattern);
        hasher.finish()
    };
    for piece in [1, 4096, 1_000_003, usize::try_from(CHUNK_SIZE).unwrap() + 1] {
        let mut hasher = FileHasher::new(&supported()).unwrap();
        for bytes in pattern.chunks(piece) {
            hasher.update(bytes);
        }
        assert_eq!(hasher.size(), pattern.len() as u64);
        assert_eq!(hasher.finish(), whole, "pieces of {piece} bytes");
    }
    // No bytes: no chunks.
    let empty = FileHasher::new(&[Algorithm::Xxh3]).unwrap().finish();
    assert_eq!(empty[0].chunks, []);
    assert_eq!(empty[0].file, Algorithm::Xxh3.digest(b"").unwrap());
}

/// A file table of three files: the manifest, a payload of three chunks
/// and an auxiliary file.
fn table() -> (FileTable, Vec<Vec<u8>>) {
    let manifest = serde_json::json!({
        "format": "furymodel",
        "format_version": 1,
        "name": "tiny",
        "arch": "tinyarch",
        "schema_version": 1,
        "config": {},
        "alignment": 16384,
        "files": [{
            "path": "weights.bin",
            "size": 2 * CHUNK_SIZE + 16384,
            "tags": [],
            "tensors": [],
        }],
        "groups": [],
        "aux": [{"path": "aux/vocab.txt", "size": 3, "role": "vocab"}],
        "provenance": {
            "source": {"kind": "synthetic", "files": []},
            "tool": {"name": "tests", "version": "0"},
            "transforms": [],
            "notes": [],
        },
    });
    let manifest = serde_json::to_vec(&manifest).unwrap();
    let (_, table) = Manifest::parse(&manifest).unwrap();
    let contents: Vec<Vec<u8>> = table
        .entries()
        .iter()
        .map(|entry| match entry.path.as_str() {
            "furymodel.json" => manifest.clone(),
            "weights.bin" => pattern(),
            "aux/vocab.txt" => b"a b".to_vec(),
            other => panic!("{other}"),
        })
        .collect();
    (table, contents)
}

fn tables(table: &FileTable, contents: &[Vec<u8>]) -> Vec<HashTable> {
    let mut per_file: Vec<Vec<FileDigests>> = contents
        .iter()
        .map(|bytes| {
            let mut hasher = FileHasher::new(&supported()).unwrap();
            hasher.update(bytes);
            hasher.finish()
        })
        .collect();
    supported()
        .into_iter()
        .map(|algorithm| {
            let digests = per_file
                .iter_mut()
                .map(|digests| {
                    let index = digests
                        .iter()
                        .position(|digests| digests.algorithm == algorithm)
                        .unwrap();
                    digests.remove(index)
                })
                .collect();
            HashTable::new(algorithm, table, digests).unwrap()
        })
        .collect()
}

#[test]
fn tables_round_trip() {
    let (table, contents) = table();
    for hash_table in tables(&table, &contents) {
        let algorithm = hash_table.algorithm();
        let encoded = hash_table.encode();
        let decoded = HashTable::decode(&encoded, algorithm, &table).unwrap();
        assert_eq!(decoded, hash_table);
        // The manifest and the auxiliary file have a chunk each, the
        // payload three.
        let weights = table.index_of("weights.bin").unwrap();
        let (weights_chunks, chunks) = if algorithm.is_chunked() {
            (3, 5)
        } else {
            (0, 0)
        };
        assert_eq!(decoded.chunks(weights).len(), weights_chunks);
        assert_eq!(
            encoded.len(),
            128 + (3 + chunks) * algorithm.digest_size() + 16
        );
    }
}

#[test]
fn tables_reject_what_isnt_theirs() {
    let (table, contents) = table();
    let xxh3 = tables(&table, &contents)
        .into_iter()
        .find(|table| table.algorithm() == Algorithm::Xxh3)
        .unwrap();
    let encoded = xxh3.encode();

    let mut corrupt = encoded.clone();
    corrupt[200] ^= 1;
    let error = HashTable::decode(&corrupt, Algorithm::Xxh3, &table).unwrap_err();
    assert!(error.to_string().contains("corrupt"), "{error}");

    let error = HashTable::decode(&encoded, Algorithm::Gxhash3, &table).unwrap_err();
    assert!(error.to_string().contains("another algorithm"), "{error}");

    let error = HashTable::decode(&encoded[..100], Algorithm::Xxh3, &table).unwrap_err();
    assert!(error.to_string().contains("too short"), "{error}");

    // The same files with another size: another fingerprint.
    let other = {
        let mut json: serde_json::Value = serde_json::from_slice(&contents[1]).unwrap();
        json["aux"][0]["size"] = serde_json::json!(4);
        Manifest::parse(&serde_json::to_vec(&json).unwrap())
            .unwrap()
            .1
    };
    assert_ne!(fingerprint(&other), fingerprint(&table));
    let error = HashTable::decode(&encoded, Algorithm::Xxh3, &other).unwrap_err();
    assert!(
        error.to_string().contains("other files or sizes"),
        "{error}"
    );
}

// Pins the table format and the content id: a change to either changes
// these digests.
#[test]
fn format_golden_vectors() {
    let (table, contents) = table();
    let tables = tables(&table, &contents);
    let sha512 = tables
        .iter()
        .find(|table| table.algorithm() == Algorithm::Sha512)
        .unwrap();
    let sha3 = tables
        .iter()
        .find(|table| table.algorithm() == Algorithm::Sha3)
        .unwrap();
    let xxh3 = tables
        .iter()
        .find(|table| table.algorithm() == Algorithm::Xxh3)
        .unwrap();
    assert_eq!(
        hex(Algorithm::Sha256, &xxh3.encode()),
        "da89bb9e1e4e21ce7e11bde0bf960282363bf536cdaf357048e0f89d507d4836"
    );
    assert_eq!(
        hex(Algorithm::Sha256, &sha512.encode()),
        "440dbc7826e57600841bacffc53ffa0362dea222325e04b492e313f473d11622"
    );
    let id = ContentId::compute(&table, sha3).unwrap();
    assert_eq!(
        id.to_string(),
        "e5e150920af3e45d0888422c1e0f3253f788055a9d52a0a00649520aed6daf06de484cf1b6ab1190da116906c30452764ce35a140af89c8d3eb5eade88534d22"
    );
    assert_eq!(id.short(), id.to_string()[..16]);
    assert_eq!(ContentId::from_hex(&id.to_string()), Some(id));
    assert_eq!(ContentId::from_hex(&id.to_string().to_uppercase()), None);
    // Hex digits only: no signs, which a radix parse would take.
    for bad in [
        "+f".repeat(64),
        "-0".repeat(64),
        " f".repeat(64),
        "0g".repeat(64),
    ] {
        assert_eq!(ContentId::from_hex(&bad), None, "{bad}");
    }
    assert!(ContentId::compute(&table, xxh3).is_err());
    assert!(ContentId::compute(&table, sha512).is_err());
}

// A content id comes only from the sha3-512 table of the very files it
// names: a table of another file table of as many files is refused.
#[test]
fn content_id_needs_its_own_files_table() {
    let (table, contents) = table();
    let sha3 = tables(&table, &contents)
        .into_iter()
        .find(|table| table.algorithm() == Algorithm::Sha3)
        .unwrap();
    let other = {
        let mut json: serde_json::Value = serde_json::from_slice(&contents[1]).unwrap();
        json["aux"][0]["size"] = serde_json::json!(4);
        Manifest::parse(&serde_json::to_vec(&json).unwrap())
            .unwrap()
            .1
    };
    assert_eq!(other.len(), table.len());
    let error = ContentId::compute(&other, &sha3).unwrap_err();
    assert!(error.to_string().contains("of these files"), "{error}");
    ContentId::compute(&table, &sha3).unwrap();
}
