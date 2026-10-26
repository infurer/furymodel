// SPDX-FileCopyrightText: 2026 Mika Tammi
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fs;
use std::path::Path;
use std::sync::Arc;

use super::{Mismatch, Model, TensorLocation};
use crate::error::Error;
use crate::hash::{Algorithm, CHUNK_SIZE};
use crate::io::{AlignedBuf, CacheMode};
use crate::manifest::MANIFEST_PATH;
use crate::part::Part;
use crate::registry::{Dtype, Layout, Scheme};
use crate::testing::{BIG, algorithms, info, write_tiny};
use crate::writer::{TensorData, Writer};

fn len(size: u64) -> usize {
    usize::try_from(size).unwrap()
}

#[test]
fn opens_a_model_and_reads_every_chunk_verified() {
    let dir = tempfile::tempdir().unwrap();
    let written = write_tiny(dir.path(), false);
    let model = Model::open(&[dir.path()]).unwrap();
    assert_eq!(model.content_id(), written.content_id);
    assert_eq!(model.manifest(), &written.manifest);
    assert_eq!(model.hash_tables().len(), algorithms().len());
    let expected = if Algorithm::Gxhash3.is_supported() {
        Algorithm::Gxhash3
    } else {
        Algorithm::Xxh3
    };
    assert_eq!(model.verifier().unwrap(), expected);

    for mode in [CacheMode::Bypass, CacheMode::Use] {
        let reader = model.reader(mode).unwrap();
        assert_eq!(reader.algorithm(), expected);
        for (index, entry) in model.table().entries().iter().enumerate() {
            let file = reader.open(index).unwrap();
            let mut buffer = AlignedBuf::new(len(entry.size));
            file.read_all(&mut buffer).unwrap();
            assert_eq!(
                &buffer[..],
                fs::read(dir.path().join(&entry.path)).unwrap(),
                "{}",
                entry.path
            );
        }
    }

    // The second chunk of the big file, alone.
    let big = model.table().index_of("global/big.bin").unwrap();
    let file = model.reader(CacheMode::Bypass).unwrap().open(big).unwrap();
    let span = file.span(&(1..2));
    assert_eq!(span, CHUNK_SIZE..BIG.next_multiple_of(16384));
    let mut tail = AlignedBuf::new(len(span.end - span.start));
    file.read_chunks(1..2, &mut tail).unwrap();
    let contents = fs::read(model.path(big)).unwrap();
    assert_eq!(&tail[..], &contents[len(CHUNK_SIZE)..]);
}

#[test]
fn rejects_reads_that_dont_fit() {
    let dir = tempfile::tempdir().unwrap();
    write_tiny(dir.path(), false);
    let model = Model::open(&[dir.path()]).unwrap();
    let reader = model.reader(CacheMode::Use).unwrap();
    let big = model.table().index_of("global/big.bin").unwrap();
    let file = reader.open(big).unwrap();
    let mut buffer = AlignedBuf::new(len(CHUNK_SIZE));
    for chunks in [0..3, std::ops::Range { start: 2, end: 1 }] {
        let error = file.read_chunks(chunks, &mut buffer).unwrap_err();
        assert!(matches!(error, Error::Usage(_)), "{error}");
    }
    let error = file.read_chunks(0..2, &mut buffer).unwrap_err();
    assert!(matches!(error, Error::Usage(_)), "{error}");
    // Chunk numbers whose offsets overflow are a usage error too, not a
    // panic.
    let error = file
        .read_chunks(u64::MAX - 1..u64::MAX, &mut [])
        .unwrap_err();
    assert!(matches!(error, Error::Usage(_)), "{error}");
    assert_eq!(
        file.span(&(u64::MAX - 1..u64::MAX)),
        file.size()..file.size()
    );
    assert!(matches!(reader.open(1000), Err(Error::Usage(_))));
}

#[test]
fn catches_corrupt_payloads() {
    let dir = tempfile::tempdir().unwrap();
    write_tiny(dir.path(), false);
    let path = "layers/01/experts/02.bin";
    let full = dir.path().join(path);
    let mut contents = fs::read(&full).unwrap();
    contents[5] ^= 0x40;
    fs::write(&full, &contents).unwrap();

    let model = Model::open(&[dir.path()]).unwrap();
    let index = model.table().index_of(path).unwrap();
    let file = model.reader(CacheMode::Use).unwrap().open(index).unwrap();
    let mut buffer = AlignedBuf::new(contents.len());
    match file.read_all(&mut buffer) {
        Err(Error::Mismatch {
            path: found, chunk, ..
        }) => {
            assert_eq!((found.as_str(), chunk), (path, Some(0)));
        }
        other => panic!("{other:?}"),
    }

    let mismatches = model
        .verify(&algorithms(), CacheMode::Bypass, 3, &|_| {})
        .unwrap();
    let mut expected: Vec<Mismatch> = algorithms()
        .into_iter()
        .flat_map(|algorithm| {
            let file = Mismatch {
                path: path.to_owned(),
                algorithm,
                chunk: None,
            };
            let chunk = Mismatch {
                chunk: Some(0),
                ..file.clone()
            };
            if algorithm.is_chunked() {
                vec![file, chunk]
            } else {
                vec![file]
            }
        })
        .collect();
    expected.sort();
    assert_eq!(mismatches, expected);
}

#[test]
fn verifies_every_byte_and_reports_progress() {
    let dir = tempfile::tempdir().unwrap();
    let written = write_tiny(dir.path(), false);
    let model = Model::open(&[dir.path()]).unwrap();
    let bytes = std::sync::atomic::AtomicU64::new(0);
    let mismatches = model
        .verify(&algorithms(), CacheMode::Bypass, 4, &|n| {
            bytes.fetch_add(n, std::sync::atomic::Ordering::Relaxed);
        })
        .unwrap();
    assert_eq!(mismatches, []);
    assert_eq!(bytes.into_inner(), written.table.total_size());
}

#[test]
fn catches_a_corrupt_manifest() {
    let dir = tempfile::tempdir().unwrap();
    write_tiny(dir.path(), false);
    let path = dir.path().join(MANIFEST_PATH);
    // Still a valid manifest, of the same size, but not the one hashed.
    let json = String::from_utf8(fs::read(&path).unwrap()).unwrap();
    fs::write(&path, json.replace("\"tiny\"", "\"tinz\"")).unwrap();
    match Model::open(&[dir.path()]) {
        Err(Error::Mismatch { path, .. }) => assert_eq!(path, MANIFEST_PATH),
        other => panic!("{other:?}"),
    }
}

// Copies the files of the model in `from` whose paths `keep` accepts into
// `to`, with the manifest and the hash tables.
fn copy(from: &Path, to: &Path, keep: impl Fn(&str) -> bool) {
    let mut stack = vec![from.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for entry in fs::read_dir(dir).unwrap() {
            let source = entry.unwrap().path();
            if source.is_dir() {
                stack.push(source);
                continue;
            }
            let relative = source.strip_prefix(from).unwrap();
            let name = relative.display().to_string();
            if name == MANIFEST_PATH || name.starts_with("hashes/") || keep(&name) {
                let target = to.join(relative);
                fs::create_dir_all(target.parent().unwrap()).unwrap();
                fs::copy(&source, &target).unwrap();
            }
        }
    }
}

#[test]
fn opens_a_model_spread_over_roots() {
    let dir = tempfile::tempdir().unwrap();
    let written = write_tiny(&dir.path().join("whole"), false);
    let (a, b) = (dir.path().join("a"), dir.path().join("b"));
    copy(&dir.path().join("whole"), &a, |path| {
        path.starts_with("layers/")
    });
    copy(&dir.path().join("whole"), &b, |path| {
        !path.starts_with("layers/")
    });
    let model = Model::open(&[&a, &b]).unwrap();
    assert_eq!(model.content_id(), written.content_id);
    let expert = model.table().index_of("layers/00/experts/00.bin").unwrap();
    assert_eq!(model.path(expert), a.join("layers/00/experts/00.bin"));

    let error = Model::open(&[&a]).unwrap_err();
    let message = error.to_string();
    assert!(
        message.contains("incomplete: 3 of its 10 files"),
        "{message}"
    );
    assert!(message.contains("global/big.bin"), "{message}");

    // A file in two roots with different sizes.
    fs::create_dir_all(b.join("layers/00/experts")).unwrap();
    fs::write(b.join("layers/00/experts/00.bin"), b"short").unwrap();
    let error = Model::open(&[&a, &b]).unwrap_err();
    assert!(error.to_string().contains("has 5 bytes"), "{error}");
    assert!(
        matches!(error, Error::WrongSize { found: Some(5), .. }),
        "{error:?}"
    );
    fs::remove_file(b.join("layers/00/experts/00.bin")).unwrap();

    // A root of another model.
    let other = dir.path().join("other");
    copy(&dir.path().join("whole"), &other, |_| true);
    let json = fs::read_to_string(other.join(MANIFEST_PATH)).unwrap();
    fs::remove_file(other.join(MANIFEST_PATH)).unwrap();
    fs::write(
        other.join(MANIFEST_PATH),
        json.replace("tinyarch", "tinyarcj"),
    )
    .unwrap();
    let error = Model::open(&[&a, &other]).unwrap_err();
    assert!(error.to_string().contains("differ"), "{error}");
    assert!(matches!(error, Error::ManifestsDiffer { .. }), "{error:?}");

    let nowhere = dir.path().join("nowhere");
    let error = Model::open(&[&a, &nowhere]).unwrap_err();
    assert!(
        error.to_string().contains("has no furymodel.json"),
        "{error}"
    );
    assert!(
        matches!(&error, Error::NotARoot(root) if *root == nowhere),
        "{error:?}"
    );
}

#[test]
fn names_missing_parts() {
    let dir = tempfile::tempdir().unwrap();
    let written = write_tiny(&dir.path().join("whole"), false);
    let (a, b) = (dir.path().join("a"), dir.path().join("b"));
    let split = written.table.index_of("layers/00/experts/00.bin").unwrap() as u64;
    let files = written.table.len() as u64;
    copy(&dir.path().join("whole"), &a, |path| {
        written.table.index_of(path).unwrap() < usize::try_from(split).unwrap()
    });
    copy(&dir.path().join("whole"), &b, |path| {
        written.table.index_of(path).unwrap() >= usize::try_from(split).unwrap()
    });
    let part = |number, files: [u64; 2]| Part {
        format: "furymodel-part".to_owned(),
        format_version: 1,
        content_id: written.content_id.unwrap().to_string(),
        part: number,
        parts: 2,
        files: vec![files],
    };
    fs::write(a.join("part.json"), part(1, [0, split]).to_json()).unwrap();
    fs::write(b.join("part.json"), part(2, [split, files]).to_json()).unwrap();

    let model = Model::open(&[&b, &a]).unwrap();
    assert_eq!(
        model
            .parts()
            .iter()
            .map(|part| part.part)
            .collect::<Vec<_>>(),
        [1, 2]
    );
    let error = Model::open(&[&a]).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("delivery parts missing: 2 (of 2)"),
        "{error}"
    );
    // The same as data, for a runtime to act on.
    match error {
        Error::Incomplete {
            missing_parts,
            parts,
            ..
        } => assert_eq!((missing_parts, parts), (vec![2], Some(2))),
        other => panic!("{other:?}"),
    }

    fs::write(b.join("part.json"), part(1, [split, files]).to_json()).unwrap();
    let error = Model::open(&[&a, &b]).unwrap_err();
    assert!(error.to_string().contains("doesn't go with"), "{error}");
}

// A reader that holds the model by Arc, and the files it opens, go to
// threads of their own, and read there.
#[test]
fn shares_a_reader_across_threads() {
    fn shareable<T: Send + Sync + 'static>(_: &T) {}
    let dir = tempfile::tempdir().unwrap();
    write_tiny(dir.path(), false);
    let model = Arc::new(Model::open(&[dir.path()]).unwrap());
    let reader = model.shared_reader(CacheMode::Bypass).unwrap();
    shareable(&reader);
    let threads: Vec<_> = (0..model.table().len())
        .map(|index| {
            let file = reader.open(index).unwrap();
            shareable(&file);
            let path = model.path(index);
            std::thread::spawn(move || {
                let mut contents = AlignedBuf::new(usize::try_from(file.size()).unwrap());
                file.read_all(&mut contents).unwrap();
                assert_eq!(&contents[..], fs::read(path).unwrap());
            })
        })
        .collect();
    for thread in threads {
        thread.join().unwrap();
    }
}

// Where a tensor is, and its bytes read from there: inside one chunk, across
// two, at a chunk's start, of no bytes, and at the end of its file.
#[test]
fn locates_tensors_to_read() {
    const CHUNK: usize = 2 * 1024 * 1024;
    const PAGE: usize = 16 * 1024;
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("model");
    let writer = Writer::with_algorithms(&root, info(), &algorithms()).unwrap();
    let tensors: Vec<(&str, Vec<u64>)> = vec![
        ("small", vec![100]),
        ("pad", vec![u64::try_from(CHUNK - 2 * PAGE).unwrap()]),
        ("across", vec![u64::try_from(2 * PAGE).unwrap()]),
        ("pad-again", vec![u64::try_from(CHUNK - PAGE).unwrap()]),
        ("at-chunk", vec![1000]),
        ("empty", vec![0, 4]),
        ("last", vec![u64::try_from(PAGE).unwrap()]),
    ];
    let contents: Vec<Vec<u8>> = tensors
        .iter()
        .enumerate()
        .map(|(seed, (_, shape))| {
            let len = usize::try_from(shape.iter().product::<u64>()).unwrap();
            (0..len)
                .map(|i| u8::try_from((i + seed * 7) % 251).unwrap())
                .collect()
        })
        .collect();
    let data: Vec<TensorData<'_>> = tensors
        .iter()
        .zip(&contents)
        .map(|((name, shape), bytes)| TensorData {
            name,
            dtype: Dtype::U8,
            shape,
            layout: Layout::RowMajor,
            scheme: Scheme::U8,
            bytes,
        })
        .collect();
    writer.add_file("t.bin", &[], &data).unwrap();
    writer.finish().unwrap();
    let model = Model::open(&[&root]).unwrap();
    let file = model.table().index_of("t.bin").unwrap();
    let size = model.table().entries()[file].size;
    assert_eq!(size, u64::try_from(2 * CHUNK + 2 * PAGE).unwrap());
    let chunk = u64::try_from(CHUNK).unwrap();
    let location = |chunks, span, within| TensorLocation {
        file,
        chunks,
        span,
        within,
    };
    for (name, expected) in [
        ("small", location(0..1, 0..chunk, 0..100)),
        (
            "across",
            location(0..2, 0..2 * chunk, CHUNK - PAGE..CHUNK + PAGE),
        ),
        ("at-chunk", location(2..3, 2 * chunk..size, 0..1000)),
        ("empty", location(2..2, 2 * chunk..2 * chunk, 0..0)),
        ("last", location(2..3, 2 * chunk..size, PAGE..2 * PAGE)),
    ] {
        assert_eq!(model.tensor_location(name), Some(expected), "{name}");
    }
    assert_eq!(model.tensor_location("nothing"), None);

    let reader = model.reader(CacheMode::Bypass).unwrap();
    let open = reader.open(file).unwrap();
    for ((name, _), bytes) in tensors.iter().zip(&contents) {
        let location = model.tensor_location(name).unwrap();
        assert!(location.within.start.is_multiple_of(PAGE), "{name}");
        let mut buffer =
            AlignedBuf::new(usize::try_from(location.span.end - location.span.start).unwrap());
        open.read_chunks(location.chunks, &mut buffer).unwrap();
        assert_eq!(&buffer[location.within], &bytes[..], "{name}");
    }
    assert_eq!(open.chunks_covering(&(chunk..chunk)), 1..1);
    assert_eq!(open.chunks_covering(&(chunk - 1..chunk)), 0..1);
    assert_eq!(open.chunks_covering(&(chunk..chunk + 1)), 1..2);
    open.read_chunks(1..1, &mut []).unwrap();
    // Around the page cache, the buffer must be aligned still.
    let small = model.tensor_location("small").unwrap();
    let mut buffer = AlignedBuf::new(CHUNK + 1);
    let error = open
        .read_chunks(small.chunks, &mut buffer[1..])
        .unwrap_err();
    assert!(
        matches!(&error, Error::Io { source, .. } if source.kind() == std::io::ErrorKind::InvalidInput),
        "{error:?}"
    );
}

#[test]
fn adds_a_hash_table() {
    let dir = tempfile::tempdir().unwrap();
    write_tiny(dir.path(), false);
    let path = dir.path().join(Algorithm::Blake3.table_path());
    let original = fs::read(&path).unwrap();
    fs::remove_file(&path).unwrap();

    let model = Model::open(&[dir.path()]).unwrap();
    assert!(model.hash_table(Algorithm::Blake3).is_none());
    let error = model
        .add_table(
            Algorithm::Blake3,
            &dir.path().join("x"),
            CacheMode::Use,
            2,
            &|_| {},
        )
        .unwrap_err();
    assert!(error.to_string().contains("isn't one of"), "{error}");
    let added = model
        .add_table(Algorithm::Blake3, dir.path(), CacheMode::Bypass, 2, &|_| {})
        .unwrap();
    assert_eq!(added, path);
    assert_eq!(fs::read(&path).unwrap(), original);

    let model = Model::open(&[dir.path()]).unwrap();
    assert!(model.hash_table(Algorithm::Blake3).is_some());
    let error = model
        .add_table(Algorithm::Blake3, dir.path(), CacheMode::Use, 2, &|_| {})
        .unwrap_err();
    assert!(error.to_string().contains("already"), "{error}");
}

#[test]
fn checks_hash_tables() {
    let dir = tempfile::tempdir().unwrap();
    write_tiny(dir.path(), false);
    let path = dir.path().join(Algorithm::Sha512.table_path());
    let mut bytes = fs::read(&path).unwrap();
    bytes[140] ^= 1;
    fs::write(&path, &bytes).unwrap();
    match Model::open(&[dir.path()]) {
        Err(Error::HashTable { source, .. }) => {
            assert!(source.to_string().contains("corrupt"), "{source}");
        }
        other => panic!("{other:?}"),
    }
    fs::remove_file(&path).unwrap();

    // Without the fast tables a model opens, but can't be read.
    for algorithm in [Algorithm::Gxhash3, Algorithm::Xxh3] {
        let _ = fs::remove_file(dir.path().join(algorithm.table_path()));
    }
    let model = Model::open(&[dir.path()]).unwrap();
    assert!(model.content_id().is_some());
    let error = model.reader(CacheMode::Use).unwrap_err();
    assert!(matches!(error, Error::NoVerifier(_)), "{error}");
    let error = model
        .verify(&[Algorithm::Xxh3], CacheMode::Use, 1, &|_| {})
        .unwrap_err();
    assert!(matches!(error, Error::NoTable(Algorithm::Xxh3)), "{error}");

    // The content id needs the sha3-512 table.
    fs::remove_file(dir.path().join(Algorithm::Sha3.table_path())).unwrap();
    let model = Model::open(&[dir.path()]).unwrap();
    assert!(model.content_id().is_none());
}

// Flips a bit of a table's digests, which its checksum then doesn't match.
fn corrupt(path: &Path) {
    let mut bytes = fs::read(path).unwrap();
    bytes[200] ^= 1;
    fs::write(path, &bytes).unwrap();
}

// The table `bytes` as a table of version 2 would be, its checksum holding.
fn newer(bytes: &[u8]) -> Vec<u8> {
    let mut bytes = bytes.to_vec();
    bytes[8..12].copy_from_slice(&2u32.to_le_bytes());
    let body = bytes.len() - 16;
    let checksum = xxhash_rust::xxh3::xxh3_128(&bytes[..body]).to_be_bytes();
    bytes[body..].copy_from_slice(&checksum);
    bytes
}

// Every table must be valid and the same in every root, whichever it is: one
// that isn't keeps the model from opening, even one reads never use.
#[test]
fn opens_no_model_with_a_bad_table() {
    let dir = tempfile::tempdir().unwrap();
    let a = dir.path().join("a");
    write_tiny(&a, false);
    for algorithm in algorithms() {
        let path = a.join(algorithm.table_path());
        let original = fs::read(&path).unwrap();
        corrupt(&path);
        match Model::open(&[&a]) {
            Err(Error::HashTable {
                path: found,
                source,
            }) => {
                assert_eq!(found, path);
                assert_eq!(source.algorithm, algorithm);
                assert!(source.message.contains("corrupt"), "{source}");
                assert_eq!(source.newer_version, None);
            }
            other => panic!("{algorithm}: {other:?}"),
        }
        fs::write(&path, &original).unwrap();
    }

    let path = a.join(Algorithm::Sha256.table_path());
    let original = fs::read(&path).unwrap();
    fs::write(&path, newer(&original)).unwrap();
    match Model::open(&[&a]) {
        Err(Error::HashTable { source, .. }) => {
            assert_eq!(source.newer_version, Some(2));
            assert!(source.message.contains("newer than this build"), "{source}");
        }
        other => panic!("{other:?}"),
    }
    fs::write(&path, &original).unwrap();

    let b = dir.path().join("b");
    copy(&a, &b, |_| true);
    Model::open(&[&a, &b]).unwrap();
    let other = b.join(Algorithm::Sha256.table_path());
    corrupt(&other);
    match Model::open(&[&a, &b]) {
        Err(Error::TablesDiffer {
            algorithm,
            first,
            other: found,
        }) => {
            assert_eq!(algorithm, Algorithm::Sha256);
            assert_eq!(first, path);
            assert_eq!(found, other);
        }
        other => panic!("{other:?}"),
    }
}

// A table added later vouches only for bytes the model's tables vouch for:
// a payload changed behind the model's back fails the addition, and no
// table is written.
#[test]
fn adds_no_table_of_unverified_bytes() {
    let dir = tempfile::tempdir().unwrap();
    write_tiny(dir.path(), false);
    let sha3 = dir.path().join(Algorithm::Sha3.table_path());
    fs::remove_file(&sha3).unwrap();
    let path = "layers/00/experts/01.bin";
    let full = dir.path().join(path);
    let mut contents = fs::read(&full).unwrap();
    contents[100] ^= 1;
    fs::write(&full, &contents).unwrap();

    let model = Model::open(&[dir.path()]).unwrap();
    let error = model
        .add_table(Algorithm::Sha3, dir.path(), CacheMode::Use, 3, &|_| {})
        .unwrap_err();
    match &error {
        Error::Mismatch {
            path: found,
            algorithm,
            ..
        } => {
            assert_eq!(found, path);
            assert_eq!(*algorithm, model.verifier().unwrap());
        }
        other => panic!("{other:?}"),
    }
    assert!(!sha3.exists());
    let left: Vec<_> = fs::read_dir(dir.path().join("hashes"))
        .unwrap()
        .map(|entry| entry.unwrap().file_name().into_string().unwrap())
        .filter(|name| name.contains(".partial"))
        .collect();
    assert_eq!(left, Vec::<String>::new());
}

// A table's file appears whole or not at all: a write cut short leaves a
// file of its own, which opening ignores and a later addition leaves as it
// is, writing into one of its own.
#[test]
fn adds_tables_whole() {
    let dir = tempfile::tempdir().unwrap();
    write_tiny(dir.path(), false);
    let path = dir.path().join(Algorithm::Blake3.table_path());
    let original = fs::read(&path).unwrap();
    fs::remove_file(&path).unwrap();
    let partial = dir
        .path()
        .join(format!("{}.partial-1-0", Algorithm::Blake3.table_path()));
    fs::write(&partial, &original[..100]).unwrap();

    let model = Model::open(&[dir.path()]).unwrap();
    assert!(model.hash_table(Algorithm::Blake3).is_none());
    model
        .add_table(Algorithm::Blake3, dir.path(), CacheMode::Use, 2, &|_| {})
        .unwrap();
    assert_eq!(fs::read(&path).unwrap(), original);
    assert_eq!(fs::read(&partial).unwrap(), &original[..100]);
    let left = fs::read_dir(dir.path().join("hashes"))
        .unwrap()
        .filter(|entry| {
            let name = entry.as_ref().unwrap().file_name();
            name.to_string_lossy().contains(".partial")
        })
        .count();
    assert_eq!(left, 1);
}

// Nothing of a model with no table this build can compute is used, the
// manifest included.
#[test]
fn opens_no_model_it_cant_verify() {
    let dir = tempfile::tempdir().unwrap();
    write_tiny(dir.path(), false);
    for algorithm in Algorithm::ALL {
        let _ = fs::remove_file(dir.path().join(algorithm.table_path()));
    }
    let error = Model::open(&[dir.path()]).unwrap_err();
    assert!(matches!(error, Error::Unverifiable(_)), "{error}");
}
