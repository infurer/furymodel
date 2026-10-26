// SPDX-FileCopyrightText: 2026 Mika Tammi
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Delivery parts ([the specification](crate::spec)'s section 7): roots
//! made by splitting a model so that each fits a size limit, each described
//! by its `part.json`.

use std::fs;
use std::io::{self, Write as _};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::{Error, IoContext as _, Result};
use crate::hash::{CHUNK_SIZE, chunk_count};
use crate::io::{AlignedBuf, CacheMode};
use crate::manifest::{HASHES_DIR, MANIFEST_PATH, PART_PATH, check_unique_keys};
use crate::model::{Model, Reader};
use crate::writer::{place_new, sync_dir, write_new};

/// A `part.json`'s `format`.
pub const PART_FORMAT: &str = "furymodel-part";
/// The version of `part.json` this library reads and writes.
pub const PART_VERSION: u32 = 1;

/// A delivery part's description.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Part {
    /// Always [`PART_FORMAT`].
    pub format: String,
    /// Always [`PART_VERSION`].
    pub format_version: u32,
    /// The model's content id, in hex.
    pub content_id: String,
    /// This part's number, from 1.
    pub part: u32,
    /// The number of parts.
    pub parts: u32,
    /// The part's files, as `[start, end)` ranges of file-table indices.
    pub files: Vec<[u64; 2]>,
}

impl Part {
    /// Parses and checks a `part.json` of a model of `files` files.
    ///
    /// # Errors
    ///
    /// If it isn't a part of version 1, an object has a key twice, or its
    /// numbers or ranges don't fit.
    pub fn parse(bytes: &[u8], files: usize) -> Result<Self, String> {
        check_unique_keys(bytes).map_err(|error| error.to_string())?;
        let part: Self = serde_json::from_slice(bytes).map_err(|error| error.to_string())?;
        if part.format != PART_FORMAT || part.format_version != PART_VERSION {
            return Err(format!(
                "not a part of version {PART_VERSION}: format {:?} version {}",
                part.format, part.format_version
            ));
        }
        if part.part == 0 || part.part > part.parts {
            return Err(format!(
                "part {} of {} doesn't exist",
                part.part, part.parts
            ));
        }
        let mut end = 0;
        for &[start, stop] in &part.files {
            if start < end || start >= stop || stop > files as u64 {
                return Err(format!(
                    "the range [{start}, {stop}) isn't after the one before, is empty, or \
                     reaches past the {files} files"
                ));
            }
            end = stop;
        }
        Ok(part)
    }

    /// The part as its `part.json` holds it.
    ///
    /// # Panics
    ///
    /// Never: a part serializes to JSON.
    #[must_use]
    pub fn to_json(&self) -> Vec<u8> {
        let mut json = serde_json::to_vec_pretty(self).expect("a part serializes");
        json.push(b'\n');
        json
    }

    /// Whether the file of `index` is the part's.
    #[must_use]
    pub fn contains(&self, index: usize) -> bool {
        let index = index as u64;
        self.files
            .iter()
            .any(|&[start, stop]| (start..stop).contains(&index))
    }
}

/// What a part's `part.json` takes at most, which a part's size limit
/// allows for.
const PART_JSON_ALLOWANCE: u64 = 16 * 1024;

/// Splits `model` into delivery parts of at most `max_part_size` bytes of
/// files each, written as the directories `part-01`, `part-02`, … of `out`,
/// which must not exist or be empty. Every part holds the manifest, every
/// hash table and its `part.json`, and the files of the file table go into
/// parts in order. The files are hard links into the model's roots where
/// the file system allows it, copies otherwise, each chunk of a copy read
/// around the page cache and verified, and never over a file. A part gets
/// its manifest last, once the rest of it is durable, and always as such a
/// copy, put in place whole: a directory with a manifest is a whole part,
/// with the manifest the model was opened with.
///
/// # Errors
///
/// If the model has no sha2-512 table, for the content id `part.json`
/// names, or no table to verify chunks with ([`Error::NoVerifier`]); a
/// file doesn't fit a part next to the manifest and the hash
/// tables; `out` isn't empty; a file doesn't match the model's table as it
/// is copied ([`Error::Mismatch`]); or writing fails.
pub fn split(model: &Model, max_part_size: u64, out: &Path) -> Result<Vec<PathBuf>> {
    let content_id = model.content_id().ok_or_else(|| {
        Error::Usage("splitting needs the sha2-512 table, for the parts' content id".to_owned())
    })?;
    let table = model.table();
    let manifest = table
        .index_of(MANIFEST_PATH)
        .ok_or_else(|| Error::Usage("the file table has no manifest".to_owned()))?;
    let tables: Vec<(String, Vec<u8>)> = model
        .hash_tables()
        .iter()
        .map(|table| (table.algorithm().table_path(), table.encode()))
        .collect();
    let overhead = table.entries()[manifest].size
        + tables
            .iter()
            .map(|(_, bytes)| bytes.len() as u64)
            .sum::<u64>()
        + PART_JSON_ALLOWANCE;
    let capacity = max_part_size.checked_sub(overhead).ok_or_else(|| {
        Error::Usage(format!(
            "a part of {max_part_size} bytes doesn't even hold the manifest and the hash \
             tables, {overhead} bytes with part.json"
        ))
    })?;

    // Contiguous ranges of the file table; the manifest counts as overhead.
    let mut ranges = Vec::new();
    let (mut start, mut used) = (0, 0);
    for (index, entry) in table.entries().iter().enumerate() {
        let size = if index == manifest { 0 } else { entry.size };
        if size > capacity {
            return Err(Error::Usage(format!(
                "{}, of {size} bytes, doesn't fit a part of {max_part_size} bytes next to the \
                 manifest and the hash tables",
                entry.path
            )));
        }
        if used + size > capacity {
            ranges.push(start..index);
            (start, used) = (index, 0);
        }
        used += size;
    }
    ranges.push(start..table.len());

    fs::create_dir_all(out).at(out)?;
    if fs::read_dir(out).at(out)?.next().is_some() {
        return Err(Error::Usage(format!("{} isn't empty", out.display())));
    }
    let reader = model.reader(CacheMode::Bypass)?;
    let width = ranges.len().to_string().len().max(2);
    let parts =
        u32::try_from(ranges.len()).map_err(|_| Error::Usage("too many parts".to_owned()))?;
    let mut parts_written = Vec::with_capacity(ranges.len());
    for (number, range) in (1..=parts).zip(&ranges) {
        let directory = out.join(format!("part-{number:0width$}"));
        let hashes = directory.join(HASHES_DIR);
        fs::create_dir_all(&hashes).at(&hashes)?;
        for (path, bytes) in &tables {
            write_new(&directory.join(path), bytes)?;
        }
        let mut directories = std::collections::BTreeSet::new();
        for index in range.clone().filter(|&index| index != manifest) {
            let path = &table.entries()[index].path;
            link_or_copy(model, &reader, index, &directory.join(path))?;
            for (end, _) in path.match_indices('/') {
                directories.insert(&path[..end]);
            }
        }
        let part = Part {
            format: PART_FORMAT.to_owned(),
            format_version: PART_VERSION,
            content_id: content_id.to_string(),
            part: number,
            parts,
            files: vec![[range.start as u64, range.end as u64]],
        };
        write_new(&directory.join(PART_PATH), &part.to_json())?;
        for path in directories.iter().rev() {
            sync_dir(&directory.join(path))?;
        }
        copy_verified(&reader, manifest, &directory.join(MANIFEST_PATH))?;
        sync_dir(&directory)?;
        sync_dir(out)?;
        parts_written.push(directory);
    }
    Ok(parts_written)
}

// Puts the model's file of `index` into a part at `to`, durably and never
// over a file: hard-linked where the file system allows it, its bytes
// synced when it was written, or copied as copy_verified does.
fn link_or_copy(model: &Model, reader: &Reader<&Model>, index: usize, to: &Path) -> Result<()> {
    if let Some(parent) = to.parent() {
        fs::create_dir_all(parent).at(parent)?;
    }
    match fs::hard_link(model.path(index), to) {
        Ok(()) => Ok(()),
        Err(source) if source.kind() == io::ErrorKind::AlreadyExists => Err(Error::Io {
            path: to.to_path_buf(),
            source,
        }),
        // Another file system, or one without hard links.
        Err(_) => copy_verified(reader, index, to),
    }
}

// Copies the model's file of `index` to `to` through `reader`, every chunk
// verified as it is read, into a file of the copy's own that is synced and
// renamed to `to` only if no file has its name.
fn copy_verified(reader: &Reader<&Model>, index: usize, to: &Path) -> Result<()> {
    let file = reader.open(index)?;
    let mut buffer = AlignedBuf::new(usize::try_from(CHUNK_SIZE).unwrap_or(0));
    place_new(to, |out, partial| {
        for chunk in 0..chunk_count(file.size()) {
            let span = file.span(&(chunk..chunk + 1));
            let bytes = &mut buffer[..usize::try_from(span.end - span.start).unwrap_or(0)];
            file.read_chunks(chunk..chunk + 1, bytes)?;
            out.write_all(bytes).at(partial)?;
        }
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::io;
    use std::os::unix::fs::MetadataExt as _;
    use std::path::Path;

    use super::{Part, copy_verified, link_or_copy, split};
    use crate::error::Error;
    use crate::io::CacheMode;
    use crate::manifest::MANIFEST_PATH;
    use crate::model::Model;
    use crate::testing::write_tiny;

    fn json(part: u32, parts: u32, files: &[[u64; 2]]) -> Vec<u8> {
        Part {
            format: "furymodel-part".to_owned(),
            format_version: 1,
            content_id: "00".repeat(64),
            part,
            parts,
            files: files.to_vec(),
        }
        .to_json()
    }

    #[test]
    fn parses_and_checks() {
        let part = Part::parse(&json(2, 3, &[[0, 4], [6, 9]]), 10).unwrap();
        assert!(part.contains(3) && part.contains(6) && !part.contains(4) && !part.contains(9));
        for (bytes, needle) in [
            (json(0, 3, &[[0, 1]]), "doesn't exist"),
            (json(4, 3, &[[0, 1]]), "doesn't exist"),
            (json(1, 1, &[[3, 3]]), "is empty"),
            (json(1, 1, &[[2, 4], [3, 5]]), "isn't after"),
            (json(1, 1, &[[0, 11]]), "reaches past"),
        ] {
            let error = Part::parse(&bytes, 10).unwrap_err();
            assert!(error.contains(needle), "{error}");
        }
        let mut other: serde_json::Value = serde_json::from_slice(&json(1, 1, &[])).unwrap();
        other["format_version"] = serde_json::json!(2);
        assert!(Part::parse(&serde_json::to_vec(&other).unwrap(), 10).is_err());

        let twice = String::from_utf8(json(1, 2, &[[0, 1]])).unwrap().replacen(
            "\"part\": 1,",
            "\"part\": 1, \"part\": 2,",
            1,
        );
        let error = Part::parse(twice.as_bytes(), 10).unwrap_err();
        assert!(error.contains("twice"), "{error}");
    }

    #[test]
    fn splits_a_model_into_parts_that_open_together() {
        let dir = tempfile::tempdir().unwrap();
        let written = write_tiny(&dir.path().join("model"), false);
        let model = Model::open(&[dir.path().join("model")]).unwrap();
        let max = 2_200_000;
        let parts = split(&model, max, &dir.path().join("parts")).unwrap();
        assert!(parts.len() >= 2, "{parts:?}");
        for part in &parts {
            let mut size = 0;
            let mut stack = vec![part.clone()];
            while let Some(directory) = stack.pop() {
                for entry in fs::read_dir(directory).unwrap() {
                    let path = entry.unwrap().path();
                    if path.is_dir() {
                        stack.push(path);
                    } else {
                        size += fs::metadata(&path).unwrap().len();
                    }
                }
            }
            assert!(size <= max, "{} has {size} bytes", part.display());
        }
        let reopened = Model::open(&parts).unwrap();
        assert_eq!(reopened.content_id(), written.content_id);
        assert_eq!(reopened.parts().len(), parts.len());
        // The files hard-linked here, the manifests copied.
        let inode = |path: &Path| fs::metadata(path).unwrap().ino();
        let table = model.table();
        let manifest = table.index_of(MANIFEST_PATH).unwrap();
        for part in &parts {
            assert_ne!(
                inode(&part.join(MANIFEST_PATH)),
                inode(&model.path(manifest))
            );
            assert_eq!(
                fs::read(part.join(MANIFEST_PATH)).unwrap(),
                fs::read(model.path(manifest)).unwrap()
            );
        }
        let linked = (0..table.len())
            .find(|&index| {
                index != manifest && parts[0].join(&table.entries()[index].path).exists()
            })
            .unwrap();
        assert_eq!(
            inode(&parts[0].join(&table.entries()[linked].path)),
            inode(&model.path(linked))
        );

        let error = Model::open(&parts[1..]).unwrap_err();
        assert!(
            error.to_string().contains("delivery parts missing: 1 (of"),
            "{error}"
        );

        let error = split(&model, 1_000_000, &dir.path().join("small")).unwrap_err();
        assert!(error.to_string().contains("global/big.bin"), "{error}");
        let error = split(&model, max, &dir.path().join("parts")).unwrap_err();
        assert!(error.to_string().contains("isn't empty"), "{error}");
    }

    // A copy, as a file goes into a part where it can't be linked, is put in
    // place whole: verified chunk by chunk as it is read, never over a
    // file, and leaving nothing behind when it fails.
    #[test]
    fn copies_files_whole_or_not_at_all() {
        let dir = tempfile::tempdir().unwrap();
        write_tiny(&dir.path().join("model"), false);
        let model = Model::open(&[dir.path().join("model")]).unwrap();
        let reader = model.reader(CacheMode::Bypass).unwrap();
        let out = dir.path().join("out");
        fs::create_dir(&out).unwrap();
        let big = model.table().index_of("global/big.bin").unwrap();
        let manifest = model.table().index_of(MANIFEST_PATH).unwrap();
        for (index, name) in [(big, "big.bin"), (manifest, "manifest.json")] {
            copy_verified(&reader, index, &out.join(name)).unwrap();
            assert_eq!(
                fs::read(out.join(name)).unwrap(),
                fs::read(model.path(index)).unwrap()
            );
        }

        let taken = out.join("taken");
        fs::write(&taken, b"another file").unwrap();
        for error in [
            copy_verified(&reader, big, &taken).unwrap_err(),
            link_or_copy(&model, &reader, big, &taken).unwrap_err(),
        ] {
            assert!(
                matches!(&error, Error::Io { source, .. }
                    if source.kind() == io::ErrorKind::AlreadyExists),
                "{error:?}"
            );
        }
        assert_eq!(fs::read(&taken).unwrap(), b"another file");

        let mut contents = fs::read(model.path(big)).unwrap();
        contents[5] ^= 0x40;
        fs::write(model.path(big), &contents).unwrap();
        let error = copy_verified(&reader, big, &out.join("corrupt.bin")).unwrap_err();
        assert!(
            matches!(error, Error::Mismatch { chunk: Some(0), .. }),
            "{error:?}"
        );

        let mut names: Vec<String> = fs::read_dir(&out)
            .unwrap()
            .map(|entry| entry.unwrap().file_name().into_string().unwrap())
            .collect();
        names.sort();
        assert_eq!(names, ["big.bin", "manifest.json", "taken"]);
    }
}
