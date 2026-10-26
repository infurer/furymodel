// SPDX-FileCopyrightText: 2026 Mika Tammi
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Delivery parts ([the specification](crate::spec)'s section 7): roots
//! made by splitting a model so that each fits a size limit, each described
//! by its `part.json`.

use serde::{Deserialize, Serialize};

use crate::manifest::check_unique_keys;

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

#[cfg(test)]
mod tests {
    use super::Part;

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
}
