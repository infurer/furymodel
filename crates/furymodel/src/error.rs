// SPDX-FileCopyrightText: 2026 Mika Tammi
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The library's error type.

use std::io;
use std::path::PathBuf;

use crate::hash::{Algorithm, TableError};
use crate::manifest::ManifestError;

/// What can go wrong writing, opening or reading a model.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// An operation on a file failed.
    #[error("{}", path.display())]
    Io {
        /// The file.
        path: PathBuf,
        /// What failed.
        #[source]
        source: io::Error,
    },
    /// A manifest is unsupported or invalid.
    #[error("{}", path.display())]
    Manifest {
        /// The manifest's file.
        path: PathBuf,
        /// What's wrong with it.
        #[source]
        source: ManifestError,
    },
    /// A hash table is invalid, or isn't the model's.
    #[error("{}", path.display())]
    HashTable {
        /// The table's file.
        path: PathBuf,
        /// What's wrong with it.
        #[source]
        source: TableError,
    },
    /// A directory given as a root holds no manifest.
    #[error("{} has no furymodel.json: it isn't a root of a FuryModel model", .0.display())]
    NotARoot(PathBuf),
    /// Two roots hold different manifests: they are roots of different
    /// models.
    #[error(
        "the manifests of {} and {} differ: they are roots of different models",
        first.display(),
        other.display()
    )]
    ManifestsDiffer {
        /// The first root.
        first: PathBuf,
        /// The root whose manifest isn't the first's.
        other: PathBuf,
    },
    /// Two roots hold different bytes for the table of an algorithm, which
    /// must be the same in every root.
    #[error(
        "{} and {} differ: the {algorithm} table must be the same in every root",
        first.display(),
        other.display()
    )]
    TablesDiffer {
        /// The table's algorithm.
        algorithm: Algorithm,
        /// The table's file in the first root that has it.
        first: PathBuf,
        /// The file in a root whose bytes aren't the first's.
        other: PathBuf,
    },
    /// A file of the model in a root isn't a file of the manifest's size.
    #[error(
        "{}: {}; the manifest gives it {expected}",
        path.display(),
        found.map_or_else(|| "it isn't a file".to_owned(), |size| format!("it has {size} bytes"))
    )]
    WrongSize {
        /// The file in the root.
        path: PathBuf,
        /// Its size, or `None` if it isn't a regular file.
        found: Option<u64>,
        /// The size the manifest gives it.
        expected: u64,
    },
    /// Files of the model are in none of the roots.
    #[error(
        "the model is incomplete: {missing} of its {files} files are in none of the roots, among \
         them {}{}",
        examples.join(", "),
        missing_parts_note(missing_parts, *parts)
    )]
    Incomplete {
        /// How many files no root has.
        missing: usize,
        /// How many files the model has.
        files: usize,
        /// The paths of the first few of them.
        examples: Vec<String>,
        /// When the roots are delivery parts: the numbers of the parts none
        /// of the roots is, in order.
        missing_parts: Vec<u32>,
        /// When the roots are delivery parts: how many parts there are.
        parts: Option<u32>,
    },
    /// The roots don't make one model in some other way, such as delivery
    /// parts that don't go together.
    #[error("{0}")]
    Roots(String),
    /// A file's bytes don't match their digest, even when read once more.
    #[error(
        "{path}: {} doesn't match its {algorithm} digest",
        chunk.map_or_else(|| "the file".to_owned(), |chunk| format!("chunk {chunk}"))
    )]
    Mismatch {
        /// The file's path in the model.
        path: String,
        /// The chunk, or `None` for the file's digest.
        chunk: Option<u64>,
        /// The algorithm of the digest.
        algorithm: Algorithm,
    },
    /// The model has no table of the algorithm needed.
    #[error("the model has no {0} table")]
    NoTable(Algorithm),
    /// The model has no table this build can verify chunks with.
    #[error("the model has no hash table this build can verify chunks with: {0}")]
    NoVerifier(String),
    /// The model has no table this build can compute, so that none of its
    /// bytes, not even the manifest's, can be verified.
    #[error("the model can't be verified: {0}")]
    Unverifiable(String),
    /// The library was asked for something its interface rules out, such as
    /// chunks a file doesn't have.
    #[error("{0}")]
    Usage(String),
    /// A writer was used against the specification, such as twice with one
    /// path.
    #[error("{0}")]
    Write(String),
}

/// The library's result type.
pub type Result<T, E = Error> = std::result::Result<T, E>;

fn missing_parts_note(missing: &[u32], parts: Option<u32>) -> String {
    match parts {
        Some(parts) if !missing.is_empty() => {
            let numbers: Vec<String> = missing.iter().map(u32::to_string).collect();
            format!(
                "; delivery parts missing: {} (of {parts}); add their roots",
                numbers.join(", ")
            )
        }
        _ => String::new(),
    }
}

pub(crate) trait IoContext<T> {
    fn at(self, path: impl Into<PathBuf>) -> Result<T>;
}

impl<T> IoContext<T> for io::Result<T> {
    fn at(self, path: impl Into<PathBuf>) -> Result<T> {
        self.map_err(|source| Error::Io {
            path: path.into(),
            source,
        })
    }
}
