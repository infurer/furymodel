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
    /// The roots don't make one model, or not a complete one.
    #[error("{0}")]
    Roots(String),
    /// A file's bytes don't match their digest, even when read once more.
    #[error("{path}: {what} doesn't match its {algorithm} digest")]
    Mismatch {
        /// The file's path in the model.
        path: String,
        /// The chunk, or the whole file.
        what: String,
        /// The algorithm of the digest.
        algorithm: Algorithm,
    },
    /// The model has no table of the algorithm needed.
    #[error("the model has no {0} table")]
    NoTable(Algorithm),
    /// A writer was used against the specification, such as twice with one
    /// path.
    #[error("{0}")]
    Write(String),
}

/// The library's result type.
pub type Result<T, E = Error> = std::result::Result<T, E>;

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
