// SPDX-FileCopyrightText: 2026 Mika Tammi
// SPDX-License-Identifier: MIT OR Apache-2.0

//! FuryModel: a model file format laid out for streaming layers and experts
//! from disk, and its library. The format is specified in [`spec`], the
//! crate's `SPEC.md`; this crate's modules follow its sections.
//!
//! - [`manifest`]: the manifest, `furymodel.json`, and the file table it
//!   defines.
//! - [`registry`]: the dtypes, layouts and schemes a manifest may name.
//! - [`hash`]: the hash algorithms, their tables and the content id.
//! - [`writer`]: writing a model.

mod error;
pub mod hash;
pub mod manifest;
pub mod registry;
#[doc = include_str!("../SPEC.md")]
pub mod spec {}
#[cfg(test)]
mod testing;
pub mod writer;

pub use error::{Error, Result};
pub use hash::{Algorithm, ContentId, HashTable};
pub use manifest::{FileTable, Manifest, ManifestError};
pub use registry::{Dtype, Layout, Scheme};
pub use writer::Writer;
