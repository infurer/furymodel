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

pub mod hash;
pub mod manifest;
pub mod registry;
#[doc = include_str!("../SPEC.md")]
pub mod spec {}

pub use hash::{Algorithm, ContentId, HashTable};
pub use manifest::{FileTable, Manifest, ManifestError};
pub use registry::{Dtype, Layout, Scheme};
