// SPDX-FileCopyrightText: 2026 Mika Tammi
// SPDX-License-Identifier: MIT OR Apache-2.0

//! FuryModel: a model file format laid out for streaming layers and experts
//! from disk, and its library. The format is specified in [`spec`], the
//! crate's `SPEC.md`.
//!
//! The reader, the writer and the verification are added module by module.

#[doc = include_str!("../SPEC.md")]
pub mod spec {}
