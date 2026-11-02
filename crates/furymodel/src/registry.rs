// SPDX-FileCopyrightText: 2026 Mika Tammi
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The registries of the specification's section 4: dtypes, layouts and
//! schemes. Ids only ever get added, so a manifest naming one this library
//! doesn't know fails to parse.

use serde::{Deserialize, Serialize};

/// The registries' version, which section 4 of the specification lists.
pub const REGISTRY_VERSION: u32 = 1;

/// A tensor's element type. The registry only grows (section 4), so a
/// match on it needs an arm for the ones a later version adds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum Dtype {
    /// bfloat16.
    Bf16,
    /// IEEE 754 binary16.
    F16,
    /// IEEE 754 binary32.
    F32,
    /// IEEE 754 binary64.
    F64,
    /// Signed 8-bit integer.
    I8,
    /// Unsigned 8-bit integer.
    U8,
    /// Signed 16-bit integer.
    I16,
    /// Unsigned 16-bit integer.
    U16,
    /// Signed 32-bit integer.
    I32,
    /// Unsigned 32-bit integer.
    U32,
    /// Signed 64-bit integer.
    I64,
    /// Unsigned 64-bit integer.
    U64,
}

impl Dtype {
    /// Every dtype, in the registry's order.
    pub const ALL: [Self; 12] = [
        Self::Bf16,
        Self::F16,
        Self::F32,
        Self::F64,
        Self::I8,
        Self::U8,
        Self::I16,
        Self::U16,
        Self::I32,
        Self::U32,
        Self::I64,
        Self::U64,
    ];

    /// The dtype's id in a manifest.
    #[must_use]
    pub const fn id(self) -> &'static str {
        match self {
            Self::Bf16 => "bf16",
            Self::F16 => "f16",
            Self::F32 => "f32",
            Self::F64 => "f64",
            Self::I8 => "i8",
            Self::U8 => "u8",
            Self::I16 => "i16",
            Self::U16 => "u16",
            Self::I32 => "i32",
            Self::U32 => "u32",
            Self::I64 => "i64",
            Self::U64 => "u64",
        }
    }

    /// An element's size in bytes.
    #[must_use]
    pub const fn size(self) -> u64 {
        match self {
            Self::I8 | Self::U8 => 1,
            Self::Bf16 | Self::F16 | Self::I16 | Self::U16 => 2,
            Self::F32 | Self::I32 | Self::U32 => 4,
            Self::F64 | Self::I64 | Self::U64 => 8,
        }
    }
}

impl std::fmt::Display for Dtype {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.id())
    }
}

/// How a tensor's logical indices map to the order of its elements; a
/// registry that only grows, as [`Dtype`]'s.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum Layout {
    /// C order, of any rank.
    RowMajor,
    /// A linear layer's weight `W[out][in]`, row-major: rank 2.
    OutIn,
}

impl Layout {
    /// The layout's id in a manifest.
    #[must_use]
    pub const fn id(self) -> &'static str {
        match self {
            Self::RowMajor => "row_major",
            Self::OutIn => "out_in",
        }
    }

    /// Checks that a shape is one the layout allows.
    ///
    /// # Errors
    ///
    /// Describes the violation, such as an `out_in` tensor whose rank isn't 2.
    pub fn check_shape(self, shape: &[u64]) -> Result<(), String> {
        match self {
            Self::RowMajor => Ok(()),
            Self::OutIn if shape.len() == 2 => Ok(()),
            Self::OutIn => Err(format!("layout out_in needs a rank-2 shape, not {shape:?}")),
        }
    }

    /// The position, counted in elements, of the element at `index` in a
    /// tensor of `shape`, or `None` if the index is out of range or the
    /// layout doesn't allow the shape.
    #[must_use]
    pub fn element(self, shape: &[u64], index: &[u64]) -> Option<u64> {
        if self.check_shape(shape).is_err() || shape.len() != index.len() {
            return None;
        }
        // Both layouts are C order; out_in only fixes the rank.
        shape
            .iter()
            .zip(index)
            .try_fold(0u64, |position, (&extent, &i)| {
                if i < extent {
                    position.checked_mul(extent)?.checked_add(i)
                } else {
                    None
                }
            })
    }
}

impl std::fmt::Display for Layout {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.id())
    }
}

/// How a tensor's elements are encoded as bytes. Registry version 1 has
/// the plain schemes only, one per dtype and named after it: every element
/// in its dtype, little-endian. Later versions add others, block
/// quantisation say, as [`Dtype`]'s registry grows.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum Scheme {
    /// Plain bfloat16 elements.
    Bf16,
    /// Plain binary16 elements.
    F16,
    /// Plain binary32 elements.
    F32,
    /// Plain binary64 elements.
    F64,
    /// Plain signed bytes.
    I8,
    /// Plain unsigned bytes.
    U8,
    /// Plain signed 16-bit integers.
    I16,
    /// Plain unsigned 16-bit integers.
    U16,
    /// Plain signed 32-bit integers.
    I32,
    /// Plain unsigned 32-bit integers.
    U32,
    /// Plain signed 64-bit integers.
    I64,
    /// Plain unsigned 64-bit integers.
    U64,
}

impl Scheme {
    /// The plain scheme of a dtype.
    #[must_use]
    pub const fn plain(dtype: Dtype) -> Self {
        match dtype {
            Dtype::Bf16 => Self::Bf16,
            Dtype::F16 => Self::F16,
            Dtype::F32 => Self::F32,
            Dtype::F64 => Self::F64,
            Dtype::I8 => Self::I8,
            Dtype::U8 => Self::U8,
            Dtype::I16 => Self::I16,
            Dtype::U16 => Self::U16,
            Dtype::I32 => Self::I32,
            Dtype::U32 => Self::U32,
            Dtype::I64 => Self::I64,
            Dtype::U64 => Self::U64,
        }
    }

    /// The dtype whose elements the scheme encodes.
    #[must_use]
    pub const fn dtype(self) -> Dtype {
        match self {
            Self::Bf16 => Dtype::Bf16,
            Self::F16 => Dtype::F16,
            Self::F32 => Dtype::F32,
            Self::F64 => Dtype::F64,
            Self::I8 => Dtype::I8,
            Self::U8 => Dtype::U8,
            Self::I16 => Dtype::I16,
            Self::U16 => Dtype::U16,
            Self::I32 => Dtype::I32,
            Self::U32 => Dtype::U32,
            Self::I64 => Dtype::I64,
            Self::U64 => Dtype::U64,
        }
    }

    /// The scheme's id in a manifest.
    #[must_use]
    pub const fn id(self) -> &'static str {
        self.dtype().id()
    }

    /// The size in bytes of a tensor of `shape`, or `None` if it overflows.
    #[must_use]
    pub fn size(self, shape: &[u64]) -> Option<u64> {
        shape.iter().try_fold(self.dtype().size(), |size, &extent| {
            size.checked_mul(extent)
        })
    }
}

impl std::fmt::Display for Scheme {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.id())
    }
}

#[cfg(test)]
mod tests {
    use super::{Dtype, Layout, Scheme};

    #[test]
    fn ids_round_trip_through_json() {
        for dtype in Dtype::ALL {
            let json = serde_json::to_string(&dtype).unwrap();
            assert_eq!(json, format!("\"{}\"", dtype.id()));
            assert_eq!(serde_json::from_str::<Dtype>(&json).unwrap(), dtype);
            let scheme = Scheme::plain(dtype);
            assert_eq!(serde_json::to_string(&scheme).unwrap(), json);
            assert_eq!(scheme.dtype(), dtype);
        }
        for layout in [Layout::RowMajor, Layout::OutIn] {
            let json = serde_json::to_string(&layout).unwrap();
            assert_eq!(json, format!("\"{}\"", layout.id()));
        }
        assert!(serde_json::from_str::<Dtype>("\"bfloat16\"").is_err());
        assert!(serde_json::from_str::<Layout>("\"in_out\"").is_err());
    }

    #[test]
    fn sizes() {
        assert_eq!(Scheme::Bf16.size(&[2816, 704]), Some(3_964_928));
        assert_eq!(Scheme::F32.size(&[]), Some(4));
        assert_eq!(Scheme::U8.size(&[0, 7]), Some(0));
        assert_eq!(Scheme::U64.size(&[u64::MAX, 2]), None);
    }

    // Golden vectors of the layouts in the specification's section 4.2.
    #[test]
    fn layout_golden_vectors() {
        assert_eq!(Layout::RowMajor.element(&[2, 3, 4], &[1, 2, 3]), Some(23));
        assert_eq!(Layout::RowMajor.element(&[2, 3, 4], &[0, 1, 0]), Some(4));
        assert_eq!(Layout::RowMajor.element(&[], &[]), Some(0));
        assert_eq!(Layout::OutIn.element(&[3, 5], &[2, 4]), Some(14));
        assert_eq!(Layout::OutIn.element(&[3, 5], &[1, 0]), Some(5));
        assert_eq!(Layout::OutIn.element(&[3, 5], &[3, 0]), None);
        assert_eq!(Layout::OutIn.element(&[15], &[4]), None);
        assert!(Layout::OutIn.check_shape(&[2816, 704]).is_ok());
        assert!(Layout::OutIn.check_shape(&[2816]).is_err());
    }
}
