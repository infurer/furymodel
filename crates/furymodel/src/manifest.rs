// SPDX-FileCopyrightText: 2026 Mika Tammi
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The manifest, `furymodel.json` (the specification's section 2), and the
//! file table it defines (section 5).

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt::Write as _;

use serde::{Deserialize, Serialize};

use crate::registry::{Dtype, Layout, Scheme};

/// The manifest's path in every root.
pub const MANIFEST_PATH: &str = "furymodel.json";
/// The path of a delivery part's description.
pub const PART_PATH: &str = "part.json";
/// The directory of the hash tables.
pub const HASHES_DIR: &str = "hashes";
/// A manifest's `format`.
pub const FORMAT: &str = "furymodel";
/// The format version this library reads and writes.
pub const FORMAT_VERSION: u32 = 1;
/// The alignment of payload file sizes and tensor offsets, in bytes.
pub const ALIGNMENT: u64 = 16 * 1024;
/// The most files a file table may hold, the manifest included (the
/// specification's section 2.9).
pub const MAX_FILES: u64 = 1 << 18;
/// The most tensors a model's payload files may hold in all.
pub const MAX_TENSORS: u64 = 1 << 20;
/// The longest path, in bytes.
pub const MAX_PATH_LEN: usize = 1024;
/// The longest tensor name, in bytes.
pub const MAX_NAME_LEN: usize = 1024;
/// The longest group name, in bytes.
pub const MAX_GROUP_NAME_LEN: usize = 64;
/// The most axes a group may have.
pub const MAX_AXES: usize = 16;
/// The most dimensions a tensor's shape may have.
pub const MAX_RANK: usize = 16;
/// The most a file table's description may take, every group instance's
/// own counted: see [`ENTRY_COST`], [`STRING_COST`] and [`NUMBER_COST`].
pub const MAX_DESCRIPTION: u64 = 256 << 20;
/// What a file, or a tensor, takes of the description.
pub const ENTRY_COST: u64 = 128;
/// What a path, tag, group name, role or tensor name takes of the
/// description besides its length in bytes.
pub const STRING_COST: u64 = 32;
/// What a dimension of a shape, or a value of an instance's axes, takes of
/// the description.
pub const NUMBER_COST: u64 = 8;

/// What's wrong with a manifest.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ManifestError {
    /// It isn't JSON, or not JSON of the manifest's schema.
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    /// It is the manifest of another format, or of another version of this
    /// one.
    #[error("{0}")]
    Unsupported(String),
    /// It breaks a rule of the specification.
    #[error("{0}")]
    Invalid(String),
}

type Result<T, E = ManifestError> = std::result::Result<T, E>;

fn invalid(message: impl Into<String>) -> ManifestError {
    ManifestError::Invalid(message.into())
}

/// A model's manifest.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    /// Always [`FORMAT`].
    pub format: String,
    /// Always [`FORMAT_VERSION`].
    pub format_version: u32,
    /// The display name.
    pub name: String,
    /// The runtime's architecture id.
    pub arch: String,
    /// The runtime's tensor-schema version for `arch`.
    pub schema_version: u32,
    /// The runtime's typed configuration of the model, a JSON object.
    pub config: serde_json::Value,
    /// Always [`ALIGNMENT`].
    pub alignment: u64,
    /// Payload files described one by one.
    pub files: Vec<FileEntry>,
    /// Payload files described by templates.
    pub groups: Vec<GroupEntry>,
    /// Auxiliary files.
    pub aux: Vec<AuxEntry>,
    /// Where the tensors came from.
    pub provenance: Provenance,
    /// Data outside the specification, by namespaced key.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub extensions: BTreeMap<String, serde_json::Value>,
}

/// A payload file described on its own.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FileEntry {
    /// The file's path.
    pub path: String,
    /// The file's size in bytes, a multiple of [`ALIGNMENT`].
    pub size: u64,
    /// The file's tags.
    pub tags: Vec<String>,
    /// The file's tensors, by increasing offset.
    pub tensors: Vec<TensorEntry>,
}

/// A tensor of a payload file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TensorEntry {
    /// The runtime's canonical name; in a group, a template.
    pub name: String,
    /// The element type.
    pub dtype: Dtype,
    /// The logical shape; empty for a scalar.
    pub shape: Vec<u64>,
    /// How logical indices map to elements.
    pub layout: Layout,
    /// How elements are encoded as bytes.
    pub scheme: Scheme,
    /// The byte offset in the file, a multiple of [`ALIGNMENT`].
    pub offset: u64,
    /// The size in bytes, as the scheme computes it from the shape.
    pub size: u64,
}

/// Payload files of identical structure, described by a template.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GroupEntry {
    /// The group's name.
    pub name: String,
    /// The template of the instances' paths.
    pub path: String,
    /// The template's axes, in order.
    pub axes: Vec<Axis>,
    /// Every instance file's size.
    pub size: u64,
    /// Every instance file's tags.
    pub tags: Vec<String>,
    /// The tensors of every instance; names are templates.
    pub tensors: Vec<TensorEntry>,
}

/// An axis of a group: either `0..count` or a list of values.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Axis {
    /// The name templates refer to the axis by.
    pub name: String,
    /// The axis is `0..count`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub count: Option<u64>,
    /// The axis is these values, strictly increasing.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub values: Option<Vec<u64>>,
}

impl Axis {
    /// The axis `0..count`.
    #[must_use]
    pub fn counted(name: impl Into<String>, count: u64) -> Self {
        Self {
            name: name.into(),
            count: Some(count),
            values: None,
        }
    }

    /// The axis of the given values.
    #[must_use]
    pub fn listed(name: impl Into<String>, values: Vec<u64>) -> Self {
        Self {
            name: name.into(),
            count: None,
            values: Some(values),
        }
    }

    /// How many values the axis has.
    #[must_use]
    pub fn value_count(&self) -> u64 {
        match (&self.count, &self.values) {
            (Some(count), _) => *count,
            (None, Some(values)) => values.len() as u64,
            (None, None) => 0,
        }
    }

    /// Whether `value` is one of the axis's values.
    #[must_use]
    pub fn contains(&self, value: u64) -> bool {
        match (&self.count, &self.values) {
            (Some(count), _) => value < *count,
            (None, Some(values)) => values.binary_search(&value).is_ok(),
            (None, None) => false,
        }
    }

    /// The axis's `index`-th value.
    fn value(&self, index: u64) -> u64 {
        match &self.values {
            Some(values) => values[usize::try_from(index).expect("validated axis index")],
            None => index,
        }
    }

    /// The axis's largest value, of a validated axis.
    fn max_value(&self) -> u64 {
        self.value(self.value_count().saturating_sub(1))
    }

    /// The total length of the axis's values in decimal, each zero-padded
    /// to at least `width` digits.
    fn rendered_len_sum(&self, width: usize) -> u64 {
        match (&self.count, &self.values) {
            (Some(count), _) => {
                // The values of each number of digits at once.
                let mut sum = 0u64;
                let (mut low, mut high, mut digits) = (0u128, 10u128, 1usize);
                while low < u128::from(*count) {
                    let n = high.min(u128::from(*count)) - low;
                    let each = digits.max(width) as u128;
                    sum = sum.saturating_add(u64::try_from(n * each).unwrap_or(u64::MAX));
                    (low, high, digits) = (high, high * 10, digits + 1);
                }
                sum
            }
            (None, Some(values)) => values
                .iter()
                .map(|&value| rendered_len(value, width))
                .fold(0, u64::saturating_add),
            (None, None) => 0,
        }
    }
}

// The length of `value` in decimal, zero-padded to at least `width` digits.
fn rendered_len(value: u64, width: usize) -> u64 {
    (u64::from(value.checked_ilog10().unwrap_or(0)) + 1).max(width as u64)
}

/// An auxiliary file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AuxEntry {
    /// The file's path.
    pub path: String,
    /// The file's size in bytes.
    pub size: u64,
    /// What the file is to the runtime, such as `tokenizer`.
    pub role: String,
}

/// Where a model's tensors came from.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Provenance {
    /// The sources the tensors were converted from, at least one.
    pub sources: Vec<Source>,
    /// The program that wrote the model.
    pub tool: Tool,
    /// Every transform applied on the way, in order.
    pub transforms: Vec<String>,
    /// Anything else a reader of the provenance should know.
    pub notes: Vec<String>,
}

/// The source of a model.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Source {
    /// The source's format, such as `safetensors`.
    pub kind: String,
    /// Where the source was published, such as a Hugging Face repository.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repository: Option<String>,
    /// The source's revision.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub revision: Option<String>,
    /// The source's files.
    pub files: Vec<SourceFile>,
}

/// A file of a model's source.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourceFile {
    /// The file's path in the source.
    pub path: String,
    /// The file's size in bytes.
    pub size: u64,
    /// The file's SHA-256 in hex, as Hugging Face's LFS object id gives it.
    #[serde(default, rename = "sha2-256", skip_serializing_if = "Option::is_none")]
    pub sha2_256: Option<String>,
}

/// The program that wrote a model.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Tool {
    /// The program's name.
    pub name: String,
    /// The program's version.
    pub version: String,
}

impl Manifest {
    /// Parses and checks a manifest, and builds the file table it defines,
    /// `bytes` being the whole of its `furymodel.json`.
    ///
    /// # Errors
    ///
    /// [`ManifestError::Unsupported`] for a manifest of another format or
    /// version, [`ManifestError::Json`] if it doesn't parse, and
    /// [`ManifestError::Invalid`] if it breaks a rule of the specification.
    pub fn parse(bytes: &[u8]) -> Result<(Self, FileTable)> {
        // The format and its version first, so that a newer manifest is
        // reported as such rather than as unknown fields.
        #[derive(Deserialize)]
        struct Header {
            format: Option<serde_json::Value>,
            format_version: Option<serde_json::Value>,
        }
        let header: Header = serde_json::from_slice(bytes)?;
        if header.format.as_ref().and_then(serde_json::Value::as_str) != Some(FORMAT) {
            return Err(ManifestError::Unsupported(format!(
                "not a FuryModel manifest: its format is {}",
                header
                    .format
                    .map_or_else(|| "missing".to_owned(), |format| format.to_string())
            )));
        }
        let version = header
            .format_version
            .as_ref()
            .and_then(serde_json::Value::as_u64);
        if version != Some(u64::from(FORMAT_VERSION)) {
            return Err(ManifestError::Unsupported(format!(
                "FuryModel format version {} isn't supported, only {FORMAT_VERSION}",
                header
                    .format_version
                    .map_or_else(|| "missing".to_owned(), |version| version.to_string())
            )));
        }
        check_unique_keys(bytes)?;
        let manifest: Self = serde_json::from_slice(bytes)?;
        manifest.validate()?;
        let table = manifest.build_table(bytes.len() as u64)?;
        Ok((manifest, table))
    }

    /// The manifest as `furymodel.json` holds it: pretty-printed JSON with a
    /// final newline. The same manifest always gives the same bytes.
    ///
    /// # Panics
    ///
    /// Never: every part of a manifest serializes to JSON.
    #[must_use]
    pub fn to_json(&self) -> Vec<u8> {
        let mut json = serde_json::to_vec_pretty(self).expect("a manifest serializes");
        json.push(b'\n');
        json
    }

    /// Checks every rule of the specification that concerns one entry at a
    /// time, and the limits of section 2.9, which it checks before any
    /// group's instances are written out; [`Manifest::file_table`] checks
    /// the rest.
    ///
    /// # Errors
    ///
    /// [`ManifestError::Unsupported`] or [`ManifestError::Invalid`],
    /// describing the first violation found.
    pub fn validate(&self) -> Result<()> {
        if self.format != FORMAT || self.format_version != FORMAT_VERSION {
            return Err(ManifestError::Unsupported(format!(
                "format {:?} version {} isn't {FORMAT:?} version {FORMAT_VERSION}",
                self.format, self.format_version
            )));
        }
        if self.alignment != ALIGNMENT {
            return Err(invalid(format!(
                "alignment is {}, not {ALIGNMENT}",
                self.alignment
            )));
        }
        check_description(
            &self.name,
            &self.arch,
            self.schema_version,
            &self.config,
            &self.provenance,
            &self.extensions,
        )?;
        let mut totals = Totals::new();
        for file in &self.files {
            check_path(&file.path).map_err(invalid)?;
            check_payload(&file.path, file.size, &file.tags, &file.tensors)?;
            totals.add_payload(&file.path, &file.tags, &file.tensors)?;
        }
        let mut groups = BTreeSet::new();
        for group in &self.groups {
            if !groups.insert(group.name.as_str()) {
                return Err(invalid(format!("two groups are named {:?}", group.name)));
            }
            check_group(group)?;
            totals.add_group(group)?;
        }
        let mut roles = BTreeSet::new();
        for aux in &self.aux {
            check_path(&aux.path).map_err(invalid)?;
            check_role(&aux.path, &aux.role)?;
            if !roles.insert(aux.role.as_str()) {
                return Err(invalid(format!(
                    "auxiliary files {:?} and another have the role {:?}",
                    aux.path, aux.role
                )));
            }
            totals.add_aux(&aux.path, &aux.role)?;
        }
        Ok(())
    }

    /// The file table: every payload file, group instance and auxiliary
    /// file, and the manifest itself of `manifest_size` bytes, sorted by
    /// path. Validates the manifest first, then checks that paths and tensor
    /// names are unique and that no path is another's directory, paths
    /// compared without regard to case as well.
    ///
    /// # Errors
    ///
    /// As [`Manifest::validate`], and [`ManifestError::Invalid`] for paths
    /// or tensor names that collide.
    pub fn file_table(&self, manifest_size: u64) -> Result<FileTable> {
        self.validate()?;
        self.build_table(manifest_size)
    }

    // The file table of a validated manifest.
    fn build_table(&self, manifest_size: u64) -> Result<FileTable> {
        let mut entries = vec![TableEntry {
            path: MANIFEST_PATH.to_owned(),
            size: manifest_size,
            kind: EntryKind::Manifest,
        }];
        for file in &self.files {
            entries.push(TableEntry {
                path: file.path.clone(),
                size: file.size,
                kind: EntryKind::Payload(Payload {
                    tags: file.tags.clone(),
                    tensors: file.tensors.clone(),
                    group: None,
                }),
            });
        }
        for group in &self.groups {
            expand_group(group, &mut entries)?;
        }
        for aux in &self.aux {
            entries.push(TableEntry {
                path: aux.path.clone(),
                size: aux.size,
                kind: EntryKind::Aux {
                    role: aux.role.clone(),
                },
            });
        }
        entries.sort_by(|a, b| a.path.cmp(&b.path));
        check_collisions(&entries)?;
        if entries
            .iter()
            .try_fold(0u64, |total, entry| total.checked_add(entry.size))
            .is_none()
        {
            return Err(invalid("the files' sizes add up to more than 2^64 bytes"));
        }
        let mut tensors = HashMap::new();
        for (file, entry) in entries.iter().enumerate() {
            if let EntryKind::Payload(payload) = &entry.kind {
                for (index, tensor) in payload.tensors.iter().enumerate() {
                    if tensors.insert(tensor.name.clone(), (file, index)).is_some() {
                        return Err(invalid(format!("two tensors are named {:?}", tensor.name)));
                    }
                }
            }
        }
        Ok(FileTable { entries, tensors })
    }
}

/// The files of a model, sorted by path: the order hash tables and the
/// content id follow.
#[derive(Debug, Clone, PartialEq)]
pub struct FileTable {
    entries: Vec<TableEntry>,
    tensors: HashMap<String, (usize, usize)>,
}

/// A file of the file table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TableEntry {
    /// The file's path.
    pub path: String,
    /// The file's size in bytes.
    pub size: u64,
    /// What the file is.
    pub kind: EntryKind,
}

/// What a file of the file table is.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EntryKind {
    /// The manifest, `furymodel.json`.
    Manifest,
    /// A payload file.
    Payload(Payload),
    /// An auxiliary file.
    Aux {
        /// What the file is to the runtime.
        role: String,
    },
}

/// A payload file of the file table, with its group instance's names
/// substituted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Payload {
    /// The file's tags.
    pub tags: Vec<String>,
    /// The file's tensors, by increasing offset.
    pub tensors: Vec<TensorEntry>,
    /// The group and the axis values the file is an instance of, if any.
    pub group: Option<GroupInstance>,
}

/// The group instance a payload file is.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GroupInstance {
    /// The group's name.
    pub group: String,
    /// The instance's value of each of the group's axes, in order.
    pub values: Vec<u64>,
}

impl FileTable {
    /// The files, sorted by path.
    #[must_use]
    pub fn entries(&self) -> &[TableEntry] {
        &self.entries
    }

    /// The number of files.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the table has no files. It never has: the manifest is one.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The index of the file of `path`.
    #[must_use]
    pub fn index_of(&self, path: &str) -> Option<usize> {
        self.entries
            .binary_search_by(|entry| entry.path.as_str().cmp(path))
            .ok()
    }

    /// The file holding the tensor `name`, by index, and the tensor.
    #[must_use]
    pub fn tensor(&self, name: &str) -> Option<(usize, &TensorEntry)> {
        let &(file, index) = self.tensors.get(name)?;
        match &self.entries[file].kind {
            EntryKind::Payload(payload) => Some((file, &payload.tensors[index])),
            EntryKind::Manifest | EntryKind::Aux { .. } => None,
        }
    }

    /// Every tensor, file by file, with the index of its file.
    pub fn tensors(&self) -> impl Iterator<Item = (usize, &TensorEntry)> {
        self.entries
            .iter()
            .enumerate()
            .filter_map(|(file, entry)| match &entry.kind {
                EntryKind::Payload(payload) => Some((file, payload)),
                EntryKind::Manifest | EntryKind::Aux { .. } => None,
            })
            .flat_map(|(file, payload)| payload.tensors.iter().map(move |tensor| (file, tensor)))
    }

    /// The total size of the files in bytes, which a file table's files
    /// never take more than 2^64 of.
    #[must_use]
    pub fn total_size(&self) -> u64 {
        self.entries
            .iter()
            .fold(0u64, |total, entry| total.saturating_add(entry.size))
    }
}

/// Checks a path against the specification's section 1, on its own: what
/// concerns two paths, [`Manifest::file_table`] checks.
///
/// # Errors
///
/// Describes what's wrong with the path.
pub fn check_path(path: &str) -> std::result::Result<(), String> {
    if path.is_empty() {
        return Err("a path is empty".to_owned());
    }
    if path.len() > MAX_PATH_LEN {
        return Err(format!(
            "path {:?}… is longer than {MAX_PATH_LEN} bytes",
            &path[..path.floor_char_boundary(64)]
        ));
    }
    for segment in path.split('/') {
        if segment.is_empty() || segment == "." || segment == ".." || segment.len() > 255 {
            return Err(format!(
                "path {path:?} has an empty, \".\", \"..\" or overlong segment"
            ));
        }
        if !segment
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
        {
            return Err(format!(
                "path {path:?} has characters other than A-Z a-z 0-9 . _ -"
            ));
        }
    }
    // Reserved as files and as directories, and on file systems that ignore
    // case in every case.
    let first = path.split('/').next().unwrap_or_default();
    if [MANIFEST_PATH, PART_PATH, HASHES_DIR]
        .iter()
        .any(|reserved| first.eq_ignore_ascii_case(reserved))
    {
        return Err(format!("path {path:?} is reserved"));
    }
    Ok(())
}

// A path's key in the order that finds the paths no file system holds
// side by side: case-folded, as file systems that ignore case see it, with
// `/` as the least byte, which no path has. Two paths that name one file
// where case is ignored have one key, and every key between a path's and
// that of a path below it is below it too.
pub(crate) fn path_key(path: &str) -> String {
    path.to_ascii_lowercase().replace('/', "\0")
}

// Whether the path of `key` needs that of `directory` as a directory.
pub(crate) fn is_below(key: &str, directory: &str) -> bool {
    key.strip_prefix(directory)
        .is_some_and(|rest| rest.starts_with('\0'))
}

// What concerns two paths of a file table: none twice, none another's
// directory, and the same when case is ignored, as many file systems do.
// In the order of their keys, two paths with one key are neighbours, and
// so are a file and the first path below it: one sort and one pass find
// both.
fn check_collisions(entries: &[TableEntry]) -> Result<()> {
    let mut keys: Vec<(String, &str)> = entries
        .iter()
        .map(|entry| (path_key(&entry.path), entry.path.as_str()))
        .collect();
    keys.sort_unstable();
    for pair in keys.windows(2) {
        let [(first_key, first), (second_key, second)] = pair else {
            unreachable!("windows of two")
        };
        if first_key == second_key {
            return Err(invalid(if first == second {
                format!("two files have the path {first:?}")
            } else {
                format!("the paths {first:?} and {second:?} name one file where case is ignored")
            }));
        }
        if is_below(second_key, first_key) {
            return Err(invalid(format!(
                "{first:?} is a file, but {second:?} needs it as a directory"
            )));
        }
    }
    Ok(())
}

fn is_word(name: &str) -> bool {
    !name.is_empty()
        && name.bytes().all(|byte| {
            byte.is_ascii_lowercase() || byte.is_ascii_digit() || matches!(byte, b'_' | b'-')
        })
}

fn check_tags(what: &str, tags: &[String]) -> Result<()> {
    let mut seen = BTreeSet::new();
    for tag in tags {
        if !is_word(tag) || !seen.insert(tag.as_str()) {
            return Err(invalid(format!(
                "{what} has the invalid or repeated tag {tag:?}"
            )));
        }
    }
    Ok(())
}

pub(crate) fn check_payload(
    what: &str,
    size: u64,
    tags: &[String],
    tensors: &[TensorEntry],
) -> Result<()> {
    if !size.is_multiple_of(ALIGNMENT) {
        return Err(invalid(format!(
            "{what}: size {size} isn't a multiple of {ALIGNMENT}"
        )));
    }
    check_tags(what, tags)?;
    let mut end = 0;
    for tensor in tensors {
        let name = &tensor.name;
        if name.is_empty() || name.len() > MAX_NAME_LEN {
            return Err(invalid(format!(
                "{what}: a tensor's name is empty or longer than {MAX_NAME_LEN} bytes"
            )));
        }
        if tensor.shape.len() > MAX_RANK {
            return Err(invalid(format!(
                "{what}: tensor {name:?} has more than {MAX_RANK} dimensions"
            )));
        }
        tensor
            .layout
            .check_shape(&tensor.shape)
            .map_err(|message| invalid(format!("{what}: tensor {name:?}: {message}")))?;
        if tensor.scheme.dtype() != tensor.dtype {
            return Err(invalid(format!(
                "{what}: tensor {name:?}: scheme {} doesn't encode dtype {}",
                tensor.scheme, tensor.dtype
            )));
        }
        if tensor.scheme.size(&tensor.shape) != Some(tensor.size) {
            return Err(invalid(format!(
                "{what}: tensor {name:?}: size {} isn't what scheme {} gives shape {:?}",
                tensor.size, tensor.scheme, tensor.shape
            )));
        }
        if !tensor.offset.is_multiple_of(ALIGNMENT) || tensor.offset < end {
            return Err(invalid(format!(
                "{what}: tensor {name:?}: offset {} isn't aligned to {ALIGNMENT}, or overlaps or \
                 precedes the tensor before it",
                tensor.offset
            )));
        }
        end = tensor
            .offset
            .checked_add(tensor.size)
            .filter(|&end| end <= size)
            .ok_or_else(|| invalid(format!("{what}: tensor {name:?} ends past the file")))?;
    }
    Ok(())
}

pub(crate) fn check_group(group: &GroupEntry) -> Result<()> {
    if group.name.len() > MAX_GROUP_NAME_LEN {
        return Err(invalid(format!(
            "group {:?}… has a name longer than {MAX_GROUP_NAME_LEN} bytes",
            &group.name[..group.name.floor_char_boundary(MAX_GROUP_NAME_LEN)]
        )));
    }
    let what = format!("group {:?}", group.name);
    if !is_word(&group.name) {
        return Err(invalid(format!("{what}: names are made of a-z 0-9 _ -")));
    }
    if group.axes.is_empty() || group.axes.len() > MAX_AXES {
        return Err(invalid(format!(
            "{what} has no axes, or more than {MAX_AXES}"
        )));
    }
    let mut names = BTreeSet::new();
    let mut instances = 1u64;
    for axis in &group.axes {
        let valid_name = axis
            .name
            .bytes()
            .next()
            .is_some_and(|byte| byte.is_ascii_lowercase())
            && axis
                .name
                .bytes()
                .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_');
        if !valid_name || !names.insert(axis.name.as_str()) {
            return Err(invalid(format!(
                "{what}: axis {:?} is repeated, or isn't a-z 0-9 _ starting with a letter",
                axis.name
            )));
        }
        let valid_values = match (&axis.count, &axis.values) {
            (Some(count), None) => *count >= 1,
            (None, Some(values)) => {
                !values.is_empty() && values.windows(2).all(|pair| pair[0] < pair[1])
            }
            _ => false,
        };
        if !valid_values {
            return Err(invalid(format!(
                "{what}: axis {:?} needs either a count of at least 1 or strictly increasing \
                 values",
                axis.name
            )));
        }
        instances = instances
            .checked_mul(axis.value_count())
            .filter(|&instances| instances < MAX_FILES)
            .ok_or_else(|| {
                invalid(format!(
                    "{what} has more instances than a file table holds files"
                ))
            })?;
    }
    let path = Template::parse(&group.path, &group.axes)
        .map_err(|message| invalid(format!("{what}: {message}")))?;
    if let Some(axis) = (0..group.axes.len()).find(|&axis| !path.uses(axis)) {
        return Err(invalid(format!(
            "{what}: the path template doesn't use axis {:?}",
            group.axes[axis].name
        )));
    }
    // Digits are valid in any segment and never make one empty, "." or
    // "..", nor a reserved name: the first and the last instance (the
    // longest) stand for all of them.
    for values in [
        group
            .axes
            .iter()
            .map(|axis| axis.value(0))
            .collect::<Vec<_>>(),
        group.axes.iter().map(Axis::max_value).collect(),
    ] {
        check_path(&path.render(&values))
            .map_err(|message| invalid(format!("{what}: {message}")))?;
    }
    for tensor in &group.tensors {
        let name = Template::parse(&tensor.name, &group.axes)
            .map_err(|message| invalid(format!("{what}: {message}")))?;
        if let Some(axis) = (0..group.axes.len())
            .find(|&axis| group.axes[axis].value_count() > 1 && !name.uses(axis))
        {
            return Err(invalid(format!(
                "{what}: the tensor name template {:?} doesn't use axis {:?}, which has more \
                 than one value",
                tensor.name, group.axes[axis].name
            )));
        }
        if name.max_len(&group.axes) > MAX_NAME_LEN as u64 {
            return Err(invalid(format!(
                "{what}: tensor {:?} has names longer than {MAX_NAME_LEN} bytes",
                tensor.name
            )));
        }
    }
    check_payload(&what, group.size, &group.tags, &group.tensors)
}

fn expand_group(group: &GroupEntry, entries: &mut Vec<TableEntry>) -> Result<()> {
    let what = format!("group {:?}", group.name);
    let path = Template::parse(&group.path, &group.axes)
        .map_err(|message| invalid(format!("{what}: {message}")))?;
    let names = group
        .tensors
        .iter()
        .map(|tensor| Template::parse(&tensor.name, &group.axes))
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|message| invalid(format!("{what}: {message}")))?;
    for values in instances(&group.axes) {
        let instance_path = path.render(&values);
        check_path(&instance_path).map_err(|message| invalid(format!("{what}: {message}")))?;
        let tensors = group
            .tensors
            .iter()
            .zip(&names)
            .map(|(tensor, name)| TensorEntry {
                name: name.render(&values),
                ..tensor.clone()
            })
            .collect();
        entries.push(TableEntry {
            path: instance_path,
            size: group.size,
            kind: EntryKind::Payload(Payload {
                tags: group.tags.clone(),
                tensors,
                group: Some(GroupInstance {
                    group: group.name.clone(),
                    values,
                }),
            }),
        });
    }
    Ok(())
}

/// Every combination of the values of `axes`, the last axis fastest.
pub(crate) fn instances(axes: &[Axis]) -> impl Iterator<Item = Vec<u64>> + '_ {
    let mut next = Some(vec![0u64; axes.len()]);
    std::iter::from_fn(move || {
        let counters = next.take()?;
        let values = axes
            .iter()
            .zip(&counters)
            .map(|(axis, &counter)| axis.value(counter))
            .collect();
        let mut following = counters;
        for axis in (0..axes.len()).rev() {
            following[axis] += 1;
            if following[axis] < axes[axis].value_count() {
                next = Some(following);
                break;
            }
            following[axis] = 0;
        }
        Some(values)
    })
}

/// Checks what a manifest says besides its files: the name, the
/// architecture and its schema version, the configuration, the provenance
/// and the extensions' keys.
pub(crate) fn check_description(
    name: &str,
    arch: &str,
    schema_version: u32,
    config: &serde_json::Value,
    provenance: &Provenance,
    extensions: &BTreeMap<String, serde_json::Value>,
) -> Result<()> {
    if name.is_empty() || arch.is_empty() {
        return Err(invalid("name and arch must not be empty"));
    }
    if schema_version == 0 {
        return Err(invalid("schema_version must be at least 1"));
    }
    if !config.is_object() {
        return Err(invalid("config must be a JSON object"));
    }
    check_provenance(provenance)?;
    for key in extensions.keys() {
        check_extension_key(key)?;
    }
    Ok(())
}

pub(crate) fn check_role(path: &str, role: &str) -> Result<()> {
    if role.is_empty() {
        return Err(invalid(format!(
            "auxiliary file {path:?} has an empty role"
        )));
    }
    Ok(())
}

/// The running totals of a file table, held to the limits of section 2.9
/// as entries are added, before any group's instances are written out: the
/// files, the tensors, and the description, which stands for what a reader
/// holds of the file table, every instance's copy of its group's tags, name
/// and tensors included.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Totals {
    files: u64,
    tensors: u64,
    description: u64,
}

// What a string takes of the description.
fn string_cost(text: &str) -> u64 {
    (text.len() as u64).saturating_add(STRING_COST)
}

// What a tensor takes of the description, but for its name's length.
fn tensor_cost(tensor: &TensorEntry) -> u64 {
    ENTRY_COST + STRING_COST + NUMBER_COST.saturating_mul(tensor.shape.len() as u64)
}

impl Totals {
    /// The totals of a file table of the manifest alone.
    pub(crate) fn new() -> Self {
        Self {
            files: 1,
            tensors: 0,
            description: ENTRY_COST + string_cost(MANIFEST_PATH),
        }
    }

    fn add(&mut self, files: u64, tensors: u64, description: u64) -> Result<()> {
        let next = Self {
            files: self.files.saturating_add(files),
            tensors: self.tensors.saturating_add(tensors),
            description: self.description.saturating_add(description),
        };
        if next.files > MAX_FILES {
            return Err(invalid(format!(
                "the file table holds more than {MAX_FILES} files"
            )));
        }
        if next.tensors > MAX_TENSORS {
            return Err(invalid(format!(
                "the payload files hold more than {MAX_TENSORS} tensors"
            )));
        }
        if next.description > MAX_DESCRIPTION {
            return Err(invalid(format!(
                "the file table's description takes more than {} MiB",
                MAX_DESCRIPTION >> 20
            )));
        }
        *self = next;
        Ok(())
    }

    /// Adds a payload file described on its own.
    pub(crate) fn add_payload(
        &mut self,
        path: &str,
        tags: &[String],
        tensors: &[TensorEntry],
    ) -> Result<()> {
        let description = tensors.iter().fold(
            tags.iter()
                .map(|tag| string_cost(tag))
                .fold(ENTRY_COST + string_cost(path), u64::saturating_add),
            |sum, tensor| {
                sum.saturating_add(tensor_cost(tensor))
                    .saturating_add(tensor.name.len() as u64)
            },
        );
        self.add(1, tensors.len() as u64, description)
    }

    /// Adds every instance of a group that [`check_group`] accepted.
    pub(crate) fn add_group(&mut self, group: &GroupEntry) -> Result<()> {
        let rendered = |template: &str| {
            Template::parse(template, &group.axes)
                .map(|template| template.total_len(&group.axes))
                .map_err(invalid)
        };
        let instances = group
            .axes
            .iter()
            .map(Axis::value_count)
            .fold(1u64, u64::saturating_mul);
        // Each instance's own copy of its group's tags and name, its axis
        // values, its tensors and its path but for the rendered lengths.
        let each = [
            ENTRY_COST,
            STRING_COST,
            string_cost(&group.name),
            NUMBER_COST.saturating_mul(group.axes.len() as u64),
        ]
        .into_iter()
        .chain(group.tags.iter().map(|tag| string_cost(tag)))
        .chain(group.tensors.iter().map(tensor_cost))
        .fold(0, u64::saturating_add);
        let mut description = instances
            .saturating_mul(each)
            .saturating_add(rendered(&group.path)?);
        for tensor in &group.tensors {
            description = description.saturating_add(rendered(&tensor.name)?);
        }
        self.add(
            instances,
            instances.saturating_mul(group.tensors.len() as u64),
            description,
        )
    }

    /// Adds an auxiliary file.
    pub(crate) fn add_aux(&mut self, path: &str, role: &str) -> Result<()> {
        self.add(1, 0, ENTRY_COST + string_cost(path) + string_cost(role))
    }
}

/// Checks that no JSON object of `bytes` has a key twice, as section 2
/// asks: JSON parsers disagree on which of two equal keys wins.
pub(crate) fn check_unique_keys(bytes: &[u8]) -> Result<(), serde_json::Error> {
    use serde::de::{self, DeserializeSeed, Deserializer, MapAccess, SeqAccess, Visitor};

    struct Unique;

    impl<'de> DeserializeSeed<'de> for Unique {
        type Value = ();

        fn deserialize<D: Deserializer<'de>>(self, deserializer: D) -> Result<(), D::Error> {
            deserializer.deserialize_any(self)
        }
    }

    impl<'de> Visitor<'de> for Unique {
        type Value = ();

        fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("JSON")
        }

        fn visit_bool<E>(self, _: bool) -> Result<(), E> {
            Ok(())
        }

        fn visit_i64<E>(self, _: i64) -> Result<(), E> {
            Ok(())
        }

        fn visit_u64<E>(self, _: u64) -> Result<(), E> {
            Ok(())
        }

        fn visit_f64<E>(self, _: f64) -> Result<(), E> {
            Ok(())
        }

        fn visit_str<E>(self, _: &str) -> Result<(), E> {
            Ok(())
        }

        fn visit_unit<E>(self) -> Result<(), E> {
            Ok(())
        }

        fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<(), A::Error> {
            while seq.next_element_seed(Unique)?.is_some() {}
            Ok(())
        }

        fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<(), A::Error> {
            let mut keys = std::collections::HashSet::new();
            while let Some(key) = map.next_key::<String>()? {
                if !keys.insert(key.clone()) {
                    return Err(de::Error::custom(format!(
                        "an object has the key {key:?} twice"
                    )));
                }
                map.next_value_seed(Unique)?;
            }
            Ok(())
        }
    }

    let mut deserializer = serde_json::Deserializer::from_slice(bytes);
    Unique.deserialize(&mut deserializer)?;
    deserializer.end()
}

fn check_provenance(provenance: &Provenance) -> Result<()> {
    if provenance.sources.is_empty()
        || provenance
            .sources
            .iter()
            .any(|source| source.kind.is_empty())
        || provenance.tool.name.is_empty()
        || provenance.tool.version.is_empty()
    {
        return Err(invalid(
            "provenance needs a source, each with a kind, and the tool's name and version",
        ));
    }
    for file in provenance.sources.iter().flat_map(|source| &source.files) {
        let valid_sha = file.sha2_256.as_ref().is_none_or(|sha| {
            sha.len() == 64
                && sha
                    .bytes()
                    .all(|byte| matches!(byte, b'0'..=b'9' | b'a'..=b'f'))
        });
        if file.path.is_empty() || !valid_sha {
            return Err(invalid(format!(
                "provenance: source file {:?} needs a path and, if it has one, a sha2-256 of 64 \
                 lowercase hex digits",
                file.path
            )));
        }
    }
    Ok(())
}

fn check_extension_key(key: &str) -> Result<()> {
    let segments: Vec<&str> = key.split('.').collect();
    if segments.len() >= 2 && segments.iter().all(|segment| is_word(segment)) {
        Ok(())
    } else {
        Err(invalid(format!(
            "extension key {key:?} isn't namespaced, as in \"owner.name\""
        )))
    }
}

/// A path or name template of a group.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Template {
    pieces: Vec<Piece>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Piece {
    Literal(String),
    Axis { axis: usize, width: usize },
}

impl Template {
    pub(crate) fn parse(template: &str, axes: &[Axis]) -> std::result::Result<Self, String> {
        let mut pieces = Vec::new();
        let mut rest = template;
        while let Some(open) = rest.find(['{', '}']) {
            if rest.as_bytes()[open] == b'}' {
                return Err(format!("template {template:?} has an unmatched '}}'"));
            }
            if open > 0 {
                pieces.push(Piece::Literal(rest[..open].to_owned()));
            }
            let after = &rest[open + 1..];
            let close = after
                .find(['{', '}'])
                .filter(|&close| after.as_bytes()[close] == b'}')
                .ok_or_else(|| format!("template {template:?} has an unclosed '{{'"))?;
            let (name, width) = match after[..close].split_once(':') {
                None => (&after[..close], 0),
                Some((name, width)) => match width.as_bytes() {
                    [b'0', digit @ b'1'..=b'9'] => (name, usize::from(digit - b'0')),
                    _ => {
                        return Err(format!(
                            "template {template:?}: the width {width:?} isn't 0W, W from 1 to 9"
                        ));
                    }
                },
            };
            let index = axes
                .iter()
                .position(|axis| axis.name == name)
                .ok_or_else(|| {
                    format!("template {template:?} uses {name:?}, which isn't an axis")
                })?;
            pieces.push(Piece::Axis { axis: index, width });
            rest = &after[close + 1..];
        }
        if !rest.is_empty() {
            pieces.push(Piece::Literal(rest.to_owned()));
        }
        // Values render as digits alone, so a character other than a digit
        // between two of them keeps their renderings apart.
        let mut apart = true;
        for piece in &pieces {
            match piece {
                Piece::Literal(literal) => {
                    apart |= literal.bytes().any(|byte| !byte.is_ascii_digit());
                }
                Piece::Axis { .. } if !apart => {
                    return Err(format!(
                        "template {template:?} has two axes with nothing but digits between them"
                    ));
                }
                Piece::Axis { .. } => apart = false,
            }
        }
        Ok(Self { pieces })
    }

    fn uses(&self, axis: usize) -> bool {
        self.pieces
            .iter()
            .any(|piece| matches!(piece, Piece::Axis { axis: used, .. } if *used == axis))
    }

    /// The length of the longest rendering over the axes' values: each
    /// value's length grows with it, so the largest values render longest.
    fn max_len(&self, axes: &[Axis]) -> u64 {
        self.pieces
            .iter()
            .map(|piece| match piece {
                Piece::Literal(literal) => literal.len() as u64,
                Piece::Axis { axis, width } => rendered_len(axes[*axis].max_value(), *width),
            })
            .fold(0, u64::saturating_add)
    }

    /// The total length of the renderings over every combination of the
    /// axes' values, computed without rendering any.
    fn total_len(&self, axes: &[Axis]) -> u64 {
        let instances = axes
            .iter()
            .map(Axis::value_count)
            .fold(1u64, u64::saturating_mul);
        self.pieces
            .iter()
            .map(|piece| match piece {
                Piece::Literal(literal) => (literal.len() as u64).saturating_mul(instances),
                // Each of the axis's values in as many combinations as the
                // other axes make.
                Piece::Axis { axis, width } => {
                    let values = &axes[*axis];
                    (instances / values.value_count().max(1))
                        .saturating_mul(values.rendered_len_sum(*width))
                }
            })
            .fold(0, u64::saturating_add)
    }

    pub(crate) fn render(&self, values: &[u64]) -> String {
        let mut rendered = String::new();
        for piece in &self.pieces {
            match piece {
                Piece::Literal(literal) => rendered.push_str(literal),
                Piece::Axis { axis, width } => {
                    write!(rendered, "{:0width$}", values[*axis], width = *width)
                        .expect("writing to a String");
                }
            }
        }
        rendered
    }
}

#[cfg(test)]
mod tests;
