// SPDX-FileCopyrightText: 2026 Mika Tammi
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Writing a model: its payload files, group instances and auxiliary files,
//! hashed while they are written, and at the end a hash table of every
//! algorithm and, last, its manifest, so that a root holding a manifest
//! holds the whole model.
//!
//! A [`Writer`] is shared between threads, so that a converter can write
//! many files at once. Every rule of the specification is checked as each
//! file is added, before it is written, so that [`Writer::finish`] has
//! nothing left to refuse but missing group instances. The manifest it
//! writes doesn't depend on the order files were added in, so the same
//! model always gets the same content id.

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::fs::{self, File, OpenOptions};
use std::io::Write as _;
use std::ops::Bound;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard, PoisonError};

use crate::error::{Error, IoContext as _, Result};
use crate::hash::{Algorithm, ContentId, FileDigests, FileHasher, HashTable};
use crate::manifest::{
    ALIGNMENT, AuxEntry, Axis, FORMAT, FORMAT_VERSION, FileEntry, FileTable, GroupEntry,
    HASHES_DIR, MANIFEST_PATH, Manifest, ManifestError, Provenance, Template, TensorEntry, Totals,
    check_description, check_group, check_path, check_payload, check_role, instances, is_below,
    path_key,
};
use crate::registry::{Dtype, Layout, Scheme};

/// What a model's manifest says besides its files.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelInfo {
    /// The display name.
    pub name: String,
    /// The runtime's architecture id.
    pub arch: String,
    /// The runtime's tensor-schema version for `arch`.
    pub schema_version: u32,
    /// The runtime's typed configuration, a JSON object.
    pub config: serde_json::Value,
    /// Where the tensors came from.
    pub provenance: Provenance,
    /// Data outside the specification, by namespaced key.
    pub extensions: BTreeMap<String, serde_json::Value>,
}

/// A tensor to write into a payload file.
#[derive(Debug, Clone, Copy)]
pub struct TensorData<'a> {
    /// The runtime's canonical name.
    pub name: &'a str,
    /// The element type.
    pub dtype: Dtype,
    /// The logical shape.
    pub shape: &'a [u64],
    /// How logical indices map to elements.
    pub layout: Layout,
    /// How elements are encoded; the plain scheme of `dtype` in registry
    /// version 1.
    pub scheme: Scheme,
    /// The encoded bytes.
    pub bytes: &'a [u8],
}

/// A tensor of every instance of a group.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorSpec {
    /// The name template.
    pub name: String,
    /// The element type.
    pub dtype: Dtype,
    /// The logical shape.
    pub shape: Vec<u64>,
    /// How logical indices map to elements.
    pub layout: Layout,
    /// How elements are encoded.
    pub scheme: Scheme,
}

/// A group of payload files of identical structure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GroupSpec {
    /// The group's name.
    pub name: String,
    /// The template of the instances' paths.
    pub path: String,
    /// The template's axes.
    pub axes: Vec<Axis>,
    /// Every instance file's tags.
    pub tags: Vec<String>,
    /// The tensors of every instance, in file order.
    pub tensors: Vec<TensorSpec>,
}

/// A group added to a [`Writer`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GroupId(usize);

/// What [`Writer::finish`] wrote.
#[derive(Debug, Clone, PartialEq)]
pub struct Written {
    /// The manifest.
    pub manifest: Manifest,
    /// The model's file table.
    pub table: FileTable,
    /// The model's content id, if the sha3-512 table was among those written.
    pub content_id: Option<ContentId>,
}

/// Writes a model into a root directory.
#[derive(Debug)]
pub struct Writer {
    root: PathBuf,
    info: ModelInfo,
    algorithms: Vec<Algorithm>,
    state: Mutex<State>,
}

#[derive(Debug)]
struct State {
    files: Vec<FileEntry>,
    groups: Vec<Group>,
    aux: Vec<AuxEntry>,
    digests: BTreeMap<String, Vec<FileDigests>>,
    // Every path added or being written, so that two threads can't take
    // one, by its key (see manifest::path_key): as file systems that ignore
    // case see it, and in an order with a file next to the paths below it.
    paths: BTreeMap<String, String>,
    // The directories files were written into, as written: they and every
    // directory above them made durable before the manifest.
    written_into: BTreeSet<String>,
    tensors: HashSet<String>,
    roles: BTreeSet<String>,
    totals: Totals,
}

impl State {
    fn new() -> Self {
        Self {
            files: Vec::new(),
            groups: Vec::new(),
            aux: Vec::new(),
            digests: BTreeMap::new(),
            paths: BTreeMap::new(),
            written_into: BTreeSet::new(),
            tensors: HashSet::new(),
            roles: BTreeSet::new(),
            totals: Totals::new(),
        }
    }
}

#[derive(Debug)]
struct Group {
    entry: GroupEntry,
    path: Template,
    instances: u64,
    // The instances being written or written, and those written.
    started: BTreeSet<Vec<u64>>,
    written: BTreeSet<Vec<u64>>,
}

impl Writer {
    /// A writer of a new model into `root`, which must not exist or be an
    /// empty directory, with a hash table of every algorithm.
    ///
    /// # Errors
    ///
    /// If `info` breaks a rule of the specification, `root` isn't empty or
    /// can't be created, or an algorithm can't be computed here
    /// (gxhash3-128 needs AES instructions, as target features of the build
    /// and in the CPU).
    pub fn create(root: impl Into<PathBuf>, info: ModelInfo) -> Result<Self> {
        Self::with_algorithms(root, info, &Algorithm::ALL)
    }

    /// Like [`Writer::create`], with hash tables of the given algorithms
    /// only.
    ///
    /// # Errors
    ///
    /// As [`Writer::create`].
    pub fn with_algorithms(
        root: impl Into<PathBuf>,
        info: ModelInfo,
        algorithms: &[Algorithm],
    ) -> Result<Self> {
        let root = root.into();
        if let Some(&algorithm) = algorithms
            .iter()
            .find(|algorithm| !algorithm.is_supported())
        {
            return Err(Error::Write(format!(
                "{algorithm} can't be computed here: gxhash3-128 needs AES instructions, as \
                 target features of the build and in the CPU"
            )));
        }
        check_description(
            &info.name,
            &info.arch,
            info.schema_version,
            &info.config,
            &info.provenance,
            &info.extensions,
        )
        .map_err(|error| Error::Write(format!("the model's description: {error}")))?;
        fs::create_dir_all(&root).at(&root)?;
        if fs::read_dir(&root).at(&root)?.next().is_some() {
            return Err(Error::Write(format!(
                "{} isn't empty: a model is written into a new directory",
                root.display()
            )));
        }
        let mut algorithms = algorithms.to_vec();
        algorithms.sort();
        algorithms.dedup();
        Ok(Self {
            root,
            info,
            algorithms,
            state: Mutex::new(State::new()),
        })
    }

    /// The root the model is written into.
    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    fn state(&self) -> MutexGuard<'_, State> {
        // A thread that panicked while holding the lock left nothing half
        // done: every change to the state is a single push or insert.
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    // Takes `path` for a file about to be written, with the names of its
    // tensors and, for an auxiliary file, its role, and counts the file in
    // the totals with `count`: all of it or nothing, so that nothing gets
    // written that the finished manifest would break a rule with.
    fn reserve(
        &self,
        path: &str,
        tensors: &[&str],
        role: Option<&str>,
        count: impl FnOnce(&mut Totals) -> std::result::Result<(), ManifestError>,
    ) -> Result<()> {
        let mut state = self.state();
        let key = path_key(path);
        check_free(&state, path, &key, tensors).map_err(Error::Write)?;
        if let Some(role) = role.filter(|&role| state.roles.contains(role)) {
            return Err(Error::Write(format!(
                "two auxiliary files have the role {role:?}"
            )));
        }
        count(&mut state.totals).map_err(|error| Error::Write(error.to_string()))?;
        if let Some((parent, _)) = path.rsplit_once('/') {
            state.written_into.insert(parent.to_owned());
        }
        state.paths.insert(key, path.to_owned());
        state
            .tensors
            .extend(tensors.iter().map(|&name| name.to_owned()));
        if let Some(role) = role {
            state.roles.insert(role.to_owned());
        }
        Ok(())
    }

    /// Writes a payload file holding `tensors`, each at the next aligned
    /// offset after the one before.
    ///
    /// # Errors
    ///
    /// If the path or the tensors break a rule of the specification, alone
    /// or with what was added before (a path taken where case is ignored, a
    /// tensor's name taken, a limit of section 2.9 passed), or writing
    /// fails.
    pub fn add_file(&self, path: &str, tags: &[&str], tensors: &[TensorData<'_>]) -> Result<()> {
        let mut entries = Vec::with_capacity(tensors.len());
        let mut offset = 0;
        for tensor in tensors {
            let size = tensor.bytes.len() as u64;
            entries.push(TensorEntry {
                name: tensor.name.to_owned(),
                dtype: tensor.dtype,
                shape: tensor.shape.to_vec(),
                layout: tensor.layout,
                scheme: tensor.scheme,
                offset,
                size,
            });
            offset = align(offset + size);
        }
        let entry = FileEntry {
            path: path.to_owned(),
            size: offset,
            tags: tags.iter().map(|&tag| tag.to_owned()).collect(),
            tensors: entries,
        };
        check_path(path).map_err(Error::Write)?;
        check_payload(path, entry.size, &entry.tags, &entry.tensors)
            .map_err(|error| Error::Write(error.to_string()))?;
        let names: Vec<&str> = entry
            .tensors
            .iter()
            .map(|tensor| tensor.name.as_str())
            .collect();
        self.reserve(path, &names, None, |totals| {
            totals.add_payload(path, &entry.tags, &entry.tensors)
        })?;
        let placed: Vec<(u64, &[u8])> = entry
            .tensors
            .iter()
            .zip(tensors)
            .map(|(entry, tensor)| (entry.offset, tensor.bytes))
            .collect();
        let digests = self.write_payload(path, entry.size, &placed)?;
        let mut state = self.state();
        state.digests.insert(path.to_owned(), digests);
        state.files.push(entry);
        Ok(())
    }

    /// Adds a group, whose instances [`Writer::add_instance`] then writes.
    ///
    /// # Errors
    ///
    /// If the group breaks a rule of the specification (its path template
    /// leaving the root, say), a group of its name was added before, its
    /// instances would pass a limit of section 2.9, or an instance's path
    /// or a tensor's name was taken before: every instance's are taken as
    /// the group is added.
    pub fn add_group(&self, spec: GroupSpec) -> Result<GroupId> {
        let mut tensors = Vec::with_capacity(spec.tensors.len());
        let mut offset = 0;
        for tensor in spec.tensors {
            let size = tensor.scheme.size(&tensor.shape).ok_or_else(|| {
                Error::Write(format!("tensor {}: its size overflows", tensor.name))
            })?;
            tensors.push(TensorEntry {
                name: tensor.name,
                dtype: tensor.dtype,
                shape: tensor.shape,
                layout: tensor.layout,
                scheme: tensor.scheme,
                offset,
                size,
            });
            offset = align(offset + size);
        }
        let entry = GroupEntry {
            name: spec.name,
            path: spec.path,
            axes: spec.axes,
            size: offset,
            tags: spec.tags,
            tensors,
        };
        check_group(&entry).map_err(|error| Error::Write(error.to_string()))?;
        let path = Template::parse(&entry.path, &entry.axes).map_err(Error::Write)?;
        let names = entry
            .tensors
            .iter()
            .map(|tensor| Template::parse(&tensor.name, &entry.axes))
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(Error::Write)?;
        let instances = entry.axes.iter().map(Axis::value_count).product();
        let mut state = self.state();
        if state
            .groups
            .iter()
            .any(|group| group.entry.name == entry.name)
        {
            return Err(Error::Write(format!(
                "a group named {:?} was added before",
                entry.name
            )));
        }
        // Within the limits first, which bound what the instances render to.
        let totals = state.totals;
        state
            .totals
            .add_group(&entry)
            .map_err(|error| Error::Write(error.to_string()))?;
        if let Err(message) = take_instances(&mut state, &entry, &path, &names) {
            state.totals = totals;
            return Err(Error::Write(format!("group {:?}: {message}", entry.name)));
        }
        state.groups.push(Group {
            entry,
            path,
            instances,
            started: BTreeSet::new(),
            written: BTreeSet::new(),
        });
        Ok(GroupId(state.groups.len() - 1))
    }

    /// Writes the instance of `group` at the axis `values`, from the bytes
    /// of each of the group's tensors, in order.
    ///
    /// # Errors
    ///
    /// If the values aren't an instance of the group, the bytes don't fit
    /// the tensors, the instance was added before, or writing fails.
    pub fn add_instance(&self, group: GroupId, values: &[u64], tensors: &[&[u8]]) -> Result<()> {
        let (path, size, placed) = {
            let mut state = self.state();
            let group = state
                .groups
                .get_mut(group.0)
                .ok_or_else(|| Error::Write("no such group".to_owned()))?;
            let entry = &group.entry;
            if values.len() != entry.axes.len()
                || !entry
                    .axes
                    .iter()
                    .zip(values)
                    .all(|(axis, &value)| axis.contains(value))
            {
                return Err(Error::Write(format!(
                    "{values:?} isn't an instance of group {:?}",
                    entry.name
                )));
            }
            if tensors.len() != entry.tensors.len()
                || entry
                    .tensors
                    .iter()
                    .zip(tensors)
                    .any(|(tensor, bytes)| tensor.size != bytes.len() as u64)
            {
                return Err(Error::Write(format!(
                    "the bytes don't fit the tensors of group {:?}",
                    entry.name
                )));
            }
            let placed: Vec<(u64, &[u8])> = entry
                .tensors
                .iter()
                .zip(tensors)
                .map(|(tensor, &bytes)| (tensor.offset, bytes))
                .collect();
            let size = entry.size;
            let path = group.path.render(values);
            // Checked before anything is written, so that no instance's
            // path leaves the root.
            check_path(&path).map_err(Error::Write)?;
            // Its path and tensor names were taken as the group was added.
            if !group.started.insert(values.to_vec()) {
                return Err(Error::Write(format!("{path} was added twice")));
            }
            (path, size, placed)
        };
        let digests = self.write_payload(&path, size, &placed)?;
        let mut state = self.state();
        state.groups[group.0].written.insert(values.to_vec());
        state.digests.insert(path, digests);
        Ok(())
    }

    /// Writes an auxiliary file.
    ///
    /// # Errors
    ///
    /// If the path is invalid or was added before, the role is empty or
    /// another file's, or writing fails.
    pub fn add_aux(&self, path: &str, role: &str, bytes: &[u8]) -> Result<()> {
        check_path(path).map_err(Error::Write)?;
        check_role(path, role).map_err(|error| Error::Write(error.to_string()))?;
        self.reserve(path, &[], Some(role), |totals| totals.add_aux(path, role))?;
        let digests = write_file(&self.root, &self.algorithms, path, |file, hasher| {
            file.write_all(bytes)?;
            hasher.update(bytes);
            Ok(())
        })?;
        let mut state = self.state();
        state.digests.insert(path.to_owned(), digests);
        state.aux.push(AuxEntry {
            path: path.to_owned(),
            size: bytes.len() as u64,
            role: role.to_owned(),
        });
        Ok(())
    }

    fn write_payload(
        &self,
        path: &str,
        size: u64,
        tensors: &[(u64, &[u8])],
    ) -> Result<Vec<FileDigests>> {
        write_file(&self.root, &self.algorithms, path, |file, hasher| {
            let mut position = 0;
            for &(offset, bytes) in tensors {
                zeros(file, hasher, offset - position)?;
                file.write_all(bytes)?;
                hasher.update(bytes);
                position = offset + bytes.len() as u64;
            }
            zeros(file, hasher, size - position)
        })
    }

    /// Writes the hash tables and then the manifest, completing the model.
    /// Each is written whole under another name and then renamed into
    /// place, and every directory is made durable before the manifest, so
    /// that a root holding a manifest holds the whole model even after a
    /// crash.
    ///
    /// # Errors
    ///
    /// If a group has instances left to write, the model breaks a rule of
    /// the specification, or writing fails.
    pub fn finish(self) -> Result<Written> {
        let Self {
            root,
            info,
            algorithms,
            state,
        } = self;
        let State {
            mut files,
            groups,
            mut aux,
            mut digests,
            written_into,
            ..
        } = state.into_inner().unwrap_or_else(PoisonError::into_inner);
        if let Some(group) = groups
            .iter()
            .find(|group| group.written.len() as u64 != group.instances)
        {
            return Err(Error::Write(format!(
                "group {:?} has {} of its {} instances written",
                group.entry.name,
                group.written.len(),
                group.instances
            )));
        }
        // Independent of the order files were added in.
        files.sort_by(|a, b| a.path.cmp(&b.path));
        aux.sort_by(|a, b| a.path.cmp(&b.path));
        let mut groups: Vec<GroupEntry> = groups.into_iter().map(|group| group.entry).collect();
        groups.sort_by(|a, b| a.name.cmp(&b.name));
        let manifest = Manifest {
            format: FORMAT.to_owned(),
            format_version: FORMAT_VERSION,
            name: info.name,
            arch: info.arch,
            schema_version: info.schema_version,
            config: info.config,
            alignment: ALIGNMENT,
            files,
            groups,
            aux,
            provenance: info.provenance,
            extensions: info.extensions,
        };
        let manifest_path = root.join(MANIFEST_PATH);
        let json = manifest.to_json();
        let table = manifest
            .file_table(json.len() as u64)
            .map_err(|source| Error::Manifest {
                path: manifest_path.clone(),
                source,
            })?;
        // The manifest's digests from its bytes in memory: it is written
        // last.
        let mut manifest_hasher = FileHasher::new(&algorithms)
            .map_err(|algorithm| Error::Write(format!("{algorithm} can't be computed here")))?;
        manifest_hasher.update(&json);
        digests.insert(MANIFEST_PATH.to_owned(), manifest_hasher.finish());

        let hashes = root.join(HASHES_DIR);
        fs::create_dir_all(&hashes).at(&hashes)?;
        let mut content_id = None;
        for &algorithm in &algorithms {
            let mut per_file = Vec::with_capacity(table.len());
            for entry in table.entries() {
                let file = digests
                    .get_mut(&entry.path)
                    .and_then(|digests| {
                        let index = digests
                            .iter()
                            .position(|digests| digests.algorithm == algorithm)?;
                        Some(digests.swap_remove(index))
                    })
                    .ok_or_else(|| Error::Write(format!("{} has no digests", entry.path)))?;
                per_file.push(file);
            }
            let path = root.join(algorithm.table_path());
            let hash_table =
                HashTable::new(algorithm, &table, per_file).map_err(|source| Error::HashTable {
                    path: path.clone(),
                    source,
                })?;
            if algorithm == Algorithm::Sha3 {
                content_id = Some(ContentId::compute(&table, &hash_table).map_err(|source| {
                    Error::HashTable {
                        path: path.clone(),
                        source,
                    }
                })?);
            }
            write_new(&path, &hash_table.encode())?;
        }
        sync_directories(&root, &written_into)?;
        write_new(&manifest_path, &json)?;
        Ok(Written {
            manifest,
            table,
            content_id,
        })
    }
}

// Syncs the directories files were written into and every directory above
// them, each once and after the directories below it, then `root`.
fn sync_directories(root: &Path, written_into: &BTreeSet<String>) -> Result<()> {
    let mut directories = BTreeSet::new();
    for directory in written_into {
        directories.insert(directory.as_str());
        for (end, _) in directory.match_indices('/') {
            directories.insert(&directory[..end]);
        }
    }
    for directory in directories.iter().rev() {
        sync_dir(&root.join(directory))?;
    }
    sync_dir(root)
}

// Whether `path`, of key `key`, and the tensor names `tensors` are free:
// no path taken names the same file where case is ignored, needs `path`
// as a directory or is one `path` needs, and no name is taken or twice in
// `tensors`.
fn check_free(
    state: &State,
    path: &str,
    key: &str,
    tensors: &[&str],
) -> std::result::Result<(), String> {
    if let Some(other) = state.paths.get(key) {
        return Err(if other == path {
            format!("{path} was added twice")
        } else {
            format!("{path} and {other} name one file where case is ignored")
        });
    }
    // No path taken is below another, so a file that this path needs as a
    // directory has the key right before its key, and the first path that
    // needs this one as a directory, if any does, the key right after.
    if let Some((_, other)) = state
        .paths
        .range::<str, _>((Bound::Unbounded, Bound::Excluded(key)))
        .next_back()
        .filter(|(before, _)| is_below(key, before))
    {
        return Err(format!(
            "{other} is a file, but {path} needs it as a directory"
        ));
    }
    if let Some((_, other)) = state
        .paths
        .range::<str, _>((Bound::Excluded(key), Bound::Unbounded))
        .next()
        .filter(|(after, _)| is_below(after, key))
    {
        return Err(format!(
            "{path} is a file, but {other} needs it as a directory"
        ));
    }
    let mut names = HashSet::with_capacity(tensors.len());
    if let Some(&name) = tensors
        .iter()
        .find(|&&name| state.tensors.contains(name) || !names.insert(name))
    {
        return Err(format!("two tensors are named {name:?}"));
    }
    Ok(())
}

// Takes the path and tensor names of every instance of the group `entry`,
// each checked against all taken before it, the group's earlier instances
// included; on a conflict, none of them.
fn take_instances(
    state: &mut State,
    entry: &GroupEntry,
    path: &Template,
    names: &[Template],
) -> std::result::Result<(), String> {
    let mut parents = BTreeSet::new();
    let mut taken = 0;
    let result = instances(&entry.axes).try_for_each(|values| {
        let instance = path.render(&values);
        let key = path_key(&instance);
        let tensors: Vec<String> = names.iter().map(|name| name.render(&values)).collect();
        let free: Vec<&str> = tensors.iter().map(String::as_str).collect();
        check_free(state, &instance, &key, &free)?;
        if let Some((parent, _)) = instance.rsplit_once('/') {
            parents.insert(parent.to_owned());
        }
        state.paths.insert(key, instance);
        state.tensors.extend(tensors);
        taken += 1;
        Ok(())
    });
    if result.is_ok() {
        state.written_into.extend(parents);
    } else {
        for values in instances(&entry.axes).take(taken) {
            state.paths.remove(&path_key(&path.render(&values)));
            for name in names {
                state.tensors.remove(&name.render(&values));
            }
        }
    }
    result
}

/// Writes the new file `path` under `root`, hashing its contents.
fn write_file(
    root: &Path,
    algorithms: &[Algorithm],
    path: &str,
    contents: impl FnOnce(&mut File, &mut FileHasher) -> std::io::Result<()>,
) -> Result<Vec<FileDigests>> {
    let full = root.join(path);
    if let Some(parent) = full.parent() {
        fs::create_dir_all(parent).at(parent)?;
    }
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&full)
        .at(&full)?;
    let mut hasher = FileHasher::new(algorithms)
        .map_err(|algorithm| Error::Write(format!("{algorithm} can't be computed here")))?;
    contents(&mut file, &mut hasher).at(&full)?;
    // Durable before the manifest, which only the end writes: a model with
    // a manifest has all of its files.
    file.sync_all().at(&full)?;
    Ok(hasher.finish())
}

/// Writes the new file `path` whole and durably, or not at all: its bytes
/// go to a file of this write's own, `path` with
/// `.partial-<process id>-<number>` appended, which only a crash leaves
/// behind and nothing reads, and that is renamed to `path` only if no file
/// has its name.
pub(crate) fn write_new(path: &Path, bytes: &[u8]) -> Result<()> {
    place_new(path, |file, partial| file.write_all(bytes).at(partial))?;
    sync_dir(
        path.parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .unwrap_or(Path::new(".")),
    )
}

/// Puts the new file `path` in place whole and durably, or not at all, as
/// [`write_new`] does but for syncing its directory: `contents` writes it
/// into a file of this write's own, which is synced and renamed to `path`
/// only if no file has its name, and removed if any of that fails.
pub(crate) fn place_new(
    path: &Path,
    contents: impl FnOnce(&mut File, &Path) -> Result<()>,
) -> Result<()> {
    let (partial, file) = create_partial(path)?;
    let placed = fill_and_rename(file, &partial, path, contents);
    if placed.is_err() {
        // This write's own file, which no other write has: the attempt is
        // all that removing it can fail.
        let _ = fs::remove_file(&partial);
    }
    placed
}

// A file for `path`'s bytes until they are whole, of this write's own: its
// name has the process's id and a number no other write of the process
// has, and it is created new, so that a file a crash left behind under the
// name isn't written into.
fn create_partial(path: &Path) -> Result<(PathBuf, File)> {
    static WRITES: AtomicU64 = AtomicU64::new(0);
    let mut taken = 0;
    loop {
        let mut name = path.as_os_str().to_owned();
        name.push(format!(
            ".partial-{}-{}",
            std::process::id(),
            WRITES.fetch_add(1, Ordering::Relaxed)
        ));
        let partial = PathBuf::from(name);
        match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&partial)
        {
            Ok(file) => return Ok((partial, file)),
            // Left by a crashed process that had this one's id.
            Err(source) if source.kind() == std::io::ErrorKind::AlreadyExists && taken < 1000 => {
                taken += 1;
            }
            Err(source) => {
                return Err(Error::Io {
                    path: partial,
                    source,
                });
            }
        }
    }
}

fn fill_and_rename(
    mut file: File,
    partial: &Path,
    path: &Path,
    contents: impl FnOnce(&mut File, &Path) -> Result<()>,
) -> Result<()> {
    contents(&mut file, partial)?;
    file.sync_all().at(partial)?;
    drop(file);
    rename_new(partial, path)
}

// Renames `from` to `to` if no file has `to`'s name: a new hard link fails
// if one does, and the link then stands for the rename. On file systems
// without hard links, a rename once `to` is found free.
fn rename_new(from: &Path, to: &Path) -> Result<()> {
    match fs::hard_link(from, to) {
        Ok(()) => fs::remove_file(from).at(from),
        Err(source) if source.kind() == std::io::ErrorKind::AlreadyExists => Err(Error::Io {
            path: to.to_path_buf(),
            source,
        }),
        Err(_) if !to.exists() => fs::rename(from, to).at(to),
        Err(source) => Err(Error::Io {
            path: to.to_path_buf(),
            source,
        }),
    }
}

/// Makes a directory's entries durable.
pub(crate) fn sync_dir(path: &Path) -> Result<()> {
    File::open(path).and_then(|dir| dir.sync_all()).at(path)
}

const fn align(offset: u64) -> u64 {
    offset.next_multiple_of(ALIGNMENT)
}

fn zeros(file: &mut File, hasher: &mut FileHasher, mut count: u64) -> std::io::Result<()> {
    const ZEROS: [u8; 16384] = [0; 16384];
    while count > 0 {
        let n = count.min(ZEROS.len() as u64);
        let bytes = &ZEROS[..usize::try_from(n).unwrap_or(ZEROS.len())];
        file.write_all(bytes)?;
        hasher.update(bytes);
        count -= n;
    }
    Ok(())
}

#[cfg(test)]
mod tests;
