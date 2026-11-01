// SPDX-FileCopyrightText: 2026 Mika Tammi
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Opening a model from its roots ([the specification](crate::spec)'s
//! section 7), and reading it with every chunk verified (section 8).

use std::collections::BTreeSet;
use std::fs;
use std::io;
use std::ops::{Deref, Range};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use crate::error::{Error, IoContext as _, Result};
use crate::hash::{
    Algorithm, CHUNK_SIZE, ContentId, FileDigests, FileHasher, HashTable, chunk_count,
};
use crate::io::{AlignedBuf, CacheMode, DIRECT_ALIGNMENT, ReadFile};
use crate::manifest::{EntryKind, FileTable, HASHES_DIR, MANIFEST_PATH, Manifest, PART_PATH};
use crate::part::Part;
use crate::writer::{sync_dir, write_new};

/// A complete model, opened from one or more roots.
#[derive(Debug)]
pub struct Model {
    roots: Vec<PathBuf>,
    manifest: Manifest,
    table: FileTable,
    // The index in `roots` of the root each file is read from.
    locations: Vec<usize>,
    tables: Vec<HashTable>,
    parts: Vec<Part>,
    content_id: Option<ContentId>,
}

/// What vouches for a first table of a model that has none to check its
/// files against ([`Model::add_first_table`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Vouch {
    /// The model's content id, from a source trusted for it such as the
    /// model's publisher: the sha3-512 table computed from the files is
    /// written only if it gives this id, which makes the files, the
    /// manifest among them, the model's.
    ContentId(ContentId),
    /// The files as they are, for a model with no table at all: the table
    /// then vouches for nothing but the bytes the files hold.
    Files,
}

/// A file whose bytes don't match a digest.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Mismatch {
    /// The file's path.
    pub path: String,
    /// The algorithm of the digest.
    pub algorithm: Algorithm,
    /// The chunk, or `None` for the file's digest.
    pub chunk: Option<u64>,
}

impl Model {
    /// Opens the model whose roots are `roots`, in the order files are read
    /// from them. Checks the manifests, that the model is complete, and its
    /// hash tables, and verifies the manifest with the fastest table.
    ///
    /// Every hash table must be valid, whichever it is: a model with one
    /// that isn't doesn't open, even when it is a table that reads never
    /// use.
    ///
    /// # Errors
    ///
    /// If a root has no manifest or a different one, a file is missing or
    /// has another size than the manifest gives, a hash table is invalid
    /// ([`Error::HashTable`]) or differs between roots
    /// ([`Error::TablesDiffer`]), a `part.json` is invalid or differs
    /// between roots, the model has no table this build can compute
    /// ([`Error::Unverifiable`]), or the manifest doesn't match its digest.
    pub fn open<P: AsRef<Path>>(roots: &[P]) -> Result<Self> {
        let (model, manifest_bytes) = Self::gather(roots)?;
        if !model
            .tables
            .iter()
            .any(|table| table.algorithm().is_supported())
        {
            return Err(Error::Unverifiable(if model.tables.is_empty() {
                "it has no hash table".to_owned()
            } else {
                "its only table is gxhash3-128's, which needs AES instructions".to_owned()
            }));
        }
        model.check_parts()?;
        let index = model
            .table
            .index_of(MANIFEST_PATH)
            .ok_or_else(|| Error::Roots("the file table has no manifest".to_owned()))?;
        model.check_bytes(index, &manifest_bytes)?;
        Ok(model)
    }

    /// Adds a first table of `algorithm` to the model whose roots are
    /// `roots`, a model with no table this build can compute and so none to
    /// check its files against, and writes it into `root`, one of the
    /// roots. `vouch` vouches for the table instead (section 5.2 of the
    /// specification). The manifest's digest is of the bytes the file table
    /// was read from. Opening the model then finds the table, and further
    /// tables are added with [`Model::add_table`]; as there, the table goes
    /// into `root` alone, and belongs in a model's other parts too.
    ///
    /// # Errors
    ///
    /// If the model has a table this build can compute, `root` isn't one of
    /// its roots, a content id vouches for another table than sha3-512's,
    /// the files vouch for a model that has a table, the files give another
    /// content id than the one vouching ([`Error::ContentIdDiffers`]), parts
    /// name another, or opening, reading or writing fails as in
    /// [`Model::open`] and [`Model::add_table`].
    pub fn add_first_table<P: AsRef<Path>>(
        roots: &[P],
        root: &Path,
        algorithm: Algorithm,
        vouch: Vouch,
        mode: CacheMode,
        threads: usize,
        progress: &(dyn Fn(u64) + Sync),
    ) -> Result<PathBuf> {
        let (mut model, manifest_bytes) = Self::gather(roots)?;
        if model
            .tables
            .iter()
            .any(|table| table.algorithm().is_supported())
        {
            return Err(Error::Usage(
                "the model has a table this build can check its files against: add tables to it \
                 with Model::add_table"
                    .to_owned(),
            ));
        }
        if !model.roots.iter().any(|known| known == root) {
            return Err(Error::Usage(format!(
                "{} isn't one of the model's roots",
                root.display()
            )));
        }
        match vouch {
            Vouch::ContentId(_) if algorithm != Algorithm::Sha3 => {
                return Err(Error::Usage(
                    "a content id vouches for the sha3-512 table it is computed from".to_owned(),
                ));
            }
            Vouch::Files if !model.tables.is_empty() => {
                return Err(Error::Usage(
                    "the model has a table that can't be computed here: vouch for a first \
                     table with the model's content id, or add tables where it can be"
                        .to_owned(),
                ));
            }
            Vouch::Files if !model.parts.is_empty() && algorithm != Algorithm::Sha3 => {
                return Err(Error::Usage(
                    "a model in parts needs its sha3-512 table first, for the content id its \
                     parts name"
                        .to_owned(),
                ));
            }
            _ => {}
        }
        let mut digests: Vec<FileDigests> = model
            .digests(&[algorithm], mode, threads, progress, None)?
            .into_iter()
            .map(|mut digests| digests.remove(0))
            .collect();
        // The manifest's of the bytes in hand, which the file table was
        // built from, rather than of the file read once more.
        let manifest = model
            .table
            .index_of(MANIFEST_PATH)
            .ok_or_else(|| Error::Roots("the file table has no manifest".to_owned()))?;
        let mut manifest_hasher = FileHasher::new(&[algorithm]).map_err(Error::NoTable)?;
        manifest_hasher.update(&manifest_bytes);
        digests[manifest] = manifest_hasher.finish().remove(0);
        let path = root.join(algorithm.table_path());
        let table = HashTable::new(algorithm, &model.table, digests).map_err(|source| {
            Error::HashTable {
                path: path.clone(),
                source,
            }
        })?;
        if algorithm == Algorithm::Sha3 {
            let found =
                ContentId::compute(&model.table, &table).map_err(|source| Error::HashTable {
                    path: path.clone(),
                    source,
                })?;
            if let Vouch::ContentId(expected) = vouch
                && found != expected
            {
                return Err(Error::ContentIdDiffers {
                    expected: Box::new(expected),
                    found: Box::new(found),
                });
            }
            model.content_id = Some(found);
            model.check_parts()?;
        }
        let hashes = root.join(HASHES_DIR);
        fs::create_dir_all(&hashes).at(&hashes)?;
        write_new(&path, &table.encode())?;
        sync_dir(root)?;
        Ok(path)
    }

    // The model whose roots are `roots` as its files say, with the bytes of
    // its manifest: complete, its manifests one and its tables valid for
    // its file table, but nothing verified yet.
    fn gather<P: AsRef<Path>>(roots: &[P]) -> Result<(Self, Vec<u8>)> {
        let roots: Vec<PathBuf> = roots
            .iter()
            .map(|root| root.as_ref().to_path_buf())
            .collect();
        let Some(first) = roots.first() else {
            return Err(Error::Roots("a model needs a root".to_owned()));
        };
        let manifest_bytes = read_manifest(first)?;
        for root in &roots[1..] {
            if read_manifest(root)? != manifest_bytes {
                return Err(Error::ManifestsDiffer {
                    first: first.clone(),
                    other: root.clone(),
                });
            }
        }
        let (manifest, table) =
            Manifest::parse(&manifest_bytes).map_err(|source| Error::Manifest {
                path: first.join(MANIFEST_PATH),
                source,
            })?;

        let (locations, missing) = locate(&roots, &table)?;
        let parts = read_parts(&roots, &table)?;
        if !missing.is_empty() {
            return Err(incomplete(&table, &missing, &parts));
        }
        let tables = read_tables(&roots, &table)?;
        let content_id = tables
            .iter()
            .find(|table| table.algorithm() == Algorithm::Sha3)
            .map(|sha3| ContentId::compute(&table, sha3))
            .transpose()
            .map_err(|source| Error::HashTable {
                path: first.join(Algorithm::Sha3.table_path()),
                source,
            })?;
        let model = Self {
            roots,
            manifest,
            table,
            locations,
            tables,
            parts,
            content_id,
        };
        Ok((model, manifest_bytes))
    }

    // That every part is one of this model's, of its content id.
    fn check_parts(&self) -> Result<()> {
        let content_id = self.content_id.map(|id| id.to_string());
        if let Some(part) = self
            .parts
            .iter()
            .find(|part| Some(part.content_id.as_str()) != content_id.as_deref())
        {
            return Err(Error::Roots(format!(
                "part {} of {} is of a model with another content id, or the model has no \
                 sha3-512 table",
                part.part, part.parts
            )));
        }
        Ok(())
    }

    /// The roots, in the order files are read from them.
    #[must_use]
    pub fn roots(&self) -> &[PathBuf] {
        &self.roots
    }

    /// The manifest.
    #[must_use]
    pub fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    /// The file table.
    #[must_use]
    pub fn table(&self) -> &FileTable {
        &self.table
    }

    /// Where the file of index `file` is read from.
    #[must_use]
    pub fn path(&self, file: usize) -> PathBuf {
        self.roots[self.locations[file]].join(&self.table.entries()[file].path)
    }

    /// The hash tables present, by algorithm.
    #[must_use]
    pub fn hash_tables(&self) -> &[HashTable] {
        &self.tables
    }

    /// The hash table of `algorithm`, if the model has it.
    #[must_use]
    pub fn hash_table(&self, algorithm: Algorithm) -> Option<&HashTable> {
        self.tables
            .iter()
            .find(|table| table.algorithm() == algorithm)
    }

    /// The content id the model's sha3-512 table gives, if it has one. It
    /// is that table's claim, which opening doesn't check: the id is the
    /// files' only once the table has been checked against them
    /// ([`Model::verify`] with [`Algorithm::Sha3`]), and they are the model
    /// a trusted source names only if that source's id is this one
    /// (section 8 of the specification).
    #[must_use]
    pub fn content_id(&self) -> Option<ContentId> {
        self.content_id
    }

    /// The `part.json` of every root that is a delivery part.
    #[must_use]
    pub fn parts(&self) -> &[Part] {
        &self.parts
    }

    /// The algorithm chunks are verified with: the fastest chunked one the
    /// model has a table of and this build supports.
    ///
    /// # Errors
    ///
    /// [`Error::NoVerifier`] if there is none.
    pub fn verifier(&self) -> Result<Algorithm> {
        Algorithm::fastest_chunked(
            self.tables.iter().map(HashTable::algorithm),
            Algorithm::is_supported,
        )
        .ok_or_else(|| {
            Error::NoVerifier(if self.hash_table(Algorithm::Gxhash3).is_some() {
                "it has gxhash3-128, which needs AES instructions, and no xxh3-128".to_owned()
            } else {
                "it has neither gxhash3-128 nor xxh3-128".to_owned()
            })
        })
    }

    /// A reader of the model's files, through the page cache or around it,
    /// that verifies every chunk with [`Model::verifier`]'s table.
    ///
    /// # Errors
    ///
    /// [`Error::NoVerifier`] if the model has no table to verify with.
    pub fn reader(&self, mode: CacheMode) -> Result<Reader<&Self>> {
        Reader::new(self, mode)
    }

    /// A reader as [`Model::reader`] gives, that holds the model, as the
    /// files it opens do, rather than borrowing it: they are `Send`, `Sync`
    /// and `'static`, for threads and tasks a borrow wouldn't outlive.
    ///
    /// # Errors
    ///
    /// As [`Model::reader`].
    pub fn shared_reader(self: &Arc<Self>, mode: CacheMode) -> Result<Reader<Arc<Self>>> {
        Reader::new(Arc::clone(self), mode)
    }

    /// Where the tensor `name` is, to read it with: `None` if the model has
    /// no tensor of that name.
    #[must_use]
    pub fn tensor_location(&self, name: &str) -> Option<TensorLocation> {
        let (file, tensor) = self.table.tensor(name)?;
        let bytes = tensor.offset..tensor.offset + tensor.size;
        let chunks = chunks_covering(&bytes);
        let span = chunk_span(&chunks, self.table.entries()[file].size);
        let within = if bytes.is_empty() {
            0..0
        } else {
            usize::try_from(bytes.start - span.start).ok()?
                ..usize::try_from(bytes.end - span.start).ok()?
        };
        Some(TensorLocation {
            file,
            chunks,
            span,
            within,
        })
    }

    // Verifies bytes of the file of index `file` that are already in memory,
    // with the verifier's table if there is one, or else with any table of
    // an algorithm this build supports, which `open` made sure of.
    fn check_bytes(&self, file: usize, bytes: &[u8]) -> Result<()> {
        let path = &self.table.entries()[file].path;
        if let Ok(algorithm) = self.verifier() {
            let table = self
                .hash_table(algorithm)
                .ok_or(Error::NoTable(algorithm))?;
            let chunk = usize::try_from(CHUNK_SIZE).unwrap_or(usize::MAX);
            for (index, (bytes, digest)) in bytes.chunks(chunk).zip(table.chunks(file)).enumerate()
            {
                if algorithm.digest(bytes).as_ref() != Some(digest) {
                    return Err(Error::Mismatch {
                        path: path.clone(),
                        chunk: Some(index as u64),
                        algorithm,
                    });
                }
            }
        } else {
            let table = self
                .tables
                .iter()
                .find(|table| table.algorithm().is_supported())
                .ok_or_else(|| {
                    Error::Unverifiable("it has no table this build can compute".to_owned())
                })?;
            let algorithm = table.algorithm();
            let mut hasher = FileHasher::new(&[algorithm]).map_err(Error::NoTable)?;
            hasher.update(bytes);
            if hasher.finish().first().map(|digests| &digests.file) != Some(table.file(file)) {
                return Err(Error::Mismatch {
                    path: path.clone(),
                    chunk: None,
                    algorithm,
                });
            }
        }
        Ok(())
    }

    /// Reads every file and checks it against the tables of `algorithms`,
    /// using `threads` threads, and calls `progress` with the number of
    /// bytes after every chunk read. Payload files are read in `mode`, the
    /// others through the page cache. Around it, a file's pages are dropped
    /// from the cache before it is read, so that what is checked is the
    /// storage.
    ///
    /// # Errors
    ///
    /// If the model has no table of an algorithm asked for, this build
    /// doesn't support one, or a read fails. Files that don't match aren't
    /// errors but the returned mismatches, sorted.
    pub fn verify(
        &self,
        algorithms: &[Algorithm],
        mode: CacheMode,
        threads: usize,
        progress: &(dyn Fn(u64) + Sync),
    ) -> Result<Vec<Mismatch>> {
        let tables = algorithms
            .iter()
            .map(|&algorithm| self.hash_table(algorithm).ok_or(Error::NoTable(algorithm)))
            .collect::<Result<Vec<_>>>()?;
        let digests = self.digests(algorithms, mode, threads, progress, None)?;
        let mut mismatches = Vec::new();
        for (file, digests) in digests.iter().enumerate() {
            let path = &self.table.entries()[file].path;
            for (digests, table) in digests.iter().zip(&tables) {
                let algorithm = digests.algorithm;
                for (chunk, (computed, stored)) in
                    digests.chunks.iter().zip(table.chunks(file)).enumerate()
                {
                    if computed != stored {
                        mismatches.push(Mismatch {
                            path: path.clone(),
                            algorithm,
                            chunk: Some(chunk as u64),
                        });
                    }
                }
                if &digests.file != table.file(file) {
                    mismatches.push(Mismatch {
                        path: path.clone(),
                        algorithm,
                        chunk: None,
                    });
                }
            }
        }
        mismatches.sort();
        Ok(mismatches)
    }

    /// Computes the table of `algorithm`, which the model doesn't have yet,
    /// from its files, and writes it into `root`, one of the model's roots.
    /// The files are read in `mode` as [`Model::verify`] reads them, and
    /// every file is checked as it is read against a table the model has,
    /// the one chunks are verified with if there is one, so that the new
    /// table never vouches for bytes the model's others don't; the table is
    /// written whole or not at all. Opening the model again finds it. The
    /// table goes into `root` alone: a model in parts has every table in
    /// every part (section 7 of the specification), so it belongs in the
    /// other parts' roots too.
    ///
    /// # Errors
    ///
    /// If the model has the table already, `root` isn't one of its roots,
    /// this build doesn't support the algorithm, a file doesn't match the
    /// model's table ([`Error::Mismatch`]), or reading or writing fails.
    pub fn add_table(
        &self,
        algorithm: Algorithm,
        root: &Path,
        mode: CacheMode,
        threads: usize,
        progress: &(dyn Fn(u64) + Sync),
    ) -> Result<PathBuf> {
        if self.hash_table(algorithm).is_some() {
            return Err(Error::Usage(format!(
                "the model has a {algorithm} table already"
            )));
        }
        if !self.roots.iter().any(|known| known == root) {
            return Err(Error::Usage(format!(
                "{} isn't one of the model's roots",
                root.display()
            )));
        }
        let check = self
            .verifier()
            .ok()
            .and_then(|algorithm| self.hash_table(algorithm))
            .or_else(|| {
                self.tables
                    .iter()
                    .find(|table| table.algorithm().is_supported())
            })
            .ok_or_else(|| {
                Error::Unverifiable("it has no table this build can compute".to_owned())
            })?;
        let digests = self
            .digests(
                &[algorithm, check.algorithm()],
                mode,
                threads,
                progress,
                Some(check),
            )?
            .into_iter()
            .map(|mut digests| digests.remove(0))
            .collect();
        let path = root.join(algorithm.table_path());
        let table =
            HashTable::new(algorithm, &self.table, digests).map_err(|source| Error::HashTable {
                path: path.clone(),
                source,
            })?;
        let hashes = root.join(HASHES_DIR);
        fs::create_dir_all(&hashes).at(&hashes)?;
        write_new(&path, &table.encode())?;
        sync_dir(root)?;
        Ok(path)
    }

    /// Every file's digests by `algorithms`, in file-table order, computed
    /// with `threads` threads; with `check`, the table of one of them, each
    /// file's digests by its algorithm must match it, or the first that
    /// doesn't stops them all.
    fn digests(
        &self,
        algorithms: &[Algorithm],
        mode: CacheMode,
        threads: usize,
        progress: &(dyn Fn(u64) + Sync),
        check: Option<&HashTable>,
    ) -> Result<Vec<Vec<FileDigests>>> {
        if let Some(&algorithm) = algorithms
            .iter()
            .find(|algorithm| !algorithm.is_supported())
        {
            return Err(Error::NoVerifier(format!(
                "{algorithm} can't be computed here: gxhash3-128 needs AES instructions, as \
                 target features of the build and in the CPU"
            )));
        }
        let next = AtomicUsize::new(0);
        let failed = AtomicBool::new(false);
        let results = Mutex::new(vec![Vec::new(); self.table.len()]);
        let error = Mutex::new(None);
        std::thread::scope(|scope| {
            for _ in 0..threads.clamp(1, self.table.len()) {
                scope.spawn(|| {
                    let mut buffer = AlignedBuf::new(usize::try_from(CHUNK_SIZE).unwrap_or(0));
                    while !failed.load(Ordering::Relaxed) {
                        let file = next.fetch_add(1, Ordering::Relaxed);
                        if file >= self.table.len() {
                            break;
                        }
                        let digests = self
                            .digest_file(file, algorithms, mode, &mut buffer, progress)
                            .and_then(|digests| {
                                if let Some(table) = check {
                                    self.check_digests(file, &digests, table)?;
                                }
                                Ok(digests)
                            });
                        match digests {
                            Ok(digests) => lock(&results)[file] = digests,
                            Err(found) => {
                                failed.store(true, Ordering::Relaxed);
                                lock(&error).get_or_insert(found);
                            }
                        }
                    }
                });
            }
        });
        match error.into_inner().unwrap_or_else(PoisonError::into_inner) {
            Some(error) => Err(error),
            None => Ok(results.into_inner().unwrap_or_else(PoisonError::into_inner)),
        }
    }

    // Checks the file of index `file`'s digests by `table`'s algorithm,
    // among `digests`, against `table`.
    fn check_digests(&self, file: usize, digests: &[FileDigests], table: &HashTable) -> Result<()> {
        let algorithm = table.algorithm();
        let path = &self.table.entries()[file].path;
        let computed = digests
            .iter()
            .find(|digests| digests.algorithm == algorithm)
            .ok_or(Error::NoTable(algorithm))?;
        if let Some(chunk) = computed
            .chunks
            .iter()
            .zip(table.chunks(file))
            .position(|(computed, stored)| computed != stored)
        {
            return Err(Error::Mismatch {
                path: path.clone(),
                chunk: Some(chunk as u64),
                algorithm,
            });
        }
        if &computed.file != table.file(file) {
            return Err(Error::Mismatch {
                path: path.clone(),
                chunk: None,
                algorithm,
            });
        }
        Ok(())
    }

    fn digest_file(
        &self,
        file: usize,
        algorithms: &[Algorithm],
        mode: CacheMode,
        buffer: &mut AlignedBuf,
        progress: &(dyn Fn(u64) + Sync),
    ) -> Result<Vec<FileDigests>> {
        let entry = &self.table.entries()[file];
        let path = self.path(file);
        let mode = match entry.kind {
            EntryKind::Payload(_) => mode,
            EntryKind::Manifest | EntryKind::Aux { .. } => CacheMode::Use,
        };
        let reader = ReadFile::open(&path, mode).at(&path)?;
        // Around the cache, the storage is what is checked: on macOS a read
        // around it still returns the pages it holds. Pages the system
        // keeps are read as they are.
        if reader.mode() == CacheMode::Bypass {
            let _ = reader.evict(0, entry.size);
        }
        let mut hasher = FileHasher::new(algorithms).map_err(Error::NoTable)?;
        let mut offset = 0;
        while offset < entry.size {
            let len = usize::try_from((entry.size - offset).min(CHUNK_SIZE)).unwrap_or(0);
            reader.read_exact_at(offset, &mut buffer[..len]).at(&path)?;
            hasher.update(&buffer[..len]);
            offset += len as u64;
            progress(len as u64);
        }
        // In the order of `algorithms`, which FileHasher doesn't keep.
        let mut digests = hasher.finish();
        digests.sort_by_key(|digests| {
            algorithms
                .iter()
                .position(|&algorithm| algorithm == digests.algorithm)
        });
        Ok(digests)
    }
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

fn read_manifest(root: &Path) -> Result<Vec<u8>> {
    let path = root.join(MANIFEST_PATH);
    match fs::read(&path) {
        Ok(bytes) => Ok(bytes),
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            Err(Error::NotARoot(root.to_path_buf()))
        }
        Err(source) => Err(Error::Io { path, source }),
    }
}

// The root each file is read from, the first that has it, and the files no
// root has.
fn locate(roots: &[PathBuf], table: &FileTable) -> Result<(Vec<usize>, Vec<usize>)> {
    let mut locations = Vec::with_capacity(table.len());
    let mut missing = Vec::new();
    for (index, entry) in table.entries().iter().enumerate() {
        let mut found = None;
        for (root_index, root) in roots.iter().enumerate() {
            let path = root.join(&entry.path);
            match fs::metadata(&path) {
                Ok(metadata) if metadata.is_file() && metadata.len() == entry.size => {
                    found.get_or_insert(root_index);
                }
                Ok(metadata) => {
                    return Err(Error::WrongSize {
                        path,
                        found: metadata.is_file().then_some(metadata.len()),
                        expected: entry.size,
                    });
                }
                Err(error) if error.kind() == io::ErrorKind::NotFound => {}
                Err(source) => return Err(Error::Io { path, source }),
            }
        }
        if found.is_none() {
            missing.push(index);
        }
        locations.push(found.unwrap_or(0));
    }
    Ok((locations, missing))
}

fn read_parts(roots: &[PathBuf], table: &FileTable) -> Result<Vec<Part>> {
    let mut parts: Vec<Part> = Vec::new();
    for root in roots {
        let path = root.join(PART_PATH);
        let bytes = match fs::read(&path) {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == io::ErrorKind::NotFound => continue,
            Err(source) => return Err(Error::Io { path, source }),
        };
        let part = Part::parse(&bytes, table.len())
            .map_err(|message| Error::Roots(format!("{}: {message}", path.display())))?;
        if let Some(other) = parts
            .iter()
            .find(|other| other.parts != part.parts || other.part == part.part)
        {
            return Err(Error::Roots(format!(
                "{}: part {} of {} doesn't go with part {} of {}",
                path.display(),
                part.part,
                part.parts,
                other.part,
                other.parts
            )));
        }
        parts.push(part);
    }
    parts.sort_by_key(|part| part.part);
    Ok(parts)
}

fn incomplete(table: &FileTable, missing: &[usize], parts: &[Part]) -> Error {
    let present: BTreeSet<u32> = parts.iter().map(|part| part.part).collect();
    let total = parts.first().map(|part| part.parts);
    Error::Incomplete {
        missing: missing.len(),
        files: table.len(),
        examples: missing
            .iter()
            .take(5)
            .map(|&index| table.entries()[index].path.clone())
            .collect(),
        missing_parts: (1..=total.unwrap_or(0))
            .filter(|number| !present.contains(number))
            .collect(),
        parts: total,
    }
}

// The model's hash tables, each read from every root that holds it and
// checked (section 5.2).
fn read_tables(roots: &[PathBuf], table: &FileTable) -> Result<Vec<HashTable>> {
    let mut tables = Vec::new();
    for algorithm in Algorithm::ALL {
        tables.extend(read_table(roots, table, algorithm)?);
    }
    Ok(tables)
}

// The table of `algorithm`, if a root holds it: the same in every root
// that does, and checked.
fn read_table(
    roots: &[PathBuf],
    table: &FileTable,
    algorithm: Algorithm,
) -> Result<Option<HashTable>> {
    let copies = table_copies(roots, algorithm)?;
    let Some((first, bytes)) = copies.first() else {
        return Ok(None);
    };
    if let Some((other, _)) = copies.iter().find(|(_, other)| other != bytes) {
        return Err(Error::TablesDiffer {
            algorithm,
            first: first.clone(),
            other: other.clone(),
        });
    }
    HashTable::decode(bytes, algorithm, table)
        .map(Some)
        .map_err(|source| Error::HashTable {
            path: first.clone(),
            source,
        })
}

// The file of the table of `algorithm` in every root that holds one, with
// its bytes.
fn table_copies(roots: &[PathBuf], algorithm: Algorithm) -> Result<Vec<(PathBuf, Vec<u8>)>> {
    let mut copies = Vec::new();
    for root in roots {
        let path = root.join(algorithm.table_path());
        match fs::read(&path) {
            Ok(bytes) => copies.push((path, bytes)),
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(source) => return Err(Error::Io { path, source }),
        }
    }
    Ok(copies)
}

/// Where a tensor's bytes are, to read them verified with
/// [`OpenFile::read_chunks`]: the file of the file table that holds them,
/// the chunks that cover them, those chunks' span of the file, and the
/// tensor's bytes within a buffer that holds the span.
///
/// Chunks are what is verified, so reading a tensor reads the whole chunks
/// around it, up to 2 MiB less 16 KiB more on either side. A span starts at
/// a chunk and a tensor at an offset aligned to 16 KiB, so `within` starts
/// at a multiple of 16 KiB of the buffer, as a GPU may want a buffer's
/// contents aligned. A tensor of no bytes has no chunks, and lies at `0..0`
/// of an empty buffer.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct TensorLocation {
    /// The file's index in the file table.
    pub file: usize,
    /// The chunks that cover the tensor.
    pub chunks: Range<u64>,
    /// The bytes of the file the chunks span.
    pub span: Range<u64>,
    /// The tensor's bytes within a buffer that holds the span.
    pub within: Range<usize>,
}

/// Reads a model's files, verifying every chunk. `M` is how it holds the
/// model: `&Model` from [`Model::reader`], or `Arc<Model>` from
/// [`Model::shared_reader`], which the reader and the files it opens can
/// take to threads and tasks of their own.
#[derive(Debug, Clone, Copy)]
pub struct Reader<M> {
    model: M,
    // The index in the model's tables of the one chunks are verified with.
    table: usize,
    mode: CacheMode,
}

impl<M: Deref<Target = Model> + Clone> Reader<M> {
    fn new(model: M, mode: CacheMode) -> Result<Self> {
        let algorithm = model.verifier()?;
        let table = model
            .tables
            .iter()
            .position(|table| table.algorithm() == algorithm)
            .ok_or(Error::NoTable(algorithm))?;
        Ok(Self { model, table, mode })
    }

    /// The algorithm chunks are verified with.
    #[must_use]
    pub fn algorithm(&self) -> Algorithm {
        self.model.tables[self.table].algorithm()
    }

    /// Opens the file of index `file`.
    ///
    /// # Errors
    ///
    /// If `file` isn't an index of the file table, or opening fails.
    pub fn open(&self, file: usize) -> Result<OpenFile<M>> {
        if file >= self.model.table.len() {
            return Err(Error::Usage(format!(
                "the model has no file of index {file}"
            )));
        }
        let path = self.model.path(file);
        let mode = match self.model.table.entries()[file].kind {
            EntryKind::Payload(_) => self.mode,
            // Their sizes aren't aligned for direct reads.
            EntryKind::Manifest | EntryKind::Aux { .. } => CacheMode::Use,
        };
        Ok(OpenFile {
            model: self.model.clone(),
            table: self.table,
            index: file,
            file: ReadFile::open(&path, mode).at(&path)?,
            path,
        })
    }
}

/// A file of a model, open for verified reads, holding the model as the
/// [`Reader`] that opened it does.
#[derive(Debug)]
pub struct OpenFile<M> {
    model: M,
    // As the reader's.
    table: usize,
    index: usize,
    file: ReadFile,
    path: PathBuf,
}

impl<M: Deref<Target = Model>> OpenFile<M> {
    /// How the file's reads go in fact: around the page cache only if the
    /// file system allows it.
    #[must_use]
    pub fn mode(&self) -> CacheMode {
        self.file.mode()
    }

    /// The file's size in bytes.
    #[must_use]
    pub fn size(&self) -> u64 {
        self.model.table.entries()[self.index].size
    }

    /// The bytes the chunks `chunks` span: from the first chunk's start to the
    /// end of the last, or of the file.
    #[must_use]
    pub fn span(&self, chunks: &Range<u64>) -> Range<u64> {
        chunk_span(chunks, self.size())
    }

    /// The chunks that cover `bytes` of the file, none if it is empty: the
    /// ones to read them with [`OpenFile::read_chunks`].
    #[must_use]
    pub fn chunks_covering(&self, bytes: &Range<u64>) -> Range<u64> {
        chunks_covering(bytes)
    }

    /// Reads the chunks `chunks` into `dst`, which must be exactly as long
    /// as [`OpenFile::span`] says, and verifies each. A payload file's
    /// reader in [`CacheMode::Bypass`] needs `dst` to start at an address
    /// aligned to [`DIRECT_ALIGNMENT`], as an [`AlignedBuf`]'s does. A chunk
    /// that doesn't match is read once more from the storage, its copy in
    /// the page cache dropped first as far as the system allows; if it still
    /// doesn't match, the read fails.
    ///
    /// # Errors
    ///
    /// [`Error::Usage`] if the chunks aren't the file's or `dst` has another
    /// length, [`Error::Mismatch`] if a chunk doesn't match twice, and
    /// [`Error::Io`] if reading fails, or `dst` isn't aligned as it must be.
    pub fn read_chunks(&self, chunks: Range<u64>, dst: &mut [u8]) -> Result<()> {
        let span = self.span(&chunks);
        if chunks.start > chunks.end
            || chunks.end > chunk_count(self.size())
            || dst.len() as u64 != span.end.saturating_sub(span.start)
        {
            return Err(Error::Usage(format!(
                "chunks {chunks:?} of {}, of {} bytes, don't fit a buffer of {}",
                self.path.display(),
                self.size(),
                dst.len()
            )));
        }
        if chunks.is_empty() {
            return Ok(());
        }
        self.file.read_exact_at(span.start, dst).at(&self.path)?;
        let table = &self.model.tables[self.table];
        let algorithm = table.algorithm();
        let digests = table.chunks(self.index);
        let chunk = usize::try_from(CHUNK_SIZE).unwrap_or(usize::MAX);
        for (number, bytes) in chunks.clone().zip(dst.chunks_mut(chunk)) {
            let digest = &digests[usize::try_from(number).unwrap_or(usize::MAX)];
            if algorithm.digest(bytes).as_ref() == Some(digest) {
                continue;
            }
            self.reread(number * CHUNK_SIZE, bytes)?;
            if algorithm.digest(bytes).as_ref() != Some(digest) {
                return Err(Error::Mismatch {
                    path: self.model.table.entries()[self.index].path.clone(),
                    chunk: Some(number),
                    algorithm,
                });
            }
        }
        Ok(())
    }

    // Reads `dst` once more from `offset` on, from the storage: the page
    // cache's copy is dropped first, and the read goes around the cache. A
    // copy the system keeps is read as it is, and checked as before.
    fn reread(&self, offset: u64, dst: &mut [u8]) -> Result<()> {
        let _ = self.file.evict(offset, dst.len() as u64);
        if self.file.mode() == CacheMode::Bypass {
            return self.file.read_exact_at(offset, dst).at(&self.path);
        }
        let file = ReadFile::open(&self.path, CacheMode::Bypass).at(&self.path)?;
        // Direct reads need aligned memory and lengths, and may end at the
        // end of the file before the aligned length does.
        let mut buffer = AlignedBuf::new(dst.len().next_multiple_of(DIRECT_ALIGNMENT));
        let read = file.read_at_most(offset, &mut buffer).at(&self.path)?;
        if read < dst.len() {
            return Err(Error::Io {
                path: self.path.clone(),
                source: io::Error::from(io::ErrorKind::UnexpectedEof),
            });
        }
        dst.copy_from_slice(&buffer[..dst.len()]);
        Ok(())
    }

    /// Reads the whole file into `dst`, which must be as long as the file,
    /// verifying every chunk.
    ///
    /// # Errors
    ///
    /// As [`OpenFile::read_chunks`].
    pub fn read_all(&self, dst: &mut [u8]) -> Result<()> {
        self.read_chunks(0..chunk_count(self.size()), dst)
    }
}

// The chunks that cover `bytes`, none if it is empty.
fn chunks_covering(bytes: &Range<u64>) -> Range<u64> {
    let first = bytes.start / CHUNK_SIZE;
    if bytes.is_empty() {
        first..first
    } else {
        first..bytes.end.div_ceil(CHUNK_SIZE)
    }
}

// The bytes the chunks `chunks` span of a file of `size` bytes.
fn chunk_span(chunks: &Range<u64>, size: u64) -> Range<u64> {
    chunks.start.saturating_mul(CHUNK_SIZE).min(size)
        ..chunks.end.saturating_mul(CHUNK_SIZE).min(size)
}

#[cfg(test)]
mod tests;
