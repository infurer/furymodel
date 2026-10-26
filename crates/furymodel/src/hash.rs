// SPDX-FileCopyrightText: 2026 Mika Tammi
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The hash tables of [the specification](crate::spec)'s section 5 and the
//! content id of its section 6.

use std::fmt;

use sha2::Digest as _;

use crate::manifest::{FileTable, HASHES_DIR};

/// The chunk size of the chunked algorithms, in bytes: 2 MiB.
pub const CHUNK_SIZE: u64 = CHUNK as u64;

const CHUNK: usize = 2 * 1024 * 1024;

/// Whether this build has gxhash3-128. gxhash needs AES instructions (and
/// SSE2 or NEON) at compile time; without them the crate leaves it out and
/// verifies with xxh3-128. Where the build has it, the CPU must have them
/// too, which [`Algorithm::is_supported`] asks it at run time. This
/// condition and the `gxhash` dependency's in the crate's Cargo.toml must
/// stay the same.
pub const GXHASH: bool = cfg!(any(
    all(
        target_arch = "x86_64",
        target_feature = "aes",
        target_feature = "sse2"
    ),
    all(
        target_arch = "aarch64",
        target_feature = "aes",
        target_feature = "neon"
    ),
));

/// A hash algorithm of section 5.1. More can be added to the format later,
/// so a match on it needs an arm for those.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[non_exhaustive]
pub enum Algorithm {
    /// `gxhash3-128`: gxhash 3, 128 bits, seed 0. Chunked.
    Gxhash3,
    /// `xxh3-128`: XXH3-128, seed 0. Chunked.
    Xxh3,
    /// `sha2-512`: SHA-512.
    Sha512,
    /// `sha3-512`: SHA3-512.
    Sha3,
    /// `blake3`: BLAKE3, 256 bits.
    Blake3,
    /// `sha2-256`: SHA-256.
    Sha256,
}

impl Algorithm {
    /// Every algorithm: the fast ones first, fastest first.
    pub const ALL: [Self; 6] = [
        Self::Gxhash3,
        Self::Xxh3,
        Self::Sha512,
        Self::Sha3,
        Self::Blake3,
        Self::Sha256,
    ];

    /// The algorithm's id, which also names its table file.
    #[must_use]
    pub const fn id(self) -> &'static str {
        match self {
            Self::Gxhash3 => "gxhash3-128",
            Self::Xxh3 => "xxh3-128",
            Self::Sha512 => "sha2-512",
            Self::Sha3 => "sha3-512",
            Self::Blake3 => "blake3",
            Self::Sha256 => "sha2-256",
        }
    }

    /// The algorithm of an id.
    #[must_use]
    pub fn from_id(id: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|algorithm| algorithm.id() == id)
    }

    /// A digest's size in bytes.
    #[must_use]
    pub const fn digest_size(self) -> usize {
        match self {
            Self::Gxhash3 | Self::Xxh3 => 16,
            Self::Blake3 | Self::Sha256 => 32,
            Self::Sha512 | Self::Sha3 => 64,
        }
    }

    /// Whether the algorithm has chunk digests: the fast checks do.
    #[must_use]
    pub const fn is_chunked(self) -> bool {
        matches!(self, Self::Gxhash3 | Self::Xxh3)
    }

    /// Whether the algorithm can be computed here: all but gxhash3-128
    /// always can, and gxhash3-128 where this build has it ([`GXHASH`]) and
    /// the CPU has AES instructions, as it answers at run time.
    #[must_use]
    pub fn is_supported(self) -> bool {
        match self {
            Self::Gxhash3 => GXHASH && cpu_has_aes(),
            Self::Xxh3 | Self::Sha512 | Self::Sha3 | Self::Blake3 | Self::Sha256 => true,
        }
    }

    /// The path of the algorithm's table file.
    #[must_use]
    pub fn table_path(self) -> String {
        format!("{HASHES_DIR}/{}.bin", self.id())
    }

    /// The algorithm's digest of `bytes`, or `None` if this build doesn't
    /// support it.
    #[must_use]
    pub fn digest(self, bytes: &[u8]) -> Option<Digest> {
        self.digest_fn().map(|digest| digest(bytes))
    }

    /// The algorithm's digest function, if this build supports it.
    fn digest_fn(self) -> Option<fn(&[u8]) -> Digest> {
        Some(match self {
            Self::Gxhash3 => gxhash128()?,
            Self::Xxh3 => |bytes| Digest::from_u128(xxhash_rust::xxh3::xxh3_128(bytes)),
            Self::Sha512 => |bytes| Digest::new(&sha2::Sha512::digest(bytes)),
            Self::Sha3 => |bytes| Digest::new(&sha3::Sha3_512::digest(bytes)),
            Self::Blake3 => |bytes| Digest::new(blake3::hash(bytes).as_bytes()),
            Self::Sha256 => |bytes| Digest::new(&sha2::Sha256::digest(bytes)),
        })
    }

    /// The fastest of `present` that `supported` accepts, among the chunked
    /// algorithms: the table a reader verifies chunks with.
    #[must_use]
    pub fn fastest_chunked(
        present: impl IntoIterator<Item = Self>,
        supported: impl Fn(Self) -> bool,
    ) -> Option<Self> {
        present
            .into_iter()
            .filter(|&algorithm| algorithm.is_chunked() && supported(algorithm))
            .min()
    }
}

impl fmt::Display for Algorithm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.id())
    }
}

#[cfg(any(
    all(
        target_arch = "x86_64",
        target_feature = "aes",
        target_feature = "sse2"
    ),
    all(
        target_arch = "aarch64",
        target_feature = "aes",
        target_feature = "neon"
    ),
))]
fn gxhash128() -> Option<fn(&[u8]) -> Digest> {
    let digest: fn(&[u8]) -> Digest = |bytes| Digest::from_u128(gxhash::gxhash128(bytes, 0));
    cpu_has_aes().then_some(digest)
}

#[cfg(not(any(
    all(
        target_arch = "x86_64",
        target_feature = "aes",
        target_feature = "sse2"
    ),
    all(
        target_arch = "aarch64",
        target_feature = "aes",
        target_feature = "neon"
    ),
)))]
fn gxhash128() -> Option<fn(&[u8]) -> Digest> {
    None
}

// Whether the CPU has the AES instructions, and SSE2 or NEON, that gxhash
// needs, as it answers at run time. std's feature detection answers from
// the target features where the build has them, as one with gxhash does:
// asked that way, a binary built with them would use them on a CPU
// without them. Asked once: the answer is the machine's, and CPUID
// serializes, a virtual machine's hypervisor taking every one.
fn cpu_has_aes() -> bool {
    static ANSWER: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ANSWER.get_or_init(cpu_answers_aes)
}

#[cfg(target_arch = "x86_64")]
fn cpu_answers_aes() -> bool {
    // CPUID's leaf 1: AES-NI is ECX's bit 25. Every x86-64 CPU has SSE2.
    core::arch::x86_64::__cpuid(1).ecx & (1 << 25) != 0
}

#[cfg(all(target_arch = "aarch64", target_os = "linux"))]
fn cpu_answers_aes() -> bool {
    // The kernel's hwcaps: HWCAP_ASIMD (NEON) is bit 1, HWCAP_AES bit 3.
    let (hwcap, _) = rustix::param::linux_hwcap();
    hwcap & 0b1010 == 0b1010
}

#[cfg(all(target_arch = "aarch64", target_vendor = "apple"))]
fn cpu_answers_aes() -> bool {
    // Every CPU Apple's aarch64 targets run on has them.
    true
}

#[cfg(not(any(
    target_arch = "x86_64",
    all(
        target_arch = "aarch64",
        any(target_os = "linux", target_vendor = "apple")
    ),
)))]
fn cpu_answers_aes() -> bool {
    false
}

/// A digest of up to 64 bytes.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Digest {
    bytes: [u8; 64],
    len: u8,
}

impl Digest {
    /// A digest of the given bytes, at most 64 of them.
    ///
    /// # Panics
    ///
    /// If `bytes` is longer than 64 bytes.
    #[must_use]
    pub fn new(bytes: &[u8]) -> Self {
        let mut digest = Self {
            bytes: [0; 64],
            len: u8::try_from(bytes.len()).expect("a digest of at most 64 bytes"),
        };
        digest.bytes[..bytes.len()].copy_from_slice(bytes);
        digest
    }

    /// A 128-bit digest, stored big-endian (section 5.1).
    #[must_use]
    pub fn from_u128(value: u128) -> Self {
        Self::new(&value.to_be_bytes())
    }

    /// The digest's bytes.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes[..usize::from(self.len)]
    }

    /// The digest in lowercase hex.
    #[must_use]
    pub fn to_hex(&self) -> String {
        hex(self.as_bytes())
    }
}

impl fmt::Debug for Digest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_hex())
    }
}

impl fmt::Display for Digest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_hex())
    }
}

fn hex(bytes: &[u8]) -> String {
    use fmt::Write as _;
    bytes.iter().fold(String::new(), |mut hex, byte| {
        write!(hex, "{byte:02x}").expect("writing to a String");
        hex
    })
}

/// The number of chunks of a file of `size` bytes.
#[must_use]
pub const fn chunk_count(size: u64) -> u64 {
    size.div_ceil(CHUNK_SIZE)
}

/// A file's digests by one algorithm.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileDigests {
    /// The algorithm.
    pub algorithm: Algorithm,
    /// The file's digest.
    pub file: Digest,
    /// The chunk digests, empty for algorithms that aren't chunked.
    pub chunks: Vec<Digest>,
}

/// Computes a file's digests by several algorithms in one pass over its
/// bytes, which it takes in pieces of any size.
pub struct FileHasher {
    chunked: Vec<Chunked>,
    pending: Vec<u8>,
    streams: Vec<Stream>,
    size: u64,
}

struct Chunked {
    algorithm: Algorithm,
    digest: fn(&[u8]) -> Digest,
    chunks: Vec<Digest>,
}

enum Stream {
    Sha512(sha2::Sha512),
    Sha3(sha3::Sha3_512),
    Blake3(Box<blake3::Hasher>),
    Sha256(sha2::Sha256),
}

impl FileHasher {
    /// A hasher for the given algorithms.
    ///
    /// # Errors
    ///
    /// An algorithm this build doesn't support.
    pub fn new(algorithms: &[Algorithm]) -> Result<Self, Algorithm> {
        let mut hasher = Self {
            chunked: Vec::new(),
            pending: Vec::new(),
            streams: Vec::new(),
            size: 0,
        };
        for &algorithm in algorithms {
            let digest = algorithm.digest_fn().ok_or(algorithm)?;
            match algorithm {
                Algorithm::Gxhash3 | Algorithm::Xxh3 => hasher.chunked.push(Chunked {
                    algorithm,
                    digest,
                    chunks: Vec::new(),
                }),
                Algorithm::Sha512 => hasher.streams.push(Stream::Sha512(sha2::Sha512::new())),
                Algorithm::Sha3 => hasher.streams.push(Stream::Sha3(sha3::Sha3_512::new())),
                Algorithm::Blake3 => {
                    hasher
                        .streams
                        .push(Stream::Blake3(Box::new(blake3::Hasher::new())));
                }
                Algorithm::Sha256 => hasher.streams.push(Stream::Sha256(sha2::Sha256::new())),
            }
        }
        Ok(hasher)
    }

    /// Feeds the file's next bytes.
    pub fn update(&mut self, mut bytes: &[u8]) {
        self.size += bytes.len() as u64;
        for stream in &mut self.streams {
            match stream {
                Stream::Sha512(hasher) => hasher.update(bytes),
                Stream::Sha3(hasher) => hasher.update(bytes),
                Stream::Blake3(hasher) => {
                    hasher.update(bytes);
                }
                Stream::Sha256(hasher) => hasher.update(bytes),
            }
        }
        if self.chunked.is_empty() {
            return;
        }
        while !bytes.is_empty() {
            if self.pending.is_empty() && bytes.len() >= CHUNK {
                // A whole chunk straight from the caller's bytes.
                let (whole, rest) = bytes.split_at(CHUNK);
                self.chunk_done(whole);
                bytes = rest;
            } else {
                let take = (CHUNK - self.pending.len()).min(bytes.len());
                self.pending.extend_from_slice(&bytes[..take]);
                bytes = &bytes[take..];
                if self.pending.len() == CHUNK {
                    let pending = std::mem::take(&mut self.pending);
                    self.chunk_done(&pending);
                    self.pending = pending;
                    self.pending.clear();
                }
            }
        }
    }

    fn chunk_done(&mut self, chunk: &[u8]) {
        for chunked in &mut self.chunked {
            chunked.chunks.push((chunked.digest)(chunk));
        }
    }

    /// The number of bytes fed so far.
    #[must_use]
    pub fn size(&self) -> u64 {
        self.size
    }

    /// The file's digests, in the order the algorithms were given in, apart
    /// from the chunked ones, which come first.
    #[must_use]
    pub fn finish(mut self) -> Vec<FileDigests> {
        if !self.pending.is_empty() {
            let pending = std::mem::take(&mut self.pending);
            self.chunk_done(&pending);
        }
        let mut digests = Vec::new();
        for chunked in self.chunked {
            let concatenated: Vec<u8> = chunked
                .chunks
                .iter()
                .flat_map(|digest| digest.as_bytes().iter().copied())
                .collect();
            digests.push(FileDigests {
                algorithm: chunked.algorithm,
                file: (chunked.digest)(&concatenated),
                chunks: chunked.chunks,
            });
        }
        for stream in self.streams {
            let (algorithm, file) = match stream {
                Stream::Sha512(hasher) => (Algorithm::Sha512, Digest::new(&hasher.finalize())),
                Stream::Sha3(hasher) => (Algorithm::Sha3, Digest::new(&hasher.finalize())),
                Stream::Blake3(hasher) => {
                    (Algorithm::Blake3, Digest::new(hasher.finalize().as_bytes()))
                }
                Stream::Sha256(hasher) => (Algorithm::Sha256, Digest::new(&hasher.finalize())),
            };
            digests.push(FileDigests {
                algorithm,
                file,
                chunks: Vec::new(),
            });
        }
        digests
    }
}

/// What's wrong with a hash table.
#[derive(Debug, thiserror::Error)]
#[error("hash table {algorithm}: {message}")]
pub struct TableError {
    /// The table's algorithm.
    pub algorithm: Algorithm,
    /// What's wrong.
    pub message: String,
    /// The table format version of a table whose checksum holds, if it is
    /// one newer than this build reads: such a table is left for a build
    /// that reads it.
    pub newer_version: Option<u32>,
}

const MAGIC: &[u8; 8] = b"FMHASHES";
const TABLE_VERSION: u32 = 1;
const HEADER_SIZE: usize = 128;
const TRAILER_SIZE: usize = 16;

/// The hash table of one algorithm (section 5.2): a digest of every file of
/// the file table and, for the chunked algorithms, of every chunk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HashTable {
    algorithm: Algorithm,
    fingerprint: [u8; 32],
    files: Vec<Digest>,
    chunks: Vec<Digest>,
    // Where each file's chunk digests start in `chunks`, and the end.
    chunk_starts: Vec<usize>,
}

impl HashTable {
    /// A table of `algorithm` for `table`, from every file's digests in
    /// file-table order.
    ///
    /// # Errors
    ///
    /// If the digests aren't of `algorithm`, or don't fit the files' count or
    /// sizes.
    pub fn new(
        algorithm: Algorithm,
        table: &FileTable,
        digests: Vec<FileDigests>,
    ) -> Result<Self, TableError> {
        let error = |message: String| TableError {
            algorithm,
            message,
            newer_version: None,
        };
        if digests.len() != table.len() {
            return Err(error(format!(
                "{} file digests for a table of {} files",
                digests.len(),
                table.len()
            )));
        }
        let mut files = Vec::with_capacity(digests.len());
        let mut chunks = Vec::new();
        let mut chunk_starts = vec![0];
        for (entry, file) in table.entries().iter().zip(digests) {
            let expected = if algorithm.is_chunked() {
                chunk_count(entry.size)
            } else {
                0
            };
            if file.algorithm != algorithm || file.chunks.len() as u64 != expected {
                return Err(error(format!(
                    "the digests of {} are of {} with {} chunks, not of {algorithm} with \
                     {expected}",
                    entry.path,
                    file.algorithm,
                    file.chunks.len()
                )));
            }
            files.push(file.file);
            chunks.extend(file.chunks);
            chunk_starts.push(chunks.len());
        }
        Ok(Self {
            algorithm,
            fingerprint: fingerprint(table),
            files,
            chunks,
            chunk_starts,
        })
    }

    /// The table's algorithm.
    #[must_use]
    pub fn algorithm(&self) -> Algorithm {
        self.algorithm
    }

    /// The digest of the file of index `file`.
    #[must_use]
    pub fn file(&self, file: usize) -> &Digest {
        &self.files[file]
    }

    /// The chunk digests of the file of index `file`, empty for algorithms
    /// that aren't chunked.
    #[must_use]
    pub fn chunks(&self, file: usize) -> &[Digest] {
        &self.chunks[self.chunk_starts[file]..self.chunk_starts[file + 1]]
    }

    /// The table as its file holds it.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        let size = self.algorithm.digest_size();
        let mut bytes = Vec::with_capacity(
            HEADER_SIZE + (self.files.len() + self.chunks.len()) * size + TRAILER_SIZE,
        );
        bytes.extend_from_slice(MAGIC);
        bytes.extend_from_slice(&TABLE_VERSION.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
        let mut id = [0u8; 32];
        id[..self.algorithm.id().len()].copy_from_slice(self.algorithm.id().as_bytes());
        bytes.extend_from_slice(&id);
        bytes.extend_from_slice(&u32::try_from(size).unwrap_or(u32::MAX).to_le_bytes());
        bytes.extend_from_slice(&chunk_size_field(self.algorithm).to_le_bytes());
        bytes.extend_from_slice(&(self.files.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&(self.chunks.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&self.fingerprint);
        bytes.extend_from_slice(&[0; 24]);
        for digest in self.files.iter().chain(&self.chunks) {
            bytes.extend_from_slice(digest.as_bytes());
        }
        let checksum = xxhash_rust::xxh3::xxh3_128(&bytes);
        bytes.extend_from_slice(&checksum.to_be_bytes());
        bytes
    }

    /// Reads the table of `algorithm` that a model of file table `table`
    /// holds, checking everything section 5.2 lists.
    ///
    /// # Errors
    ///
    /// If any of those checks fails.
    pub fn decode(
        bytes: &[u8],
        algorithm: Algorithm,
        table: &FileTable,
    ) -> Result<Self, TableError> {
        let error = |message: &str| TableError {
            algorithm,
            message: message.to_owned(),
            newer_version: None,
        };
        if bytes.len() < HEADER_SIZE + TRAILER_SIZE {
            return Err(error("the file is too short for a hash table"));
        }
        let (body, trailer) = bytes.split_at(bytes.len() - TRAILER_SIZE);
        if xxhash_rust::xxh3::xxh3_128(body).to_be_bytes() != trailer {
            return Err(error("its checksum doesn't match: the file is corrupt"));
        }
        let u32_at = |offset: usize| u32::from_le_bytes(std::array::from_fn(|i| body[offset + i]));
        let u64_at = |offset: usize| u64::from_le_bytes(std::array::from_fn(|i| body[offset + i]));
        let mut id = [0u8; 32];
        id[..algorithm.id().len()].copy_from_slice(algorithm.id().as_bytes());
        if &body[..8] == MAGIC && u32_at(8) > TABLE_VERSION {
            return Err(TableError {
                algorithm,
                message: format!(
                    "it is a table of version {}, newer than this build reads",
                    u32_at(8)
                ),
                newer_version: Some(u32_at(8)),
            });
        }
        if &body[..8] != MAGIC || u32_at(8) != TABLE_VERSION || u32_at(12) != 0 {
            return Err(error("it isn't a hash table of version 1"));
        }
        if body[16..48] != id {
            return Err(error("it is the table of another algorithm"));
        }
        let size = algorithm.digest_size();
        let expected_chunks: u64 = if algorithm.is_chunked() {
            table
                .entries()
                .iter()
                .try_fold(0u64, |chunks, entry| {
                    chunks.checked_add(chunk_count(entry.size))
                })
                .ok_or_else(|| error("its counts overflow"))?
        } else {
            0
        };
        if u32_at(48) as usize != size
            || u32_at(52) != chunk_size_field(algorithm)
            || u64_at(56) != table.len() as u64
            || u64_at(64) != expected_chunks
        {
            return Err(error(
                "its digest size, chunk size or counts don't fit the model's files",
            ));
        }
        if body[72..104] != fingerprint(table) {
            return Err(error(
                "it was computed for other files or sizes than the model's",
            ));
        }
        if body[104..HEADER_SIZE].iter().any(|&byte| byte != 0) {
            return Err(error("its reserved bytes aren't zero"));
        }
        let digests = usize::try_from(expected_chunks)
            .ok()
            .and_then(|chunks| chunks.checked_add(table.len()))
            .and_then(|count| count.checked_mul(size))
            .ok_or_else(|| error("its counts overflow"))?;
        if body.len() != HEADER_SIZE + digests {
            return Err(error("its length doesn't fit its counts"));
        }
        let mut all = body[HEADER_SIZE..].chunks_exact(size).map(Digest::new);
        let files: Vec<Digest> = all.by_ref().take(table.len()).collect();
        let chunks: Vec<Digest> = all.collect();
        let mut chunk_starts = vec![0];
        let mut start = 0;
        for entry in table.entries() {
            if algorithm.is_chunked() {
                start += usize::try_from(chunk_count(entry.size))
                    .map_err(|_| error("its counts overflow"))?;
            }
            chunk_starts.push(start);
        }
        Ok(Self {
            algorithm,
            fingerprint: fingerprint(table),
            files,
            chunks,
            chunk_starts,
        })
    }
}

fn chunk_size_field(algorithm: Algorithm) -> u32 {
    if algorithm.is_chunked() {
        u32::try_from(CHUNK_SIZE).expect("2 MiB fits a u32")
    } else {
        0
    }
}

/// The fingerprint of a file table (section 5.3), which ties a hash table to
/// the files and sizes it was computed for.
#[must_use]
pub fn fingerprint(table: &FileTable) -> [u8; 32] {
    let mut hasher = sha2::Sha512::new();
    hasher.update(b"furymodel file table v1\n");
    for entry in table.entries() {
        hasher.update(path_length(&entry.path));
        hasher.update(entry.path.as_bytes());
        hasher.update(entry.size.to_le_bytes());
    }
    let digest = hasher.finalize();
    std::array::from_fn(|i| digest[i])
}

fn path_length(path: &str) -> [u8; 4] {
    u32::try_from(path.len())
        .expect("validated paths are short")
        .to_le_bytes()
}

/// A model's content id (section 6): the SHA-512 over its files' paths,
/// sizes and sha2-512 digests.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct ContentId([u8; 64]);

impl ContentId {
    /// The content id of the files of `table`, from their sha2-512 table.
    ///
    /// # Errors
    ///
    /// If `sha512` is the table of another algorithm, or of other files or
    /// sizes than `table`'s.
    pub fn compute(table: &FileTable, sha512: &HashTable) -> Result<Self, TableError> {
        if sha512.algorithm != Algorithm::Sha512
            || sha512.files.len() != table.len()
            || sha512.fingerprint != fingerprint(table)
        {
            return Err(TableError {
                algorithm: sha512.algorithm,
                message: "the content id needs the sha2-512 table of these files".to_owned(),
                newer_version: None,
            });
        }
        let mut hasher = sha2::Sha512::new();
        hasher.update(b"furymodel content id v1\n");
        for (entry, digest) in table.entries().iter().zip(&sha512.files) {
            hasher.update(path_length(&entry.path));
            hasher.update(entry.path.as_bytes());
            hasher.update(entry.size.to_le_bytes());
            hasher.update(digest.as_bytes());
        }
        let digest = hasher.finalize();
        Ok(Self(std::array::from_fn(|i| digest[i])))
    }

    /// The content id of its hex form: 128 lowercase hex digits, nothing
    /// else.
    #[must_use]
    pub fn from_hex(hex: &str) -> Option<Self> {
        fn nibble(digit: u8) -> Option<u8> {
            match digit {
                b'0'..=b'9' => Some(digit - b'0'),
                b'a'..=b'f' => Some(digit - b'a' + 10),
                _ => None,
            }
        }
        if hex.len() != 128 {
            return None;
        }
        let mut bytes = [0u8; 64];
        let (pairs, _) = hex.as_bytes().as_chunks::<2>();
        for (byte, &[high, low]) in bytes.iter_mut().zip(pairs) {
            *byte = (nibble(high)? << 4) | nibble(low)?;
        }
        Some(Self(bytes))
    }

    /// The content id's bytes.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8; 64] {
        &self.0
    }

    /// The first 16 hex digits, as logs show the content id.
    #[must_use]
    pub fn short(&self) -> String {
        hex(&self.0[..8])
    }
}

impl fmt::Debug for ContentId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&hex(&self.0))
    }
}

impl fmt::Display for ContentId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&hex(&self.0))
    }
}

#[cfg(test)]
mod tests;
