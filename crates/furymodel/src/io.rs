// SPDX-FileCopyrightText: 2026 Mika Tammi
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Reads at offsets, through the page cache or around it.
//!
//! Around it, a read leaves no copy of the bytes in RAM, so that a memory
//! budget means what it says: `F_NOCACHE` on macOS, `O_DIRECT` on Linux. On
//! Linux such a read comes from the storage; on macOS it still returns the
//! pages the cache holds, and only the others come from the storage. Direct
//! I/O on Linux needs the buffer, the offset and the length aligned to the
//! device's block size; payload files are aligned to 16 KiB, and
//! [`AlignedBuf`] gives buffers aligned to [`DIRECT_ALIGNMENT`]. Reads
//! around the cache ask for that alignment on every system and file system,
//! those that can't go around it too, so that code that runs on one runs on
//! the others.

use std::fs::File;
use std::io;
use std::ops::{Deref, DerefMut};
use std::os::unix::fs::FileExt as _;
use std::path::Path;

/// The alignment direct reads need of buffers, offsets and lengths.
pub const DIRECT_ALIGNMENT: usize = 4096;

/// Whether reads go through the page cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CacheMode {
    /// Around the page cache: a read adds no copy of the bytes to it. On
    /// macOS it still returns the pages the cache holds; the second read of
    /// a chunk that didn't match, and a full verification, drop them from
    /// the cache first.
    Bypass,
    /// Through the page cache.
    Use,
}

/// A file opened for reads at offsets.
#[derive(Debug)]
pub struct ReadFile {
    file: File,
    // How reads go in fact, and how they were asked to go: the alignment
    // they need follows the latter.
    mode: CacheMode,
    opened: CacheMode,
}

impl ReadFile {
    /// Opens `path` for reads in `mode`. If the file system can't bypass the
    /// page cache (procfs, or tmpfs before Linux 6.6, has no `O_DIRECT`),
    /// the file is opened to use it instead, as [`ReadFile::mode`] then
    /// says; its reads still need the alignment of reads around the cache.
    ///
    /// # Errors
    ///
    /// If the file can't be opened.
    pub fn open(path: &Path, mode: CacheMode) -> io::Result<Self> {
        let (file, effective) = match mode {
            CacheMode::Use => (File::open(path)?, CacheMode::Use),
            CacheMode::Bypass => open_bypassing(path)?,
        };
        Ok(Self {
            file,
            mode: effective,
            opened: mode,
        })
    }

    /// How the file's reads go in fact.
    #[must_use]
    pub fn mode(&self) -> CacheMode {
        self.mode
    }

    /// The file's size in bytes.
    ///
    /// # Errors
    ///
    /// If the file's metadata can't be read.
    pub fn size(&self) -> io::Result<u64> {
        Ok(self.file.metadata()?.len())
    }

    /// Fills `dst` with the file's bytes from `offset` on.
    ///
    /// # Errors
    ///
    /// If the read fails or reaches the end of the file first, or if the
    /// file was opened in [`CacheMode::Bypass`], whatever [`ReadFile::mode`]
    /// says, and `dst`'s address, `offset` or the length isn't aligned to
    /// [`DIRECT_ALIGNMENT`].
    pub fn read_exact_at(&self, offset: u64, dst: &mut [u8]) -> io::Result<()> {
        self.check_alignment(offset, dst)?;
        self.file.read_exact_at(dst, offset)
    }

    /// Reads the file's bytes from `offset` on into `dst` until it is full
    /// or the file ends, and returns how many it read.
    ///
    /// # Errors
    ///
    /// As [`ReadFile::read_exact_at`], but for the end of the file.
    pub(crate) fn read_at_most(&self, offset: u64, dst: &mut [u8]) -> io::Result<usize> {
        self.check_alignment(offset, dst)?;
        let mut read = 0;
        while read < dst.len() {
            match self.file.read_at(&mut dst[read..], offset + read as u64) {
                Ok(0) => break,
                Ok(n) => read += n,
                Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
                Err(error) => return Err(error),
            }
        }
        Ok(read)
    }

    /// Drops the page cache's copy of the file's bytes from `offset` on for
    /// `len` bytes, as far as the system lets a process: on macOS by
    /// invalidating a mapping of them, which leaves pages wired in memory;
    /// on Linux with `posix_fadvise`'s `DONTNEED`, which leaves pages that
    /// are mapped or dirty. Elsewhere it does nothing.
    pub(crate) fn evict(&self, offset: u64, len: u64) -> io::Result<()> {
        if len == 0 {
            return Ok(());
        }
        evict(&self.file, offset, len)
    }

    fn check_alignment(&self, offset: u64, dst: &[u8]) -> io::Result<()> {
        let aligned = dst.as_ptr().addr().is_multiple_of(DIRECT_ALIGNMENT)
            && offset.is_multiple_of(DIRECT_ALIGNMENT as u64)
            && dst.len().is_multiple_of(DIRECT_ALIGNMENT);
        if self.opened == CacheMode::Bypass && !aligned {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "a file opened to read around the page cache needs its buffer, offset and length \
                 aligned to 4096 bytes",
            ));
        }
        Ok(())
    }
}

// The file opened to read around the page cache, and how its reads go.
#[cfg(target_os = "macos")]
fn open_bypassing(path: &Path) -> io::Result<(File, CacheMode)> {
    let file = File::open(path)?;
    rustix::fs::fcntl_nocache(&file, true)?;
    Ok((file, CacheMode::Bypass))
}

#[cfg(target_os = "linux")]
fn open_bypassing(path: &Path) -> io::Result<(File, CacheMode)> {
    use rustix::fs::{Mode, OFlags};
    let flags = OFlags::RDONLY | OFlags::CLOEXEC;
    match rustix::fs::open(path, flags | OFlags::DIRECT, Mode::empty()) {
        Ok(fd) => Ok((File::from(fd), CacheMode::Bypass)),
        Err(rustix::io::Errno::INVAL) => Ok((
            File::from(rustix::fs::open(path, flags, Mode::empty())?),
            CacheMode::Use,
        )),
        Err(errno) => Err(errno.into()),
    }
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn open_bypassing(path: &Path) -> io::Result<(File, CacheMode)> {
    Ok((File::open(path)?, CacheMode::Use))
}

#[cfg(target_os = "macos")]
#[expect(unsafe_code, reason = "msync drops pages through a mapping of them")]
fn evict(file: &File, offset: u64, len: u64) -> io::Result<()> {
    use rustix::mm::{MapFlags, MsyncFlags, ProtFlags, mmap, msync, munmap};
    // A mapping starts at a page.
    let start = offset - offset % rustix::param::page_size() as u64;
    let len = offset
        .checked_add(len)
        .and_then(|end| usize::try_from(end - start).ok())
        .ok_or_else(|| io::Error::from(io::ErrorKind::InvalidInput))?;
    // SAFETY: a new mapping, read-only and shared, that nothing reads
    // through and that is unmapped below.
    let map = unsafe {
        mmap(
            std::ptr::null_mut(),
            len,
            ProtFlags::READ,
            MapFlags::SHARED,
            file,
            start,
        )
    }?;
    // SAFETY: `map` is the mapping above, `len` bytes long.
    let invalidated = unsafe { msync(map, len, MsyncFlags::INVALIDATE) };
    // SAFETY: as above; nothing refers to the mapping after this.
    unsafe { munmap(map, len) }?;
    invalidated.map_err(Into::into)
}

#[cfg(target_os = "linux")]
fn evict(file: &File, offset: u64, len: u64) -> io::Result<()> {
    use rustix::fs::{Advice, fadvise};
    fadvise(
        file,
        offset,
        std::num::NonZeroU64::new(len),
        Advice::DontNeed,
    )?;
    Ok(())
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn evict(_file: &File, _offset: u64, _len: u64) -> io::Result<()> {
    Ok(())
}

/// A zeroed buffer whose start is aligned to [`DIRECT_ALIGNMENT`], for
/// reads around the page cache.
#[derive(Debug)]
pub struct AlignedBuf {
    storage: Vec<u8>,
    start: usize,
    len: usize,
}

impl AlignedBuf {
    /// A buffer of `len` zero bytes.
    #[must_use]
    pub fn new(len: usize) -> Self {
        let storage = vec![0; len + DIRECT_ALIGNMENT];
        let misalignment = storage.as_ptr().addr() % DIRECT_ALIGNMENT;
        let start = (DIRECT_ALIGNMENT - misalignment) % DIRECT_ALIGNMENT;
        Self {
            storage,
            start,
            len,
        }
    }
}

// A clone is a buffer aligned in its own right, not a copy of the storage
// with the original's start.
impl Clone for AlignedBuf {
    fn clone(&self) -> Self {
        let mut clone = Self::new(self.len);
        clone.copy_from_slice(self);
        clone
    }
}

impl Deref for AlignedBuf {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        &self.storage[self.start..self.start + self.len]
    }
}

impl DerefMut for AlignedBuf {
    fn deref_mut(&mut self) -> &mut [u8] {
        &mut self.storage[self.start..self.start + self.len]
    }
}

#[cfg(test)]
mod tests {
    use std::io;
    #[cfg(target_os = "linux")]
    use std::path::Path;

    use super::{AlignedBuf, CacheMode, DIRECT_ALIGNMENT, ReadFile};

    #[test]
    fn aligned_buffers() {
        for len in [0, 1, 4096, 2 * 1024 * 1024 + 3] {
            let buffer = AlignedBuf::new(len);
            assert_eq!(buffer.len(), len);
            assert!(buffer.as_ptr().addr().is_multiple_of(DIRECT_ALIGNMENT));
            assert!(buffer.iter().all(|&byte| byte == 0));
        }
    }

    // A clone is aligned in its own right, whatever its storage's address.
    #[test]
    fn clones_are_aligned() {
        let mut original = AlignedBuf::new(10_000);
        original[1234] = 7;
        let clones: Vec<AlignedBuf> = (0..200).map(|_| original.clone()).collect();
        for clone in &clones {
            assert!(clone.as_ptr().addr().is_multiple_of(DIRECT_ALIGNMENT));
            assert_eq!(&clone[..], &original[..]);
        }
    }

    #[test]
    fn reads_at_offsets_in_both_modes() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("data.bin");
        let contents: Vec<u8> = (0..65536u32)
            .map(|i| u8::try_from(i % 249).unwrap())
            .collect();
        std::fs::write(&path, &contents).unwrap();
        for mode in [CacheMode::Use, CacheMode::Bypass] {
            let file = ReadFile::open(&path, mode).unwrap();
            assert_eq!(file.size().unwrap(), 65536);
            let mut buffer = AlignedBuf::new(16384);
            file.read_exact_at(32768, &mut buffer).unwrap();
            assert_eq!(&buffer[..], &contents[32768..49152]);
            // Past the end.
            assert!(file.read_exact_at(57344, &mut buffer).is_err());
        }
    }

    // On every system and file system, whatever the file's reads do in
    // fact, so that code that runs on one runs on the others.
    #[test]
    fn direct_reads_need_alignment() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("data.bin");
        std::fs::write(&path, vec![1u8; 65536]).unwrap();
        let file = ReadFile::open(&path, CacheMode::Bypass).unwrap();
        let mut buffer = AlignedBuf::new(8192);
        // A length, an offset, an address that isn't aligned.
        for (offset, range) in [(4096, 0..100), (100, 0..4096), (4096, 1..4097)] {
            let error = file
                .read_exact_at(offset, &mut buffer[range.clone()])
                .unwrap_err();
            assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
            let error = file.read_at_most(offset, &mut buffer[range]).unwrap_err();
            assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
        }
        file.read_exact_at(4096, &mut buffer).unwrap();
        // Through the cache, any.
        let file = ReadFile::open(&path, CacheMode::Use).unwrap();
        file.read_exact_at(100, &mut buffer[1..4097]).unwrap();
    }

    // procfs has no O_DIRECT on any kernel, nor tmpfs before Linux 6.6:
    // their files are read through the cache, as mode() says, and still
    // need the alignment of reads around it.
    #[cfg(target_os = "linux")]
    #[test]
    fn keeps_the_alignment_where_reads_go_through_the_cache() {
        let file = ReadFile::open(Path::new("/proc/self/status"), CacheMode::Bypass).unwrap();
        assert_eq!(file.mode(), CacheMode::Use);
        let mut buffer = AlignedBuf::new(DIRECT_ALIGNMENT);
        let error = file.read_exact_at(0, &mut buffer[..100]).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
        assert!(file.read_at_most(0, &mut buffer).unwrap() > 0);
    }

    #[test]
    fn evicting_leaves_the_bytes_as_they_are() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("data.bin");
        let contents: Vec<u8> = (0..70_000u32)
            .map(|i| u8::try_from(i % 251).unwrap())
            .collect();
        std::fs::write(&path, &contents).unwrap();
        for mode in [CacheMode::Use, CacheMode::Bypass] {
            let file = ReadFile::open(&path, mode).unwrap();
            // The whole file, from within a page to the end, the last
            // bytes, nothing.
            for (offset, len) in [(0, 70_000), (20_000, 50_000), (65_536, 4_464), (100, 0)] {
                file.evict(offset, len).unwrap();
            }
            let mut buffer = AlignedBuf::new(18 * DIRECT_ALIGNMENT);
            assert_eq!(file.read_at_most(0, &mut buffer).unwrap(), 70_000);
            assert_eq!(&buffer[..70_000], &contents[..]);
        }
    }

    // On macOS a read around the cache still returns the pages it holds:
    // evicting a range drops them, and only them, and a read around the
    // cache doesn't bring them back.
    #[cfg(target_os = "macos")]
    #[test]
    #[expect(unsafe_code, reason = "mincore needs a mapping of the file to look at")]
    fn evicts_the_pages_the_cache_holds() {
        use rustix::mm::{MapFlags, ProtFlags, mmap, munmap};
        use std::ffi::{c_char, c_int, c_void};
        unsafe extern "C" {
            fn mincore(addr: *const c_void, len: usize, vec: *mut c_char) -> c_int;
        }
        let page = rustix::param::page_size();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("data.bin");
        std::fs::write(&path, vec![7; 4 * page]).unwrap();
        let file = ReadFile::open(&path, CacheMode::Use).unwrap();
        let resident = || {
            // SAFETY: a new mapping, read-only and shared, that nothing
            // reads through and that is unmapped below.
            let map = unsafe {
                mmap(
                    std::ptr::null_mut(),
                    4 * page,
                    ProtFlags::READ,
                    MapFlags::SHARED,
                    &file.file,
                    0,
                )
            }
            .unwrap();
            let mut pages: [c_char; 4] = [0; 4];
            // SAFETY: the mapping is four pages long, and `pages` has a
            // byte for each.
            let result = unsafe { mincore(map, 4 * page, pages.as_mut_ptr()) };
            // SAFETY: as above; nothing refers to the mapping after this.
            unsafe { munmap(map, 4 * page) }.unwrap();
            assert_eq!(result, 0);
            // MINCORE_INCORE.
            pages.map(|state| state & 1 != 0)
        };
        let mut buffer = vec![0; 4 * page];
        file.read_exact_at(0, &mut buffer).unwrap();
        assert_eq!(resident(), [true; 4]);
        file.evict(page as u64, 2 * page as u64).unwrap();
        assert_eq!(resident(), [true, false, false, true]);
        let bypassing = ReadFile::open(&path, CacheMode::Bypass).unwrap();
        let mut aligned = AlignedBuf::new(4 * page);
        bypassing.read_exact_at(0, &mut aligned).unwrap();
        assert!(aligned.iter().all(|&byte| byte == 7));
        assert_eq!(resident(), [true, false, false, true]);
    }
}
