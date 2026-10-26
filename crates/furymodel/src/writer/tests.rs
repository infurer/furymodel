// SPDX-FileCopyrightText: 2026 Mika Tammi
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use super::{GroupSpec, TensorData, TensorSpec, Writer, write_new};
use crate::error::Error;
use crate::hash::{Algorithm, ContentId, FileHasher, HashTable};
use crate::manifest::{ALIGNMENT, Axis, EntryKind, MANIFEST_PATH, Manifest};
use crate::registry::{Dtype, Layout, Scheme};
use crate::testing::{BIG, algorithms, bytes, info, write_tiny};

#[test]
fn writes_a_model_whose_tables_match_its_files() {
    let dir = tempfile::tempdir().unwrap();
    let written = write_tiny(dir.path(), false);

    let json = fs::read(dir.path().join(MANIFEST_PATH)).unwrap();
    let (manifest, table) = Manifest::parse(&json).unwrap();
    assert_eq!(manifest, written.manifest);
    assert_eq!(table, written.table);
    // The manifest, two files of its own, six expert files, the aux file.
    assert_eq!(table.len(), 1 + 2 + 6 + 1);

    for algorithm in algorithms() {
        let encoded = fs::read(dir.path().join(algorithm.table_path())).unwrap();
        let hash_table = HashTable::decode(&encoded, algorithm, &table).unwrap();
        for (index, entry) in table.entries().iter().enumerate() {
            let contents = fs::read(dir.path().join(&entry.path)).unwrap();
            assert_eq!(contents.len() as u64, entry.size, "{}", entry.path);
            let mut hasher = FileHasher::new(&[algorithm]).unwrap();
            hasher.update(&contents);
            let digests = hasher.finish().remove(0);
            assert_eq!(hash_table.file(index), &digests.file, "{}", entry.path);
            assert_eq!(hash_table.chunks(index), digests.chunks.as_slice());
        }
        if algorithm == Algorithm::Sha512 {
            assert_eq!(
                Some(ContentId::compute(&table, &hash_table).unwrap()),
                written.content_id
            );
        }
    }
    let tables = fs::read_dir(dir.path().join("hashes")).unwrap().count();
    assert_eq!(tables, algorithms().len());
}

#[test]
fn places_tensors_at_aligned_offsets_with_zeros_between() {
    let dir = tempfile::tempdir().unwrap();
    let written = write_tiny(dir.path(), false);
    for (file, tensor) in written.table.tensors() {
        let entry = &written.table.entries()[file];
        let contents = fs::read(dir.path().join(&entry.path)).unwrap();
        let start = usize::try_from(tensor.offset).unwrap();
        let end = start + usize::try_from(tensor.size).unwrap();
        assert_eq!(contents[start..end], bytes(&tensor.name, tensor.size));
        assert!(tensor.offset.is_multiple_of(ALIGNMENT));
    }
    let big = written.table.index_of("global/big.bin").unwrap();
    let entry = &written.table.entries()[big];
    assert_eq!(entry.size, BIG.next_multiple_of(ALIGNMENT));
    let contents = fs::read(dir.path().join(&entry.path)).unwrap();
    let tail = usize::try_from(BIG).unwrap();
    assert!(contents[tail..].iter().all(|&byte| byte == 0));
    match &entry.kind {
        EntryKind::Payload(payload) => assert_eq!(payload.tags, ["vision"]),
        other => panic!("{other:?}"),
    }
}

fn tree(root: &Path) -> Vec<(String, Vec<u8>)> {
    let mut files = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for entry in fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                stack.push(path);
            } else {
                let relative = path.strip_prefix(root).unwrap().display().to_string();
                files.push((relative, fs::read(&path).unwrap()));
            }
        }
    }
    files.sort();
    files
}

// The same model gives the same bytes, and so the same content id, in
// whatever order and from however many threads its files were added.
#[test]
fn the_same_model_always_gives_the_same_bytes() {
    let first = tempfile::tempdir().unwrap();
    let second = tempfile::tempdir().unwrap();
    let a = write_tiny(first.path(), false);
    let b = write_tiny(second.path(), true);
    assert_eq!(a.content_id, b.content_id);
    assert_eq!(tree(first.path()), tree(second.path()));
}

#[test]
fn refuses_misuse() {
    let dir = tempfile::tempdir().unwrap();
    fs::write(dir.path().join("stray"), b"").unwrap();
    let error = Writer::with_algorithms(dir.path(), info(), &algorithms()).unwrap_err();
    assert!(error.to_string().contains("isn't empty"), "{error}");

    let dir = tempfile::tempdir().unwrap();
    let writer = Writer::with_algorithms(dir.path().join("model"), info(), &algorithms()).unwrap();
    let norm = vec![0u8; 128];
    let tensor = TensorData {
        name: "norm",
        dtype: Dtype::F32,
        shape: &[32],
        layout: Layout::RowMajor,
        scheme: Scheme::F32,
        bytes: &norm,
    };
    writer.add_file("a.bin", &[], &[tensor]).unwrap();
    let error = writer
        .add_file(
            "a.bin",
            &[],
            &[TensorData {
                name: "other",
                ..tensor
            }],
        )
        .unwrap_err();
    assert!(error.to_string().contains("added twice"), "{error}");

    let error = writer
        .add_file(
            "b.bin",
            &[],
            &[TensorData {
                shape: &[31],
                ..tensor
            }],
        )
        .unwrap_err();
    assert!(
        error.to_string().contains("isn't what scheme f32 gives"),
        "{error}"
    );

    let error = writer.add_file("hashes/x.bin", &[], &[]).unwrap_err();
    assert!(error.to_string().contains("reserved"), "{error}");

    let group = writer
        .add_group(GroupSpec {
            name: "experts".to_owned(),
            path: "experts/{expert}.bin".to_owned(),
            axes: vec![Axis::counted("expert", 2)],
            tags: vec![],
            tensors: vec![TensorSpec {
                name: "experts.{expert}".to_owned(),
                dtype: Dtype::U8,
                shape: vec![4],
                layout: Layout::RowMajor,
                scheme: Scheme::U8,
            }],
        })
        .unwrap();
    let error = writer.add_instance(group, &[2], &[&[0; 4]]).unwrap_err();
    assert!(error.to_string().contains("isn't an instance"), "{error}");
    let error = writer.add_instance(group, &[0], &[&[0; 5]]).unwrap_err();
    assert!(error.to_string().contains("don't fit"), "{error}");
    writer.add_instance(group, &[0], &[&[0; 4]]).unwrap();
    let error = writer.add_instance(group, &[0], &[&[0; 4]]).unwrap_err();
    assert!(error.to_string().contains("added twice"), "{error}");

    let error = writer.finish().unwrap_err();
    assert!(
        error
            .to_string()
            .contains("has 1 of its 2 instances written"),
        "{error}"
    );
}

#[test]
fn needs_aes_for_gxhash() {
    let dir = tempfile::tempdir().unwrap();
    let result = Writer::create(dir.path(), info());
    if Algorithm::Gxhash3.is_supported() {
        result.unwrap();
    } else {
        let error = result.unwrap_err();
        assert!(error.to_string().contains("AES"), "{error}");
    }
}

fn spec(path: &str, count: u64) -> GroupSpec {
    GroupSpec {
        name: "g".to_owned(),
        path: path.to_owned(),
        axes: vec![Axis::counted("e", count)],
        tags: vec![],
        tensors: vec![TensorSpec {
            name: "g.{e}".to_owned(),
            dtype: Dtype::U8,
            shape: vec![4],
            layout: Layout::RowMajor,
            scheme: Scheme::U8,
        }],
    }
}

// No instance path leaves the root: the group is refused before anything
// is written.
#[test]
fn writes_nothing_outside_its_root() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("model");
    let writer = Writer::with_algorithms(&root, info(), &algorithms()).unwrap();
    let outside = dir.path().join("outside");
    for path in [
        "../escaped-{e}.bin".to_owned(),
        format!("{}/abs-{{e}}.bin", outside.display()),
        "a/../../b-{e}.bin".to_owned(),
        "hashes/{e}.bin".to_owned(),
    ] {
        let error = writer.add_group(spec(&path, 2)).unwrap_err();
        assert!(error.to_string().contains("path"), "{path}: {error}");
    }
    assert!(!outside.exists());
    let escaped: Vec<_> = fs::read_dir(dir.path())
        .unwrap()
        .map(|entry| entry.unwrap().file_name())
        .collect();
    assert_eq!(escaped, ["model"]);
}

// Each write has a file of its own until its rename: one a crash left, or
// another writer's still open, stays as it is, and of writes of one path,
// one wins whole and the others leave nothing behind.
#[test]
fn writes_new_files_under_names_of_their_own() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("table.bin");
    let others: Vec<PathBuf> = [
        ".partial".to_owned(),
        format!(".partial-{}-0", std::process::id()),
    ]
    .iter()
    .map(|suffix| dir.path().join(format!("table.bin{suffix}")))
    .collect();
    let open: Vec<File> = others
        .iter()
        .map(|other| {
            fs::write(other, b"another write").unwrap();
            File::options().append(true).open(other).unwrap()
        })
        .collect();
    let results: Vec<_> = std::thread::scope(|scope| {
        let writes: Vec<_> = (0..8u8)
            .map(|n| {
                let path = &path;
                scope.spawn(move || write_new(path, &vec![n; 1 << 20]).map(|()| n))
            })
            .collect();
        writes
            .into_iter()
            .map(|write| write.join().unwrap())
            .collect()
    });
    let won: Vec<u8> = results
        .iter()
        .filter_map(|result| result.as_ref().ok().copied())
        .collect();
    assert_eq!(won.len(), 1, "{results:?}");
    for result in &results {
        if let Err(error) = result {
            assert!(
                matches!(error, Error::Io { source, .. }
                    if source.kind() == std::io::ErrorKind::AlreadyExists),
                "{error:?}"
            );
        }
    }
    assert_eq!(fs::read(&path).unwrap(), vec![won[0]; 1 << 20]);
    drop(open);
    for other in &others {
        assert_eq!(fs::read(other).unwrap(), b"another write");
    }
    let mut names: Vec<PathBuf> = fs::read_dir(dir.path())
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .collect();
    names.sort();
    let mut expected = others.clone();
    expected.push(path);
    expected.sort();
    assert_eq!(names, expected);
}

// A group's instances take their paths and tensor names as it is added: a
// group that would collide with what was added before is refused, and what
// is added after can't take what its instances will need.
#[test]
fn takes_a_groups_instances_as_it_is_added() {
    let dir = tempfile::tempdir().unwrap();
    let writer = Writer::with_algorithms(dir.path().join("model"), info(), &algorithms()).unwrap();
    let bytes = [0u8; 4];
    let tensor = |name| TensorData {
        name,
        dtype: Dtype::U8,
        shape: &[4],
        layout: Layout::RowMajor,
        scheme: Scheme::U8,
        bytes: &bytes,
    };
    let group = |name: &str, path: &str, tensor: &str| GroupSpec {
        name: name.to_owned(),
        path: path.to_owned(),
        axes: vec![Axis::counted("e", 3)],
        tags: vec![],
        tensors: vec![TensorSpec {
            name: tensor.to_owned(),
            dtype: Dtype::U8,
            shape: vec![4],
            layout: Layout::RowMajor,
            scheme: Scheme::U8,
        }],
    };
    writer
        .add_file("a/2.bin", &[], &[tensor("a-file")])
        .unwrap();
    writer.add_file("b.bin", &[], &[tensor("b.1")]).unwrap();
    writer
        .add_file("C/0.bin", &[], &[tensor("c-file")])
        .unwrap();
    writer.add_file("d", &[], &[tensor("d-file")]).unwrap();
    for (spec, needle) in [
        (
            group("a", "a/{e}.bin", "a.{e}"),
            "group \"a\": a/2.bin was added twice",
        ),
        (
            group("b", "b/{e}.bin", "b.{e}"),
            "group \"b\": two tensors are named \"b.1\"",
        ),
        (
            group("c", "c/{e}.bin", "c.{e}"),
            "group \"c\": c/0.bin and C/0.bin name one file where case is ignored",
        ),
        (
            group("d", "d/{e}.bin", "d.{e}"),
            "group \"d\": d is a file, but d/0.bin needs it as a directory",
        ),
        (group("s", "s/{e}.bin", "same"), "doesn't use axis \"e\""),
        (
            group("t", "t/{e}{e}.bin", "t.{e}"),
            "nothing but digits between them",
        ),
    ] {
        let error = writer.add_group(spec).unwrap_err();
        assert!(error.to_string().contains(needle), "{error}");
    }
    // A refused group takes nothing.
    writer.add_file("a/0.bin", &[], &[tensor("a.0")]).unwrap();

    let e = writer.add_group(group("e", "e/{e}.bin", "e.{e}")).unwrap();
    let error = writer.add_file("e/1.bin", &[], &[]).unwrap_err();
    assert!(
        error.to_string().contains("e/1.bin was added twice"),
        "{error}"
    );
    let error = writer.add_file("x.bin", &[], &[tensor("e.2")]).unwrap_err();
    assert!(
        error.to_string().contains("two tensors are named \"e.2\""),
        "{error}"
    );
    let error = writer
        .add_group(group("f", "e/{e}.bin", "f.{e}"))
        .unwrap_err();
    assert!(
        error.to_string().contains("e/0.bin was added twice"),
        "{error}"
    );
    writer.add_instance(e, &[1], &[&bytes]).unwrap();
    let error = writer.add_instance(e, &[1], &[&bytes]).unwrap_err();
    assert!(
        error.to_string().contains("e/1.bin was added twice"),
        "{error}"
    );
    let written: Vec<_> = fs::read_dir(dir.path().join("model/e"))
        .unwrap()
        .map(|entry| entry.unwrap().file_name().into_string().unwrap())
        .collect();
    assert_eq!(written, ["1.bin"]);
}

// A group refused for a conflict takes nothing, however far it got: none
// of its instances' paths and names, and no share of the limits.
#[test]
fn refuses_a_group_whole() {
    let dir = tempfile::tempdir().unwrap();
    let writer = Writer::with_algorithms(dir.path().join("model"), info(), &algorithms()).unwrap();
    let group = |name: &str| GroupSpec {
        name: name.to_owned(),
        path: format!("{name}/{{e}}.bin"),
        axes: vec![Axis::counted("e", 200_000)],
        tags: vec![],
        tensors: vec![TensorSpec {
            name: format!("{name}.{{e}}"),
            dtype: Dtype::U8,
            shape: vec![4],
            layout: Layout::RowMajor,
            scheme: Scheme::U8,
        }],
    };
    writer.add_file("big/199999.bin", &[], &[]).unwrap();
    let error = writer.add_group(group("big")).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("group \"big\": big/199999.bin was added twice"),
        "{error}"
    );
    // Its first 199,999 instances were given back, and its 200,000 files
    // to the limit of 262,144: a second group as large fits.
    writer.add_file("big/0.bin", &[], &[]).unwrap();
    writer.add_group(group("other")).unwrap();
}

// A path is checked against those taken in time and memory in proportion
// to its bytes, however deep it is: here 16,384 paths of 508 segments.
#[test]
fn takes_deep_paths_in_proportion_to_their_length() {
    let dir = tempfile::tempdir().unwrap();
    let writer = Writer::with_algorithms(dir.path().join("model"), info(), &algorithms()).unwrap();
    let deep = "x/".repeat(505);
    let start = Instant::now();
    for a in 0..128 {
        for b in 0..128 {
            writer
                .reserve(&format!("{a}/{b}/{deep}y"), &[], None, |_| Ok(()))
                .unwrap();
        }
    }
    let error = writer
        .reserve(&format!("0/0/{deep}y/z"), &[], None, |_| Ok(()))
        .unwrap_err();
    let elapsed = start.elapsed();
    assert!(
        error.to_string().contains("needs it as a directory"),
        "{error}"
    );
    assert!(elapsed < Duration::from_secs(10), "took {elapsed:?}");
}

// Every rule is checked as a file is added, so that finish has nothing
// left to refuse and a conversion isn't lost at its end.
#[test]
fn checks_everything_as_files_are_added() {
    let dir = tempfile::tempdir().unwrap();
    let mut bad = info();
    bad.name = String::new();
    let error = Writer::with_algorithms(dir.path().join("a"), bad, &algorithms()).unwrap_err();
    assert!(error.to_string().contains("name and arch"), "{error}");
    assert!(!dir.path().join("a").exists());
    let mut bad = info();
    bad.config = serde_json::json!([1]);
    assert!(Writer::with_algorithms(dir.path().join("a"), bad, &algorithms()).is_err());

    let writer = Writer::with_algorithms(dir.path().join("b"), info(), &algorithms()).unwrap();
    let norm = vec![0u8; 16];
    let tensor = |name| TensorData {
        name,
        dtype: Dtype::U8,
        shape: &[16],
        layout: Layout::RowMajor,
        scheme: Scheme::U8,
        bytes: &norm,
    };
    writer
        .add_aux("aux/tokenizer.json", "tokenizer", b"{}")
        .unwrap();
    let error = writer
        .add_aux("aux/other.json", "tokenizer", b"{}")
        .unwrap_err();
    assert!(error.to_string().contains("role \"tokenizer\""), "{error}");
    let error = writer.add_aux("aux/other.json", "", b"{}").unwrap_err();
    assert!(error.to_string().contains("empty role"), "{error}");
    let error = writer.add_aux("FURYMODEL.JSON", "x", b"{}").unwrap_err();
    assert!(error.to_string().contains("reserved"), "{error}");

    writer.add_file("A.bin", &[], &[tensor("a")]).unwrap();
    let error = writer.add_file("a.bin", &[], &[tensor("b")]).unwrap_err();
    assert!(
        error.to_string().contains("where case is ignored"),
        "{error}"
    );
    let error = writer
        .add_file("a.BIN/x.bin", &[], &[tensor("c")])
        .unwrap_err();
    assert!(
        error.to_string().contains("needs it as a directory"),
        "{error}"
    );
    writer.add_file("dir/x.bin", &[], &[tensor("d")]).unwrap();
    let error = writer.add_file("DIR", &[], &[tensor("e")]).unwrap_err();
    assert!(
        error.to_string().contains("needs it as a directory"),
        "{error}"
    );
    // However many paths sort between the two byte by byte: "-" and "."
    // come before "/".
    for path in ["p", "p-x", "p.x", "q/x", "q-x", "q.x"] {
        writer.add_file(path, &[], &[]).unwrap();
    }
    let error = writer.add_file("P/x", &[], &[]).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("p is a file, but P/x needs it as a directory"),
        "{error}"
    );
    let error = writer.add_file("Q", &[], &[]).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("Q is a file, but q/x needs it as a directory"),
        "{error}"
    );
    let error = writer.add_file("c.bin", &[], &[tensor("a")]).unwrap_err();
    assert!(
        error.to_string().contains("two tensors are named \"a\""),
        "{error}"
    );

    let error = writer.add_group(spec("g/{e}.bin", 300_000)).unwrap_err();
    assert!(error.to_string().contains("instances"), "{error}");
    let group = writer.add_group(spec("x{e}y{e}.bin", 12)).unwrap();
    for value in 0..12 {
        writer.add_instance(group, &[value], &[&[0; 4]]).unwrap();
    }
    writer.finish().unwrap();
    let left: Vec<_> = fs::read_dir(dir.path().join("b/hashes"))
        .unwrap()
        .map(|entry| entry.unwrap().file_name().into_string().unwrap())
        .filter(|name| name.contains(".partial"))
        .collect();
    assert_eq!(left, Vec::<String>::new());
}
