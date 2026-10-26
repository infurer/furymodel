// SPDX-FileCopyrightText: 2026 Mika Tammi
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Comparing two models by their manifests and hash tables, without reading
//! their payload files.

use std::collections::{BTreeMap, BTreeSet};

use crate::hash::Algorithm;
use crate::manifest::{EntryKind, MANIFEST_PATH, TensorEntry};
use crate::model::Model;

/// How a new model differs from an old one.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Diff {
    /// Fields of the manifest besides its files that differ: top-level
    /// fields, and keys of `config` and `extensions` as `config.KEY`; and
    /// `manifest` when the manifests' bytes differ in nothing else this
    /// lists, such as how its files are described.
    pub fields: Vec<String>,
    /// Files only the old model has.
    pub removed: Vec<String>,
    /// Files only the new model has.
    pub added: Vec<String>,
    /// Files both have whose sizes or contents differ.
    pub changed: Vec<String>,
    /// Files both have whose tags (payload files) or role (auxiliary files)
    /// differ, or that are a payload file in one model and an auxiliary file
    /// in the other.
    pub relabeled: Vec<String>,
    /// Files both have that no hash table of both can compare.
    pub uncompared: Vec<String>,
    /// The algorithm contents were compared with: the first of sha2-512,
    /// sha3-512, blake3, sha2-256, xxh3-128 and gxhash3-128 that both
    /// models have a table of.
    pub algorithm: Option<Algorithm>,
    /// Tensors only the old model has.
    pub removed_tensors: Vec<String>,
    /// Tensors only the new model has.
    pub added_tensors: Vec<String>,
    /// Tensors both have, described differently: the old description and
    /// the new.
    pub changed_tensors: Vec<(String, String, String)>,
}

impl Diff {
    /// Whether the models are the same as far as the comparison can tell.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.fields.is_empty()
            && self.removed.is_empty()
            && self.added.is_empty()
            && self.changed.is_empty()
            && self.relabeled.is_empty()
            && self.uncompared.is_empty()
            && self.removed_tensors.is_empty()
            && self.added_tensors.is_empty()
            && self.changed_tensors.is_empty()
    }
}

/// Compares `new` with `old`. The manifest itself isn't compared as a
/// file, as it differs whenever anything does, but field by field.
#[must_use]
pub fn diff(old: &Model, new: &Model) -> Diff {
    let mut diff = Diff {
        fields: fields(old, new),
        algorithm: [
            Algorithm::Sha512,
            Algorithm::Sha3,
            Algorithm::Blake3,
            Algorithm::Sha256,
            Algorithm::Xxh3,
            Algorithm::Gxhash3,
        ]
        .into_iter()
        .find(|&algorithm| {
            old.hash_table(algorithm).is_some() && new.hash_table(algorithm).is_some()
        }),
        ..Diff::default()
    };

    let old_files: BTreeMap<&str, usize> = files(old);
    let new_files: BTreeMap<&str, usize> = files(new);
    for (&path, &old_index) in &old_files {
        let Some(&new_index) = new_files.get(path) else {
            diff.removed.push(path.to_owned());
            continue;
        };
        let (old_entry, new_entry) = (
            &old.table().entries()[old_index],
            &new.table().entries()[new_index],
        );
        if label(&old_entry.kind) != label(&new_entry.kind) {
            diff.relabeled.push(path.to_owned());
        }
        let digests = diff.algorithm.and_then(|algorithm| {
            Some((
                old.hash_table(algorithm)?.file(old_index),
                new.hash_table(algorithm)?.file(new_index),
            ))
        });
        match digests {
            _ if old_entry.size != new_entry.size => diff.changed.push(path.to_owned()),
            Some((old_digest, new_digest)) if old_digest != new_digest => {
                diff.changed.push(path.to_owned());
            }
            Some(_) => {}
            None => diff.uncompared.push(path.to_owned()),
        }
    }
    diff.added = new_files
        .keys()
        .filter(|path| !old_files.contains_key(*path))
        .map(|&path| path.to_owned())
        .collect();

    let old_tensors = tensors(old);
    let new_tensors = tensors(new);
    for (name, old_description) in &old_tensors {
        match new_tensors.get(name) {
            None => diff.removed_tensors.push(name.clone()),
            Some(new_description) if new_description != old_description => {
                diff.changed_tensors.push((
                    name.clone(),
                    old_description.clone(),
                    new_description.clone(),
                ));
            }
            Some(_) => {}
        }
    }
    diff.added_tensors = new_tensors
        .keys()
        .filter(|name| !old_tensors.contains_key(*name))
        .cloned()
        .collect();

    // Whatever else makes the manifests differ, as their digests tell: two
    // models whose files are all the same but for the manifest still
    // differ.
    if diff.is_empty() {
        let manifests = diff.algorithm.and_then(|algorithm| {
            let (old_index, new_index) = (
                old.table().index_of(MANIFEST_PATH)?,
                new.table().index_of(MANIFEST_PATH)?,
            );
            Some(
                old.hash_table(algorithm)?.file(old_index)
                    == new.hash_table(algorithm)?.file(new_index),
            )
        });
        match manifests {
            Some(true) => {}
            Some(false) => diff.fields.push("manifest".to_owned()),
            None => diff.uncompared.push(MANIFEST_PATH.to_owned()),
        }
    }
    diff
}

// What a file is labeled with besides its contents: a payload file's tags,
// an auxiliary file's role.
fn label(kind: &EntryKind) -> (u8, Vec<&str>) {
    match kind {
        EntryKind::Manifest => (0, Vec::new()),
        EntryKind::Payload(payload) => (1, payload.tags.iter().map(String::as_str).collect()),
        EntryKind::Aux { role } => (2, vec![role.as_str()]),
    }
}

fn files(model: &Model) -> BTreeMap<&str, usize> {
    model
        .table()
        .entries()
        .iter()
        .enumerate()
        .filter(|(_, entry)| entry.path != MANIFEST_PATH)
        .map(|(index, entry)| (entry.path.as_str(), index))
        .collect()
}

fn tensors(model: &Model) -> BTreeMap<String, String> {
    model
        .table()
        .tensors()
        .map(|(file, tensor)| {
            (
                tensor.name.clone(),
                describe(tensor, &model.table().entries()[file].path),
            )
        })
        .collect()
}

/// A tensor's description, as a diff shows it.
#[must_use]
pub fn describe(tensor: &TensorEntry, path: &str) -> String {
    format!(
        "{} {:?} {} {} in {path} at {}",
        tensor.dtype, tensor.shape, tensor.layout, tensor.scheme, tensor.offset
    )
}

fn fields(old: &Model, new: &Model) -> Vec<String> {
    let (old, new) = (old.manifest(), new.manifest());
    let mut fields = Vec::new();
    let mut compare = |name: &str, same: bool| {
        if !same {
            fields.push(name.to_owned());
        }
    };
    compare("name", old.name == new.name);
    compare("arch", old.arch == new.arch);
    compare("schema_version", old.schema_version == new.schema_version);
    compare("provenance", old.provenance == new.provenance);
    match (old.config.as_object(), new.config.as_object()) {
        (Some(old_config), Some(new_config)) => {
            let keys: BTreeSet<&String> = old_config.keys().chain(new_config.keys()).collect();
            for key in keys {
                compare(
                    &format!("config.{key}"),
                    old_config.get(key) == new_config.get(key),
                );
            }
        }
        _ => compare("config", old.config == new.config),
    }
    let keys: BTreeSet<&String> = old.extensions.keys().chain(new.extensions.keys()).collect();
    for key in keys {
        compare(
            &format!("extensions.{key}"),
            old.extensions.get(key) == new.extensions.get(key),
        );
    }
    fields
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::diff;
    use crate::hash::Algorithm;
    use crate::manifest::MANIFEST_PATH;
    use crate::model::Model;
    use crate::registry::{Dtype, Layout, Scheme};
    use crate::testing::{algorithms, info, rehash, write_tiny};
    use crate::writer::{TensorData, Writer};

    #[test]
    fn same_model_no_differences() {
        let dir = tempfile::tempdir().unwrap();
        write_tiny(&dir.path().join("a"), false);
        write_tiny(&dir.path().join("b"), true);
        let a = Model::open(&[dir.path().join("a")]).unwrap();
        let b = Model::open(&[dir.path().join("b")]).unwrap();
        let found = diff(&a, &b);
        assert!(found.is_empty(), "{found:?}");
        assert_eq!(found.algorithm, Some(Algorithm::Sha512));
    }

    #[test]
    fn finds_what_changed() {
        let dir = tempfile::tempdir().unwrap();
        write_tiny(&dir.path().join("a"), false);
        write_tiny(&dir.path().join("b"), false);

        // Change a manifest field, which changes nothing else; and give the
        // models tables of different algorithms, so that there is none to
        // compare contents with.
        let b = dir.path().join("b");
        let json = fs::read_to_string(b.join(MANIFEST_PATH)).unwrap();
        fs::remove_file(b.join(MANIFEST_PATH)).unwrap();
        fs::write(
            b.join(MANIFEST_PATH),
            json.replace("\"layers\": 2", "\"layers\": 3"),
        )
        .unwrap();
        rehash(&b, &[Algorithm::Xxh3]);
        rehash(&dir.path().join("a"), &[Algorithm::Sha3]);
        let a = Model::open(&[dir.path().join("a")]).unwrap();
        let b = Model::open(&[&b]).unwrap();
        let found = diff(&a, &b);
        assert_eq!(found.fields, ["config.layers"]);
        assert_eq!(found.algorithm, None);
        assert_eq!(found.uncompared.len(), 9);
        assert!(found.changed.is_empty() && found.added.is_empty());
    }

    #[test]
    fn compares_contents_by_digest() {
        let dir = tempfile::tempdir().unwrap();
        write_tiny(&dir.path().join("a"), false);
        write_tiny(&dir.path().join("b"), false);
        let a = Model::open(&[dir.path().join("a")]).unwrap();
        // Other bytes of the same size, and a sha2-512 table of them.
        let b = dir.path().join("b");
        let tokenizer = b.join("aux/tokenizer.json");
        let mut bytes = fs::read(&tokenizer).unwrap();
        bytes[3] ^= 1;
        fs::write(&tokenizer, &bytes).unwrap();
        rehash(&b, &[Algorithm::Sha512]);
        let found = diff(&a, &Model::open(&[&b]).unwrap());
        assert_eq!(found.algorithm, Some(Algorithm::Sha512));
        assert_eq!(found.changed, ["aux/tokenizer.json"]);
        assert!(found.fields.is_empty() && found.uncompared.is_empty());
    }

    // Files labeled otherwise, and manifests that differ in nothing else,
    // make models differ too: their content ids do.
    #[test]
    fn finds_tags_roles_and_manifests_that_differ() {
        let dir = tempfile::tempdir().unwrap();
        write_tiny(&dir.path().join("a"), false);
        let a = Model::open(&[dir.path().join("a")]).unwrap();
        let rewrite = |name: &str, change: &dyn Fn(&str) -> String| {
            let root = dir.path().join(name);
            write_tiny(&root, false);
            let json = fs::read_to_string(root.join(MANIFEST_PATH)).unwrap();
            fs::remove_file(root.join(MANIFEST_PATH)).unwrap();
            fs::write(root.join(MANIFEST_PATH), change(&json)).unwrap();
            rehash(&root, &algorithms());
            Model::open(&[&root]).unwrap()
        };

        let b = rewrite("b", &|json| {
            json.replacen("\"vision\"", "\"drafter\"", 1)
                .replace("\"role\": \"tokenizer\"", "\"role\": \"vocab\"")
        });
        let found = diff(&a, &b);
        assert_eq!(found.relabeled, ["aux/tokenizer.json", "global/big.bin"]);
        assert!(
            found.fields.is_empty() && found.changed.is_empty(),
            "{found:?}"
        );

        // The same manifest but for its whitespace.
        let c = rewrite("c", &|json| json.replace("{\n", "{ \n"));
        assert_ne!(a.content_id(), c.content_id());
        let found = diff(&a, &c);
        assert_eq!(found.fields, ["manifest"]);
        assert!(!found.is_empty());
    }

    #[test]
    fn finds_added_and_removed_files_and_tensors() {
        let dir = tempfile::tempdir().unwrap();
        write_tiny(&dir.path().join("a"), false);
        let a = Model::open(&[dir.path().join("a")]).unwrap();
        let writer = Writer::with_algorithms(dir.path().join("c"), info(), &algorithms()).unwrap();
        let norm = vec![1u8; 128];
        writer
            .add_file(
                "global/embed.bin",
                &["text"],
                &[TensorData {
                    name: "norm",
                    dtype: Dtype::Bf16,
                    shape: &[64],
                    layout: Layout::RowMajor,
                    scheme: Scheme::Bf16,
                    bytes: &norm,
                }],
            )
            .unwrap();
        writer.add_aux("aux/extra.txt", "extra", b"x").unwrap();
        writer.finish().unwrap();
        let c = Model::open(&[dir.path().join("c")]).unwrap();
        let found = diff(&a, &c);
        assert_eq!(found.added, ["aux/extra.txt"]);
        assert_eq!(found.removed.len(), 2 + 6);
        assert_eq!(found.changed, ["global/embed.bin"]);
        assert_eq!(found.added_tensors, Vec::<String>::new());
        assert_eq!(found.removed_tensors.len(), 1 + 1 + 2 * 6);
        assert_eq!(found.changed_tensors.len(), 1);
        let (name, old, new) = &found.changed_tensors[0];
        assert_eq!(name, "norm");
        assert!(
            old.starts_with("f32 [32]") && new.starts_with("bf16 [64]"),
            "{old} {new}"
        );
    }
}
