// SPDX-FileCopyrightText: 2026 Mika Tammi
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The commands, run as a user runs them, against a small model.

use std::path::Path;
use std::process::{Command, Output};

use furymodel::hash::Algorithm;
use furymodel::manifest::{Provenance, Source, Tool};
use furymodel::writer::{ModelInfo, TensorData};
use furymodel::{Dtype, Layout, Scheme, Writer};

fn algorithms() -> Vec<Algorithm> {
    Algorithm::ALL
        .into_iter()
        .filter(|algorithm| algorithm.is_supported())
        .collect()
}

fn write_model(root: &Path, weights: u8) {
    write_named(root, weights, "cli");
}

fn write_named(root: &Path, weights: u8, name: &str) {
    let info = ModelInfo {
        name: name.to_owned(),
        arch: "testarch".to_owned(),
        schema_version: 1,
        config: serde_json::json!({"hidden": 16}),
        provenance: Provenance {
            sources: vec![Source {
                kind: "synthetic".to_owned(),
                repository: Some("example/cli".to_owned()),
                revision: Some("abc".to_owned()),
                files: vec![],
            }],
            tool: Tool {
                name: "furymodel-cli tests".to_owned(),
                version: "0".to_owned(),
            },
            transforms: vec![],
            notes: vec!["made by the tests".to_owned()],
        },
        extensions: std::collections::BTreeMap::new(),
    };
    let writer = Writer::with_algorithms(root, info, &algorithms()).unwrap();
    for (path, size) in [("weights/a.bin", 3 << 20), ("weights/b.bin", 2 << 20)] {
        let bytes = vec![weights; size];
        writer
            .add_file(
                path,
                &["text"],
                &[TensorData {
                    name: path,
                    dtype: Dtype::U8,
                    shape: &[size as u64],
                    layout: Layout::RowMajor,
                    scheme: Scheme::U8,
                    bytes: &bytes,
                }],
            )
            .unwrap();
    }
    writer.add_aux("aux/notes.txt", "notes", b"hello").unwrap();
    writer.finish().unwrap();
}

fn run(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_furymodel-cli"))
        .args(args)
        .output()
        .unwrap()
}

fn text(output: &Output) -> String {
    format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    )
}

#[test]
fn info_verify_hash_add_split_and_diff() {
    let dir = tempfile::tempdir().unwrap();
    let model = dir.path().join("model");
    write_model(&model, 7);
    let model = model.to_str().unwrap();

    let output = run(&["info", model]);
    assert!(output.status.success(), "{}", text(&output));
    let info = text(&output);
    assert!(info.contains("content id   "), "{info}");
    assert!(info.contains("synthetic from example/cli at abc"), "{info}");
    assert!(info.contains("u8 2 of 5.0 MiB"), "{info}");

    let output = run(&["info", "--json", model]);
    let json: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(json["files"], 4);
    assert_eq!(json["tags"]["text"]["files"], 2);

    let output = run(&["verify", "--all", model]);
    assert!(output.status.success(), "{}", text(&output));
    assert!(text(&output).contains("all match"), "{}", text(&output));
    assert!(text(&output).contains("authentic"), "{}", text(&output));

    // A missing table, added back.
    std::fs::remove_file(Path::new(model).join(Algorithm::Sha3.table_path())).unwrap();
    let output = run(&["hash", "add", "sha3-512", model]);
    assert!(output.status.success(), "{}", text(&output));
    let output = run(&["hash", "add", "sha3-512", model]);
    assert_eq!(output.status.code(), Some(2), "{}", text(&output));
    assert!(text(&output).contains("already"), "{}", text(&output));

    let parts = dir.path().join("parts");
    let output = run(&[
        "split",
        "--max-part-size",
        "4MiB",
        "--out",
        parts.to_str().unwrap(),
        model,
    ]);
    assert!(output.status.success(), "{}", text(&output));
    let listed: Vec<String> = String::from_utf8(output.stdout)
        .unwrap()
        .lines()
        .map(str::to_owned)
        .collect();
    assert_eq!(listed.len(), 2, "{listed:?}");
    let mut verify = vec!["verify"];
    verify.extend(listed.iter().map(String::as_str));
    let output = run(&verify);
    assert!(output.status.success(), "{}", text(&output));

    let other = dir.path().join("other");
    write_model(&other, 8);
    let output = run(&["diff", "--old", model, "--new", other.to_str().unwrap()]);
    assert_eq!(output.status.code(), Some(1), "{}", text(&output));
    assert!(
        text(&output).contains("changed: weights/a.bin"),
        "{}",
        text(&output)
    );
    let output = run(&["diff", "--old", model, "--new", &listed[0], &listed[1]]);
    assert!(output.status.success(), "{}", text(&output));

    // A corrupt file.
    let path = Path::new(model).join("weights/b.bin");
    std::fs::write(&path, vec![9u8; 2 << 20]).unwrap();
    let output = run(&["verify", model]);
    assert_eq!(output.status.code(), Some(1), "{}", text(&output));
    assert!(
        text(&output).contains("mismatch: weights/b.bin, chunk 0"),
        "{}",
        text(&output)
    );

    let output = run(&["verify", "--algorithm", "md5", model]);
    assert_eq!(output.status.code(), Some(2));
    assert!(
        text(&output).contains("isn't an algorithm"),
        "{}",
        text(&output)
    );
}

// A model's strings reach the terminal with their control characters
// escaped: a downloaded model can't move the cursor or recolor the screen.
// With no table to check its files against, a model gets a first one only
// by its content id, or by its files taken as they are.
#[test]
fn hash_add_gives_a_model_without_tables_a_first_one() {
    let dir = tempfile::tempdir().unwrap();
    let lost = dir.path().join("lost");
    write_model(&lost, 3);
    let lost = lost.to_str().unwrap();
    let json: serde_json::Value =
        serde_json::from_slice(&run(&["info", "--json", lost]).stdout).unwrap();
    let content_id = json["content_id"].as_str().unwrap().to_owned();
    std::fs::remove_dir_all(Path::new(lost).join("hashes")).unwrap();
    let output = run(&["hash", "add", "sha3-512", lost]);
    assert_eq!(output.status.code(), Some(2), "{}", text(&output));
    assert!(text(&output).contains("--expect"), "{}", text(&output));
    let wrong = "0".repeat(128);
    let output = run(&["hash", "add", "sha3-512", "--expect", &wrong, lost]);
    assert_eq!(output.status.code(), Some(2), "{}", text(&output));
    assert!(text(&output).contains("expected"), "{}", text(&output));
    assert!(!Path::new(lost).join(Algorithm::Sha3.table_path()).exists());
    let output = run(&["hash", "add", "sha3-512", "--expect", &content_id, lost]);
    assert!(output.status.success(), "{}", text(&output));
    let output = run(&["verify", "--algorithm", "sha3-512", lost]);
    assert!(output.status.success(), "{}", text(&output));
    assert!(text(&output).contains("authentic"), "{}", text(&output));
    let output = run(&["hash", "add", "xxh3-128", "--trust-files", lost]);
    assert_eq!(output.status.code(), Some(2), "{}", text(&output));
    assert!(
        text(&output).contains("without --expect or --trust-files"),
        "{}",
        text(&output)
    );
    std::fs::remove_dir_all(Path::new(lost).join("hashes")).unwrap();
    let output = run(&["hash", "add", "xxh3-128", "--trust-files", lost]);
    assert!(output.status.success(), "{}", text(&output));
    let output = run(&[
        "hash",
        "add",
        "sha3-512",
        "--expect",
        &content_id,
        "--trust-files",
        lost,
    ]);
    assert_eq!(output.status.code(), Some(2), "{}", text(&output));
    let output = run(&["hash", "add", "sha3-512", "--force", "--trust-files", lost]);
    assert_eq!(output.status.code(), Some(2), "{}", text(&output));
    assert!(
        text(&output).contains("cannot be used with"),
        "{}",
        text(&output)
    );
    let output = run(&[
        "hash",
        "add",
        "sha3-512",
        "--force",
        "--expect",
        &content_id,
        lost,
    ]);
    assert_eq!(output.status.code(), Some(2), "{}", text(&output));
    assert!(
        text(&output).contains("no sha3-512 table to replace"),
        "{}",
        text(&output)
    );
}

// A table added to a model in parts goes into one part: hash add says the
// others need it too.
#[test]
fn hash_add_into_parts_names_the_others() {
    let dir = tempfile::tempdir().unwrap();
    let model = dir.path().join("model");
    write_model(&model, 4);
    let parts = dir.path().join("parts");
    let output = run(&[
        "split",
        "--max-part-size",
        "4MiB",
        "--out",
        parts.to_str().unwrap(),
        model.to_str().unwrap(),
    ]);
    assert!(output.status.success(), "{}", text(&output));
    let mut roots: Vec<String> = std::fs::read_dir(&parts)
        .unwrap()
        .map(|entry| entry.unwrap().path().to_str().unwrap().to_owned())
        .collect();
    roots.sort();
    assert!(roots.len() >= 2, "{roots:?}");
    for root in &roots {
        std::fs::remove_file(Path::new(root).join(Algorithm::Blake3.table_path())).unwrap();
    }
    let mut args = vec!["hash", "add", "blake3"];
    args.extend(roots.iter().map(String::as_str));
    let output = run(&args);
    assert!(output.status.success(), "{}", text(&output));
    assert!(
        text(&output).contains("into the other parts too"),
        "{}",
        text(&output)
    );
}

// A table that keeps a model from opening, here one part's copy of it:
// opening says that hash add --force replaces it, which it then does in
// every part.
#[test]
fn hash_add_force_replaces_a_bad_table() {
    let dir = tempfile::tempdir().unwrap();
    let model = dir.path().join("model");
    write_model(&model, 5);
    let parts = dir.path().join("parts");
    let output = run(&[
        "split",
        "--max-part-size",
        "4MiB",
        "--out",
        parts.to_str().unwrap(),
        model.to_str().unwrap(),
    ]);
    assert!(output.status.success(), "{}", text(&output));
    let mut roots: Vec<String> = std::fs::read_dir(&parts)
        .unwrap()
        .map(|entry| entry.unwrap().path().to_str().unwrap().to_owned())
        .collect();
    roots.sort();
    assert!(roots.len() >= 2, "{roots:?}");
    let table = Path::new(&roots[1]).join(Algorithm::Sha256.table_path());
    let original = std::fs::read(&table).unwrap();
    let mut bytes = original.clone();
    bytes[200] ^= 1;
    std::fs::write(&table, &bytes).unwrap();
    let with_roots = |args: &[&str]| {
        let mut args = args.to_vec();
        args.extend(roots.iter().map(String::as_str));
        run(&args)
    };

    let output = with_roots(&["info"]);
    assert_eq!(output.status.code(), Some(2), "{}", text(&output));
    assert!(
        text(&output).contains("`hash add --force sha2-256` replaces"),
        "{}",
        text(&output)
    );
    let output = with_roots(&["hash", "add", "sha2-256"]);
    assert_eq!(output.status.code(), Some(2), "{}", text(&output));
    let output = with_roots(&["hash", "add", "--force", "sha2-256", "--into", &roots[0]]);
    assert_eq!(output.status.code(), Some(2), "{}", text(&output));
    let output = with_roots(&["hash", "add", "--force", "sha2-256"]);
    assert!(output.status.success(), "{}", text(&output));
    assert!(text(&output).contains("replaced"), "{}", text(&output));
    assert_eq!(std::fs::read(&table).unwrap(), original);
    let output = with_roots(&["verify", "--all"]);
    assert!(output.status.success(), "{}", text(&output));
    let output = with_roots(&["hash", "add", "--force", "sha2-256"]);
    assert_eq!(output.status.code(), Some(2), "{}", text(&output));
    assert!(
        text(&output).contains("nothing to replace"),
        "{}",
        text(&output)
    );
}

// Each of several bad tables is replaced in turn, every replacement a
// success that names the tables that keep the model from opening still;
// with every table bad, the model's content id vouches for a new sha3-512
// table first.
#[test]
fn hash_add_force_replaces_bad_tables_in_turn() {
    let dir = tempfile::tempdir().unwrap();
    let model = dir.path().join("model");
    write_model(&model, 6);
    let model = model.to_str().unwrap();
    let json: serde_json::Value =
        serde_json::from_slice(&run(&["info", "--json", model]).stdout).unwrap();
    let content_id = json["content_id"].as_str().unwrap().to_owned();
    let corrupt = |algorithm: Algorithm| {
        let path = Path::new(model).join(algorithm.table_path());
        let mut bytes = std::fs::read(&path).unwrap();
        bytes[200] ^= 1;
        std::fs::write(&path, &bytes).unwrap();
    };

    corrupt(Algorithm::Sha256);
    corrupt(Algorithm::Blake3);
    let output = run(&["hash", "add", "--force", "sha2-256", model]);
    assert!(output.status.success(), "{}", text(&output));
    assert!(
        text(&output).contains("the model doesn't open yet")
            && text(&output).contains("`hash add --force blake3` replaces"),
        "{}",
        text(&output)
    );
    let output = run(&["hash", "add", "--force", "blake3", model]);
    assert!(output.status.success(), "{}", text(&output));
    assert!(!text(&output).contains("doesn't open"), "{}", text(&output));

    let algorithms = algorithms();
    for &algorithm in &algorithms {
        corrupt(algorithm);
    }
    let output = run(&["hash", "add", "--force", "sha3-512", model]);
    assert_eq!(output.status.code(), Some(2), "{}", text(&output));
    assert!(
        text(&output).contains("--force --expect CONTENT_ID sha3-512"),
        "{}",
        text(&output)
    );
    let output = run(&[
        "hash",
        "add",
        "--force",
        "--expect",
        &content_id,
        "sha3-512",
        model,
    ]);
    assert!(output.status.success(), "{}", text(&output));
    assert_eq!(
        text(&output).matches("the model doesn't open yet").count(),
        algorithms.len() - 1,
        "{}",
        text(&output)
    );
    for algorithm in algorithms.iter().filter(|&&a| a != Algorithm::Sha3) {
        let output = run(&["hash", "add", "--force", algorithm.id(), model]);
        assert!(output.status.success(), "{}", text(&output));
    }
    let output = run(&["verify", "--expect", &content_id, model]);
    assert!(output.status.success(), "{}", text(&output));
    let output = run(&["verify", "--all", model]);
    assert!(output.status.success(), "{}", text(&output));
}

// verify --expect checks that the model is the one of a content id, in one
// pass; info shows the id as what the sha3-512 table claims.
#[test]
fn verify_expect_names_the_model() {
    let dir = tempfile::tempdir().unwrap();
    let model = dir.path().join("model");
    write_model(&model, 5);
    let model = model.to_str().unwrap();
    let json: serde_json::Value =
        serde_json::from_slice(&run(&["info", "--json", model]).stdout).unwrap();
    let content_id = json["content_id"].as_str().unwrap().to_owned();
    let output = run(&["info", model]);
    assert!(
        text(&output).contains("as the sha3-512 table claims"),
        "{}",
        text(&output)
    );
    let output = run(&["verify", "--expect", &content_id, model]);
    assert!(output.status.success(), "{}", text(&output));
    assert!(
        text(&output).contains("authentic, the one expected"),
        "{}",
        text(&output)
    );
    let output = run(&["verify", "--expect", &"0".repeat(128), model]);
    assert_eq!(output.status.code(), Some(1), "{}", text(&output));
    assert!(text(&output).contains("not the"), "{}", text(&output));
    let output = run(&["verify", "--expect", &content_id, "--all", model]);
    assert_eq!(output.status.code(), Some(2), "{}", text(&output));
}

#[test]
fn info_escapes_control_characters() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("model");
    write_named(&root, 1, "evil\u{1b}[31mred\u{202e}");
    let output = run(&["info", root.to_str().unwrap()]);
    assert!(output.status.success(), "{}", text(&output));
    let shown = text(&output);
    assert!(
        !shown.contains('\u{1b}') && !shown.contains('\u{202e}'),
        "{shown:?}"
    );
    assert!(shown.contains("evil\\u{1b}[31mred\\u{202e}"), "{shown}");
}
