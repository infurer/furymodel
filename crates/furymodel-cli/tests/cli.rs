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
