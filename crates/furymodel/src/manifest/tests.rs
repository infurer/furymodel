// SPDX-FileCopyrightText: 2026 Mika Tammi
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fmt::Write as _;
use std::time::{Duration, Instant};

use super::{
    ALIGNMENT, AuxEntry, Axis, EntryKind, FileEntry, GroupEntry, MANIFEST_PATH, Manifest,
    ManifestError, Provenance, Source, SourceFile, Template, TensorEntry, Tool, Totals, check_path,
};
use crate::registry::{Dtype, Layout, Scheme};

fn tensor(name: &str, shape: &[u64], offset: u64) -> TensorEntry {
    TensorEntry {
        name: name.to_owned(),
        dtype: Dtype::Bf16,
        shape: shape.to_vec(),
        layout: if shape.len() == 2 {
            Layout::OutIn
        } else {
            Layout::RowMajor
        },
        scheme: Scheme::Bf16,
        offset,
        size: Scheme::Bf16.size(shape).unwrap(),
    }
}

/// A small manifest using every kind of entry: a file, a group over two
/// axes, one over listed values, and an auxiliary file.
fn sample() -> Manifest {
    Manifest {
        format: "furymodel".to_owned(),
        format_version: 1,
        name: "tiny".to_owned(),
        arch: "tinyarch".to_owned(),
        schema_version: 1,
        config: serde_json::json!({"hidden": 64, "eps": 1e-6}),
        alignment: ALIGNMENT,
        files: vec![FileEntry {
            path: "global/embed.bin".to_owned(),
            size: 3 * ALIGNMENT,
            tags: vec!["text".to_owned()],
            tensors: vec![
                tensor("embed", &[256, 64], 0),
                tensor("norm", &[64], 2 * ALIGNMENT),
            ],
        }],
        groups: vec![
            GroupEntry {
                name: "experts".to_owned(),
                path: "layers/{layer:03}/experts/{expert:03}.bin".to_owned(),
                axes: vec![Axis::counted("layer", 2), Axis::counted("expert", 3)],
                size: 2 * ALIGNMENT,
                tags: vec!["text".to_owned()],
                tensors: vec![
                    tensor("layers.{layer}.experts.{expert}.down", &[64, 32], 0),
                    tensor("layers.{layer}.experts.{expert}.up", &[64, 64], ALIGNMENT),
                ],
            },
            GroupEntry {
                name: "global-layers".to_owned(),
                path: "layers/{layer:03}/layer.bin".to_owned(),
                axes: vec![Axis::listed("layer", vec![0, 1])],
                size: ALIGNMENT,
                tags: vec!["text".to_owned()],
                tensors: vec![tensor("layers.{layer}.q", &[64, 64], 0)],
            },
        ],
        aux: vec![AuxEntry {
            path: "aux/tokenizer.json".to_owned(),
            size: 12,
            role: "tokenizer".to_owned(),
        }],
        provenance: Provenance {
            sources: vec![Source {
                kind: "synthetic".to_owned(),
                repository: None,
                revision: None,
                files: vec![SourceFile {
                    path: "model.safetensors".to_owned(),
                    size: 1,
                    sha2_256: Some("0".repeat(64)),
                }],
            }],
            tool: Tool {
                name: "tests".to_owned(),
                version: "0".to_owned(),
            },
            transforms: vec![],
            notes: vec![],
        },
        extensions: std::collections::BTreeMap::new(),
    }
}

fn parse(manifest: &Manifest) -> Result<super::FileTable, ManifestError> {
    Manifest::parse(&manifest.to_json()).map(|(_, table)| table)
}

fn rejects(manifest: &Manifest, needle: &str) {
    match parse(manifest) {
        Ok(_) => panic!("accepted a manifest that should fail with {needle:?}"),
        Err(error) => assert!(
            error.to_string().contains(needle),
            "{error} doesn't mention {needle:?}"
        ),
    }
}

#[test]
fn parses_and_expands() {
    let manifest = sample();
    let json = manifest.to_json();
    let (parsed, table) = Manifest::parse(&json).unwrap();
    assert_eq!(parsed, manifest);
    assert_eq!(parsed.to_json(), json, "serialization is deterministic");

    let paths: Vec<&str> = table
        .entries()
        .iter()
        .map(|entry| entry.path.as_str())
        .collect();
    assert_eq!(
        paths,
        [
            "aux/tokenizer.json",
            "furymodel.json",
            "global/embed.bin",
            "layers/000/experts/000.bin",
            "layers/000/experts/001.bin",
            "layers/000/experts/002.bin",
            "layers/000/layer.bin",
            "layers/001/experts/000.bin",
            "layers/001/experts/001.bin",
            "layers/001/experts/002.bin",
            "layers/001/layer.bin",
        ]
    );
    let manifest_entry = &table.entries()[table.index_of(MANIFEST_PATH).unwrap()];
    assert_eq!(manifest_entry.size, json.len() as u64);
    assert_eq!(manifest_entry.kind, EntryKind::Manifest);

    let (file, down) = table.tensor("layers.1.experts.2.down").unwrap();
    assert_eq!(table.entries()[file].path, "layers/001/experts/002.bin");
    assert_eq!((down.offset, down.size), (0, 4096));
    match &table.entries()[file].kind {
        EntryKind::Payload(payload) => {
            let instance = payload.group.as_ref().unwrap();
            assert_eq!(instance.group, "experts");
            assert_eq!(instance.values, [1, 2]);
        }
        other => panic!("{other:?}"),
    }
    assert_eq!(table.tensors().count(), 2 + 2 * 3 * 2 + 2);
    assert_eq!(table.index_of("layers/002/layer.bin"), None);
}

#[test]
fn rejects_unknown_fields() {
    let mut json: serde_json::Value = serde_json::from_slice(&sample().to_json()).unwrap();
    json["files"][0]["tensors"][0]["stride"] = serde_json::json!(2);
    let error = Manifest::parse(&serde_json::to_vec(&json).unwrap()).unwrap_err();
    assert!(
        error.to_string().contains("unknown field `stride`"),
        "{error}"
    );

    let mut json: serde_json::Value = serde_json::from_slice(&sample().to_json()).unwrap();
    json["towers"] = serde_json::json!([]);
    assert!(Manifest::parse(&serde_json::to_vec(&json).unwrap()).is_err());

    // Anything goes inside config and extensions.
    let mut manifest = sample();
    manifest.config["anything"] = serde_json::json!({"nested": [1, 2]});
    manifest
        .extensions
        .insert("example.draft".to_owned(), serde_json::json!({"depth": 6}));
    parse(&manifest).unwrap();
}

#[test]
fn checks_format_and_version_first() {
    let mut json: serde_json::Value = serde_json::from_slice(&sample().to_json()).unwrap();
    json["format_version"] = serde_json::json!(2);
    json["new_in_version_2"] = serde_json::json!(true);
    let error = Manifest::parse(&serde_json::to_vec(&json).unwrap()).unwrap_err();
    assert!(matches!(error, ManifestError::Unsupported(_)), "{error}");
    assert!(error.to_string().contains("version 2"), "{error}");

    json["format"] = serde_json::json!("some-other-format");
    let error = Manifest::parse(&serde_json::to_vec(&json).unwrap()).unwrap_err();
    assert!(
        error.to_string().contains("not a FuryModel manifest"),
        "{error}"
    );
}

#[test]
fn checks_alignment_and_sizes() {
    let mut manifest = sample();
    manifest.files[0].size += 1;
    rejects(&manifest, "isn't a multiple of 16384");

    let mut manifest = sample();
    manifest.files[0].tensors[1].offset = ALIGNMENT + 512;
    rejects(&manifest, "isn't aligned");

    let mut manifest = sample();
    manifest.files[0].tensors[1].offset = 0;
    rejects(&manifest, "overlaps");

    let mut manifest = sample();
    manifest.files[0].tensors[1].size *= 2;
    rejects(&manifest, "isn't what scheme bf16 gives");

    let mut manifest = sample();
    manifest.files[0].tensors[1].offset = 3 * ALIGNMENT;
    rejects(&manifest, "ends past the file");

    let mut manifest = sample();
    manifest.files[0].tensors[0].scheme = Scheme::F32;
    rejects(&manifest, "doesn't encode dtype bf16");

    let mut manifest = sample();
    manifest.files[0].tensors[1].layout = Layout::OutIn;
    rejects(&manifest, "rank-2");

    let mut manifest = sample();
    manifest.alignment = 4096;
    rejects(&manifest, "alignment");
}

#[test]
fn checks_paths_and_names() {
    for bad in [
        "",
        "/abs.bin",
        "a//b.bin",
        "a/./b.bin",
        "../b.bin",
        "a/b c.bin",
        "furymodel.json",
        "part.json",
        "hashes/blake3.bin",
        "hashes",
    ] {
        assert!(check_path(bad).is_err(), "{bad:?}");
    }
    for good in ["layer.bin", "aux/chat_template.jinja", "a/B-1_2.x"] {
        check_path(good).unwrap();
    }
    // Reserved where case is ignored too, and as directories.
    for bad in [
        "hashes",
        "HASHES",
        "FURYMODEL.JSON",
        "Part.json",
        "HASHES/x.bin",
        "furymodel.json/x.bin",
        "part.json/x.bin",
    ] {
        assert!(check_path(bad).unwrap_err().contains("reserved"), "{bad:?}");
    }
    let good = ["a", "b", "c"].map(|letter| letter.repeat(255)).join("/")
        + &format!("/{}.bin", "d".repeat(251));
    assert_eq!(good.len(), 1023);
    check_path(&good).unwrap();
    assert!(
        check_path(&format!("{good}/e"))
            .unwrap_err()
            .contains("longer than 1024")
    );

    let mut manifest = sample();
    manifest.files[0].path = "layers/000/layer.bin".to_owned();
    rejects(
        &manifest,
        "two files have the path \"layers/000/layer.bin\"",
    );

    let mut manifest = sample();
    manifest.files[0].tensors[0].name = "layers.0.q".to_owned();
    rejects(&manifest, "two tensors are named \"layers.0.q\"");

    let mut manifest = sample();
    manifest.aux.push(AuxEntry {
        path: "aux/other.json".to_owned(),
        size: 1,
        role: "tokenizer".to_owned(),
    });
    rejects(&manifest, "have the role \"tokenizer\"");

    let mut manifest = sample();
    manifest.aux[0].role = String::new();
    rejects(&manifest, "has an empty role");

    let mut manifest = sample();
    manifest.files[0].tags.push("text".to_owned());
    rejects(&manifest, "repeated tag");

    let mut manifest = sample();
    manifest
        .extensions
        .insert("draft".to_owned(), serde_json::json!(1));
    rejects(&manifest, "isn't namespaced");

    let mut manifest = sample();
    manifest.provenance.sources.clear();
    rejects(&manifest, "provenance needs a source");

    // Every source's files are checked, not only the first source's.
    let mut manifest = sample();
    let mut second = manifest.provenance.sources[0].clone();
    second.files[0].sha2_256 = Some("0".repeat(63));
    manifest.provenance.sources.push(second);
    rejects(&manifest, "lowercase hex");
}

#[test]
fn checks_groups() {
    let mut manifest = sample();
    manifest.groups[0].path = "layers/{layer:03}/experts.bin".to_owned();
    rejects(&manifest, "doesn't use axis \"expert\"");

    let mut manifest = sample();
    manifest.groups[0].tensors[0].name = "layers.{layr}.down".to_owned();
    rejects(&manifest, "which isn't an axis");

    let mut manifest = sample();
    manifest.groups[0].path = "layers/{layer:3}/{expert}.bin".to_owned();
    rejects(&manifest, "isn't 0W");

    let mut manifest = sample();
    manifest.groups[0].path = "layers/{layer/{expert}.bin".to_owned();
    rejects(&manifest, "unclosed");

    let mut manifest = sample();
    manifest.groups[0].path = "layers/layer}/{expert}.bin".to_owned();
    rejects(&manifest, "unmatched");

    let mut manifest = sample();
    manifest.groups[1].axes = vec![Axis::listed("layer", vec![1, 0])];
    rejects(&manifest, "strictly increasing");

    let mut manifest = sample();
    manifest.groups[1].axes = vec![Axis {
        name: "layer".to_owned(),
        count: Some(2),
        values: Some(vec![0, 1]),
    }];
    rejects(&manifest, "either a count");

    let mut manifest = sample();
    manifest.groups[0].axes = vec![Axis::counted("layer", 512), Axis::counted("expert", 513)];
    rejects(&manifest, "more instances than a file table holds files");

    let mut manifest = sample();
    manifest.groups[1].name = "experts".to_owned();
    rejects(&manifest, "two groups");

    // A tensor name template uses every axis with more than one value; an
    // axis of one value it may leave out.
    let mut manifest = sample();
    manifest.groups[0].tensors[0].name = "layers.{layer}.down".to_owned();
    rejects(
        &manifest,
        "the tensor name template \"layers.{layer}.down\" doesn't use axis \"expert\", which \
         has more than one value",
    );
    let mut manifest = sample();
    manifest.groups[0].axes[1] = Axis::counted("expert", 1);
    manifest.groups[0].tensors[0].name = "layers.{layer}.down".to_owned();
    manifest.groups[0].tensors[1].name = "layers.{layer}.up".to_owned();
    parse(&manifest).unwrap();

    // Between two axes, a character other than a digit, in paths and names.
    for path in [
        "layers/{layer}{expert}.bin",
        "layers/{layer:03}7{expert}.bin",
    ] {
        let mut manifest = sample();
        manifest.groups[0].path = path.to_owned();
        rejects(&manifest, "two axes with nothing but digits between them");
    }
    let mut manifest = sample();
    manifest.groups[0].tensors[0].name = "layers.{layer}{expert}.down".to_owned();
    rejects(&manifest, "two axes with nothing but digits between them");
    let mut manifest = sample();
    manifest.groups[0].path = "layers/{layer}_{expert}.bin".to_owned();
    parse(&manifest).unwrap();

    // Zero padding below 3 digits, and wider values than the padding.
    let mut manifest = sample();
    manifest.groups[1].axes = vec![Axis::listed("layer", vec![7, 1234])];
    let table = parse(&manifest).unwrap();
    assert!(table.index_of("layers/007/layer.bin").is_some());
    assert!(table.index_of("layers/1234/layer.bin").is_some());
    assert!(table.tensor("layers.1234.q").is_some());
}

// Paths that no file system holds side by side: one where case is ignored,
// or a file that is another path's directory.
#[test]
fn checks_paths_against_each_other() {
    let aux = |path: &str, role: &str| AuxEntry {
        path: path.to_owned(),
        size: 1,
        role: role.to_owned(),
    };
    let mut manifest = sample();
    manifest.aux.push(aux("AUX/Tokenizer.json", "other"));
    rejects(&manifest, "name one file where case is ignored");

    let mut manifest = sample();
    manifest.aux.push(aux("global/embed.bin/x.txt", "other"));
    rejects(
        &manifest,
        "\"global/embed.bin\" is a file, but \"global/embed.bin/x.txt\" needs it as a directory",
    );

    let mut manifest = sample();
    manifest.aux.push(aux("Global/Embed.bin/x.txt", "other"));
    rejects(&manifest, "needs it as a directory");

    // However many paths sort between the two byte by byte: "-" and "."
    // come before "/".
    let mut manifest = sample();
    manifest.aux.push(aux("global/embed.bin-x", "x"));
    manifest.aux.push(aux("global/embed.bin.x", "y"));
    manifest.aux.push(aux("global/embed.bin/x.txt", "z"));
    rejects(
        &manifest,
        "\"global/embed.bin\" is a file, but \"global/embed.bin/x.txt\" needs it as a directory",
    );

    // A directory's name may be a file's elsewhere, and "a.b" sorts
    // between "a" and "a/b".
    let mut manifest = sample();
    manifest.aux.push(aux("global/embed.binx", "x"));
    manifest.aux.push(aux("aux/tokenizer.json.d", "y"));
    parse(&manifest).unwrap();
}

// Paths are checked against each other in time and memory in proportion
// to their bytes, however deep they are: here one group of 16,384 paths
// of 508 segments.
#[test]
fn checks_deep_paths_in_proportion_to_their_length() {
    let mut manifest = sample();
    manifest.groups = vec![GroupEntry {
        name: "deep".to_owned(),
        path: format!("{{a}}/{{b}}/{}y", "x/".repeat(505)),
        axes: vec![Axis::counted("a", 128), Axis::counted("b", 128)],
        size: 0,
        tags: vec![],
        tensors: vec![],
    }];
    let start = Instant::now();
    let table = parse(&manifest).unwrap();
    let elapsed = start.elapsed();
    assert_eq!(table.len(), 128 * 128 + 3);
    assert!(elapsed < Duration::from_secs(10), "took {elapsed:?}");
}

#[test]
fn rejects_keys_given_twice() {
    let json = String::from_utf8(sample().to_json()).unwrap();
    for twice in [
        json.replacen("\"config\": {", "\"config\": {\"dup\": 1, \"dup\": 2, ", 1),
        json.replacen('{', "{\"name\": \"other\", ", 1),
        // The same key, once escaped.
        json.replacen(
            "\"config\": {",
            "\"config\": {\"dup\": 1, \"d\\u0075p\": 2, ",
            1,
        ),
    ] {
        assert_ne!(twice, json);
        let error = Manifest::parse(twice.as_bytes()).unwrap_err();
        assert!(error.to_string().contains("twice"), "{error}");
    }
    let mut manifest = sample();
    manifest
        .extensions
        .insert("example.x".to_owned(), serde_json::json!({"a": [{"b": 1}]}));
    let json = String::from_utf8(manifest.to_json()).unwrap();
    let twice = json.replacen("\"b\": 1", "\"b\": 1, \"b\": 2", 1);
    assert_ne!(twice, json);
    let error = Manifest::parse(twice.as_bytes()).unwrap_err();
    assert!(error.to_string().contains("twice"), "{error}");
}

// The limits are checked before a group's instances are written out: a
// manifest of a few kilobytes describing millions of files fails at once.
#[test]
fn holds_the_file_table_to_its_limits() {
    let group = |name: &str, instances: u64, tensors: usize| GroupEntry {
        name: name.to_owned(),
        path: format!("{name}/{{e}}.bin"),
        axes: vec![Axis::counted("e", instances)],
        size: ALIGNMENT * tensors as u64,
        tags: vec![],
        tensors: (0..tensors)
            .map(|t| tensor(&format!("{name}.{{e}}.{t}"), &[64], ALIGNMENT * t as u64))
            .collect(),
    };
    let mut manifest = sample();
    manifest.groups = (0..8)
        .map(|g| group(&format!("g{g}"), 131_071, 4))
        .collect();
    assert!(manifest.to_json().len() < 16384);
    rejects(&manifest, "holds more than 262144 files");

    let mut manifest = sample();
    manifest.groups = vec![group("g", 200_000, 6)];
    rejects(&manifest, "hold more than 1048576 tensors");

    let mut manifest = sample();
    let mut long = group("g", 200_000, 1);
    let segment = "p".repeat(250);
    long.path = format!("{segment}/{segment}/{segment}/{{e}}.bin");
    long.tensors[0].name = format!("{}.{{e}}", "n".repeat(900));
    manifest.groups = vec![long];
    rejects(&manifest, "description takes more than 256 MiB");

    // Lengths as the longest instance renders them.
    let mut manifest = sample();
    let segment = "p".repeat(252);
    manifest.groups[1].path = format!("{segment}/{segment}/{segment}/{segment}/{{layer:09}}.bin");
    rejects(&manifest, "longer than 1024 bytes");
    let mut manifest = sample();
    manifest.groups[1].tensors[0].name = format!("{}.{{layer:09}}", "n".repeat(1015));
    rejects(&manifest, "names longer than 1024 bytes");
    let mut manifest = sample();
    manifest.files[0].tensors[0].name = "n".repeat(1025);
    rejects(&manifest, "longer than 1024 bytes");

    let mut manifest = sample();
    manifest.aux[0].size = u64::MAX;
    manifest.aux.push(AuxEntry {
        path: "aux/more.bin".to_owned(),
        size: 1,
        role: "more".to_owned(),
    });
    rejects(&manifest, "add up to more than 2^64 bytes");
}

// A template's lengths, computed without rendering, are its renderings'.
#[test]
fn template_lengths_without_rendering() {
    let axes = [
        Axis::counted("a", 1234),
        Axis::listed("b", vec![3, 99, 100, 12_345]),
    ];
    for template in ["x/{a}/{b:03}.bin", "{a:05}_{a}-{b}", "plain-{b:01}.{a:09}"] {
        let parsed = Template::parse(template, &axes).unwrap();
        let mut total = 0;
        let mut longest = 0;
        for a in 0..1234 {
            for b in [3, 99, 100, 12_345] {
                let len = parsed.render(&[a, b]).len() as u64;
                total += len;
                longest = longest.max(len);
            }
        }
        assert_eq!(parsed.total_len(&axes), total, "{template}");
        assert_eq!(parsed.max_len(&axes), longest, "{template}");
    }
}

// The file table of a manifest no one validated validates it first.
#[test]
fn builds_no_file_table_of_an_invalid_manifest() {
    let mut manifest = sample();
    manifest.groups[1].axes[0].values = Some(vec![]);
    let error = manifest.file_table(1000).unwrap_err();
    assert!(error.to_string().contains("strictly increasing"), "{error}");
}

// What every instance copies of its group counts in the description, and
// what it copies per instance is bounded: a manifest of a few kilobytes
// can't make a reader hold gigabytes.
#[test]
fn bounds_what_instances_copy() {
    let group = |instances: u64| GroupEntry {
        name: "g".to_owned(),
        path: "{e}".to_owned(),
        axes: vec![Axis::counted("e", instances)],
        size: ALIGNMENT,
        tags: vec![],
        tensors: vec![tensor("t{e}", &[64], 0)],
    };
    let mut manifest = sample();
    let mut ones = group(262_000);
    ones.tensors[0] = tensor("t{e}", &[1; 1000], 0);
    manifest.groups = vec![ones];
    rejects(&manifest, "has more than 16 dimensions");

    let mut manifest = sample();
    let mut named = group(262_000);
    named.name = "g".repeat(8000);
    named.tensors.clear();
    manifest.groups = vec![named];
    rejects(&manifest, "has a name longer than 64 bytes");

    // Every distinct tag of two bytes, 1,444 of them, in each of 23,000
    // instances.
    let letters: Vec<char> = ('a'..='z').chain('0'..='9').chain(['_', '-']).collect();
    let mut tagged = group(23_000);
    tagged.tags = letters
        .iter()
        .flat_map(|&a| letters.iter().map(move |&b| format!("{a}{b}")))
        .collect();
    assert_eq!(tagged.tags.len(), 1444);
    let mut manifest = sample();
    manifest.groups = vec![tagged];
    rejects(&manifest, "description takes more than 256 MiB");

    let mut manifest = sample();
    let mut wide = group(2);
    wide.axes = (0..17)
        .map(|axis| Axis::counted(format!("a{axis}"), if axis == 0 { 2 } else { 1 }))
        .collect();
    wide.path = (0..17).fold(String::new(), |mut path, axis| {
        write!(path, "{{a{axis}}}").unwrap();
        path
    });
    manifest.groups = vec![wide];
    rejects(&manifest, "or more than 16");
}

// The description as section 2.9 counts it, for a group of 12 instances.
#[test]
fn counts_the_description_as_specified() {
    let group = GroupEntry {
        name: "g".to_owned(),
        path: "x/{e}.bin".to_owned(),
        axes: vec![Axis::counted("e", 12)],
        size: ALIGNMENT,
        tags: vec!["t".to_owned()],
        tensors: vec![tensor("g.{e}", &[4], 0)],
    };
    let mut totals = Totals::new();
    let manifest = 128 + (MANIFEST_PATH.len() as u64 + 32);
    assert_eq!(totals.description, manifest);
    totals.add_group(&group).unwrap();
    // Each instance: its file, its path's 32, the group's name and tag,
    // its axis value, its tensor with its name's 32 and one dimension.
    let each = 128 + 32 + (1 + 32) + 8 + (1 + 32) + (128 + 32 + 8);
    // The paths x/0.bin to x/11.bin and the names g.0 to g.11.
    let rendered = (12 * 6 + 10 + 2 * 2) + (12 * 2 + 10 + 2 * 2);
    assert_eq!(totals.description, manifest + 12 * each + rendered);
    assert_eq!((totals.files, totals.tensors), (13, 12));
}
