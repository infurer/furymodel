// SPDX-FileCopyrightText: 2026 Mika Tammi
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Fixtures of the tests: a tiny model with every kind of entry.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use crate::hash::{Algorithm, FileHasher, HashTable};
use crate::manifest::{
    Axis, HASHES_DIR, MANIFEST_PATH, Manifest, Provenance, Source, SourceFile, Tool,
};
use crate::registry::{Dtype, Layout, Scheme};
use crate::writer::{GroupId, GroupSpec, ModelInfo, TensorData, TensorSpec, Writer, Written};

/// The algorithms that can be computed here, so that the tests run
/// without AES instructions too.
pub fn algorithms() -> Vec<Algorithm> {
    Algorithm::ALL
        .into_iter()
        .filter(|algorithm| algorithm.is_supported())
        .collect()
}

/// The tiny model's description.
pub fn info() -> ModelInfo {
    ModelInfo {
        name: "tiny".to_owned(),
        arch: "tinyarch".to_owned(),
        schema_version: 1,
        config: serde_json::json!({"hidden": 32, "layers": 2, "eps": 1e-6}),
        provenance: Provenance {
            source: Source {
                kind: "synthetic".to_owned(),
                repository: None,
                revision: None,
                files: vec![SourceFile {
                    path: "tiny.safetensors".to_owned(),
                    size: 1234,
                    sha2_256: None,
                }],
            },
            tool: Tool {
                name: "furymodel tests".to_owned(),
                version: "0".to_owned(),
            },
            transforms: vec!["experts split per (layer, expert)".to_owned()],
            notes: vec![],
        },
        extensions: BTreeMap::new(),
    }
}

/// Deterministic bytes for the tensor `name`.
pub fn bytes(name: &str, len: u64) -> Vec<u8> {
    let seed = name.bytes().fold(7u64, |seed, byte| {
        seed.wrapping_mul(31).wrapping_add(u64::from(byte))
    });
    (0..len)
        .map(|i| u8::try_from((i.wrapping_mul(seed) ^ (i >> 7)) % 251).expect("below 251"))
        .collect()
}

const EXPERTS: u64 = 3;
const LAYERS: u64 = 2;
/// The size of the big tensor: more than one chunk.
pub const BIG: u64 = 2 * 1024 * 1024 + 20_000;

/// One file's addition to the tiny model.
type Job = Box<dyn Fn(&Writer) + Send + Sync>;

fn add_experts(writer: &Writer) -> GroupId {
    let tensor = |name: &str, shape: Vec<u64>| TensorSpec {
        name: name.to_owned(),
        dtype: Dtype::Bf16,
        shape,
        layout: Layout::OutIn,
        scheme: Scheme::Bf16,
    };
    writer
        .add_group(GroupSpec {
            name: "experts".to_owned(),
            path: "layers/{layer:02}/experts/{expert:02}.bin".to_owned(),
            axes: vec![
                Axis::counted("layer", LAYERS),
                Axis::counted("expert", EXPERTS),
            ],
            tags: vec!["text".to_owned()],
            tensors: vec![
                tensor("layers.{layer}.experts.{expert}.down", vec![32, 16]),
                tensor("layers.{layer}.experts.{expert}.up", vec![32, 32]),
            ],
        })
        .unwrap()
}

fn add_embed(writer: &Writer) {
    let embed = bytes("embed", 64 * 32 * 2);
    let norm = bytes("norm", 32 * 4);
    let tensor = |name, dtype, shape, scheme, bytes| TensorData {
        name,
        dtype,
        shape,
        layout: Layout::RowMajor,
        scheme,
        bytes,
    };
    writer
        .add_file(
            "global/embed.bin",
            &["text"],
            &[
                tensor("embed", Dtype::Bf16, &[64, 32], Scheme::Bf16, &embed),
                tensor("norm", Dtype::F32, &[32], Scheme::F32, &norm),
            ],
        )
        .unwrap();
}

fn add_big(writer: &Writer) {
    let big = bytes("big", BIG);
    writer
        .add_file(
            "global/big.bin",
            &["vision"],
            &[TensorData {
                name: "big",
                dtype: Dtype::U8,
                shape: &[BIG],
                layout: Layout::RowMajor,
                scheme: Scheme::U8,
                bytes: &big,
            }],
        )
        .unwrap();
}

fn jobs(experts: GroupId) -> Vec<Job> {
    let mut jobs: Vec<Job> = vec![
        Box::new(add_embed),
        Box::new(add_big),
        Box::new(|writer| {
            writer
                .add_aux(
                    "aux/tokenizer.json",
                    "tokenizer",
                    b"{\"vocab\": [\"a\", \"b\"]}\n",
                )
                .unwrap();
        }),
    ];
    for layer in 0..LAYERS {
        for expert in 0..EXPERTS {
            jobs.push(Box::new(move |writer| {
                let down = bytes(&format!("layers.{layer}.experts.{expert}.down"), 1024);
                let up = bytes(&format!("layers.{layer}.experts.{expert}.up"), 2048);
                writer
                    .add_instance(experts, &[layer, expert], &[&down, &up])
                    .unwrap();
            }));
        }
    }
    jobs
}

/// Writes the tiny model into `root`; `reversed` adds its files in the
/// other order, from several threads.
pub fn write_tiny(root: &Path, reversed: bool) -> Written {
    let writer = Writer::with_algorithms(root, info(), &algorithms()).unwrap();
    let mut jobs = jobs(add_experts(&writer));
    if reversed {
        jobs.reverse();
        let jobs = &jobs;
        let writer = &writer;
        std::thread::scope(|scope| {
            for thread in 0..4 {
                scope.spawn(move || {
                    for job in jobs.iter().skip(thread).step_by(4) {
                        job(writer);
                    }
                });
            }
        });
    } else {
        for job in &jobs {
            job(&writer);
        }
    }
    let written = writer.finish().unwrap();
    assert!(written.content_id.is_some());
    assert_eq!(
        Algorithm::Gxhash3.is_supported(),
        algorithms().contains(&Algorithm::Gxhash3)
    );
    written
}

/// Replaces the hash tables of the model in `root` with tables of
/// `algorithms` computed from its files as they are now, for tests that
/// change a model's files behind its back.
pub fn rehash(root: &Path, algorithms: &[Algorithm]) {
    let json = fs::read(root.join(MANIFEST_PATH)).unwrap();
    let (_, table) = Manifest::parse(&json).unwrap();
    let _ = fs::remove_dir_all(root.join(HASHES_DIR));
    fs::create_dir_all(root.join(HASHES_DIR)).unwrap();
    for &algorithm in algorithms {
        let digests = table
            .entries()
            .iter()
            .map(|entry| {
                let mut hasher = FileHasher::new(&[algorithm]).unwrap();
                hasher.update(&fs::read(root.join(&entry.path)).unwrap());
                hasher.finish().remove(0)
            })
            .collect();
        let encoded = HashTable::new(algorithm, &table, digests).unwrap().encode();
        fs::write(root.join(algorithm.table_path()), encoded).unwrap();
    }
}
