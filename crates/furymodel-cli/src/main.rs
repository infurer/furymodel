// SPDX-FileCopyrightText: 2026 Mika Tammi
// SPDX-License-Identifier: MIT OR Apache-2.0

//! furymodel-cli: describes, verifies, extends, splits and compares
//! FuryModel models.
//!
//! Exits with 0 when all is well, 1 when files don't match their digests or
//! models differ, and 2 on errors.

use std::borrow::Cow;
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::io::{self, Write};
use std::num::NonZero;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use anyhow::{Context as _, Result, anyhow, bail};
use clap::{Args, Parser, Subcommand};
use furymodel::hash::Algorithm;
use furymodel::manifest::EntryKind;
use furymodel::{CacheMode, ContentId, Dtype, Model, Vouch, diff, part};

#[derive(Parser)]
#[command(
    name = "furymodel-cli",
    version,
    about = "Describe, verify, extend, split and compare FuryModel models.",
    after_help = "A model is given by its roots: the directories holding its files. \
                  Exits with 0 when all is well, 1 when files don't match their digests or \
                  models differ, 2 on errors."
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Describe a model: its files, tensors, tags, hash tables, provenance
    /// and content id.
    Info {
        /// Print JSON instead of text.
        #[arg(long)]
        json: bool,
        /// The model's roots.
        #[arg(required = true)]
        roots: Vec<PathBuf>,
    },
    /// Check a model's files against its hash tables.
    Verify {
        /// Check against every table present that this build can compute.
        #[arg(long, conflicts_with = "algorithms")]
        all: bool,
        /// Check against this algorithm's table; may be given several
        /// times. By default, the fastest fast check.
        #[arg(long = "algorithm", value_name = "ID")]
        algorithms: Vec<String>,
        #[command(flatten)]
        reading: Reading,
        /// The model's roots.
        #[arg(required = true)]
        roots: Vec<PathBuf>,
    },
    /// Manage a model's hash tables.
    #[command(subcommand)]
    Hash(HashCommand),
    /// Split a model into delivery parts, each within a size limit.
    Split {
        /// The most bytes a part holds, with a unit: B, KB, MB, GB, TB,
        /// KiB, MiB, GiB or TiB.
        #[arg(long, value_parser = parse_size)]
        max_part_size: u64,
        /// Where the parts go; it must not exist or be empty.
        #[arg(long)]
        out: PathBuf,
        /// The model's roots.
        #[arg(required = true)]
        roots: Vec<PathBuf>,
    },
    /// Compare two models by their manifests and hash tables.
    Diff {
        /// The old model's roots.
        #[arg(long, num_args = 1.., required = true)]
        old: Vec<PathBuf>,
        /// The new model's roots.
        #[arg(long, num_args = 1.., required = true)]
        new: Vec<PathBuf>,
    },
}

#[derive(Subcommand)]
enum HashCommand {
    /// Compute the table of an algorithm the model lacks from its files.
    Add {
        /// The algorithm's id, such as sha3-512.
        algorithm: String,
        /// The root the table goes into; by default the first.
        #[arg(long)]
        into: Option<PathBuf>,
        /// For a model with no table to check its files against: its content
        /// id, from a source trusted for it, which the sha3-512 table
        /// computed from the files must give.
        #[arg(long, value_name = "CONTENT_ID", conflicts_with = "trust_files")]
        expect: Option<String>,
        /// For a model with no hash table at all: vouch for its files as they
        /// are, with nothing to check them against.
        #[arg(long)]
        trust_files: bool,
        #[command(flatten)]
        reading: Reading,
        /// The model's roots.
        #[arg(required = true)]
        roots: Vec<PathBuf>,
    },
}

#[derive(Args)]
struct Reading {
    /// Read through the page cache rather than around it.
    #[arg(long)]
    cached: bool,
    /// The number of threads reading and hashing files.
    #[arg(long, default_value_t = default_threads())]
    threads: usize,
}

impl Reading {
    fn mode(&self) -> CacheMode {
        if self.cached {
            CacheMode::Use
        } else {
            CacheMode::Bypass
        }
    }
}

fn default_threads() -> usize {
    std::thread::available_parallelism().map_or(4, NonZero::get)
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let mut out = io::stdout().lock();
    match run(cli.command, &mut out) {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::from(1),
        Err(error) => {
            // Nothing is left to report a failed write of the error to.
            let _ = writeln!(
                io::stderr(),
                "furymodel-cli: {}",
                shown(&format!("{error:#}"))
            );
            ExitCode::from(2)
        }
    }
}

fn run(command: Command, out: &mut dyn Write) -> Result<bool> {
    match command {
        Command::Info { json, roots } => {
            let model = open(&roots)?;
            if json {
                serde_json::to_writer_pretty(&mut *out, &summary(&model))?;
                writeln!(out)?;
            } else {
                info(&model, out)?;
            }
            Ok(true)
        }
        Command::Verify {
            all,
            algorithms,
            reading,
            roots,
        } => verify(&open(&roots)?, all, &algorithms, &reading, out),
        Command::Hash(command) => hash(command, out),
        Command::Split {
            max_part_size,
            out: directory,
            roots,
        } => {
            let model = open(&roots)?;
            let parts = part::split(&model, max_part_size, &directory)?;
            for part in &parts {
                writeln!(out, "{}", part.display())?;
            }
            Ok(true)
        }
        Command::Diff { old, new } => {
            let found = diff::diff(&open(&old)?, &open(&new)?);
            print_diff(&found, out)?;
            Ok(found.is_empty())
        }
    }
}

/// `text` as it is safe to show on a terminal: a model's strings could carry
/// control characters that move the cursor, recolor the screen or reorder
/// text, so those are shown escaped.
fn shown(text: &str) -> Cow<'_, str> {
    let hidden = |c: char| {
        c.is_control()
            || matches!(c, '\u{200e}' | '\u{200f}' | '\u{202a}'..='\u{202e}' | '\u{2066}'..='\u{2069}')
    };
    if text.chars().any(hidden) {
        text.chars()
            .map(|c| {
                if hidden(c) {
                    c.escape_unicode().to_string()
                } else {
                    c.to_string()
                }
            })
            .collect::<String>()
            .into()
    } else {
        text.into()
    }
}

// Hash tables: adding one.
fn hash(command: HashCommand, out: &mut dyn Write) -> Result<bool> {
    let HashCommand::Add {
        algorithm,
        into,
        expect,
        trust_files,
        reading,
        roots,
    } = command;
    let algorithm = algorithm_of(&algorithm)?;
    let vouch = match (expect, trust_files) {
        (Some(id), _) => Some(Vouch::ContentId(ContentId::from_hex(&id).ok_or_else(
            || anyhow!("{id:?} isn't a content id, 128 lowercase hex digits"),
        )?)),
        (None, true) => Some(Vouch::Files),
        (None, false) => None,
    };
    let into = match into {
        None => roots[0].clone(),
        Some(into) => root_named(&roots, &into)?,
    };
    let start = Instant::now();
    let (path, model) = match vouch {
        None => {
            let model = open(&roots).map_err(with_first_table_hint)?;
            let path =
                model.add_table(algorithm, &into, reading.mode(), reading.threads, &|_| {})?;
            (path, model)
        }
        Some(vouch) => {
            if Model::open(&roots).is_ok() {
                bail!(
                    "the model has a table to check its files against: add tables to it \
                         without --expect or --trust-files"
                );
            }
            let path = Model::add_first_table(
                &roots,
                &into,
                algorithm,
                vouch,
                reading.mode(),
                reading.threads,
                &|_| {},
            )?;
            // It opens now, by the table it has.
            (path, open(&roots)?)
        }
    };
    writeln!(
        out,
        "wrote {}, {} files hashed in {:.1} s",
        path.display(),
        model.table().len(),
        start.elapsed().as_secs_f64()
    )?;
    if !model.parts().is_empty() {
        writeln!(
            out,
            "every part holds every table: copy {} into the other parts too",
            path.display()
        )?;
    }
    Ok(true)
}

fn open(roots: &[PathBuf]) -> Result<Model> {
    Model::open(roots).with_context(|| {
        let roots: Vec<String> = roots
            .iter()
            .map(|root| root.display().to_string())
            .collect();
        format!("opening the model of {}", roots.join(", "))
    })
}

fn algorithm_of(id: &str) -> Result<Algorithm> {
    Algorithm::from_id(id).ok_or_else(|| {
        let ids: Vec<&str> = Algorithm::ALL
            .iter()
            .map(|algorithm| algorithm.id())
            .collect();
        anyhow!("{id:?} isn't an algorithm; they are {}", ids.join(", "))
    })
}

// A model with no table to check its files against: how it gets one.
fn with_first_table_hint(error: anyhow::Error) -> anyhow::Error {
    if matches!(
        error.downcast_ref::<furymodel::Error>(),
        Some(furymodel::Error::Unverifiable(_))
    ) {
        anyhow!(
            "{error:#}; to add its first table, give the model's content id from a source you \
             trust with --expect, or, if it has no hash table at all, vouch for its files as \
             they are with --trust-files"
        )
    } else {
        error
    }
}

// The root among `roots` that is the same directory as `into`.
fn root_named(roots: &[PathBuf], into: &Path) -> Result<PathBuf> {
    let wanted = into
        .canonicalize()
        .with_context(|| format!("{}", into.display()))?;
    roots
        .iter()
        .find(|root| root.canonicalize().is_ok_and(|root| root == wanted))
        .cloned()
        .ok_or_else(|| anyhow!("{} isn't one of the model's roots", into.display()))
}

fn verify(
    model: &Model,
    all: bool,
    ids: &[String],
    reading: &Reading,
    out: &mut dyn Write,
) -> Result<bool> {
    let algorithms: Vec<Algorithm> = if all {
        let mut algorithms = Vec::new();
        for table in model.hash_tables() {
            if table.algorithm().is_supported() {
                algorithms.push(table.algorithm());
            } else {
                writeln!(
                    io::stderr(),
                    "furymodel-cli: skipping {}, which can't be computed here: gxhash3-128 \
                     needs AES instructions, as target features of the build and in the CPU",
                    table.algorithm()
                )?;
            }
        }
        algorithms
    } else if ids.is_empty() {
        vec![model.verifier()?]
    } else {
        ids.iter()
            .map(|id| algorithm_of(id))
            .collect::<Result<_>>()?
    };
    let bytes = AtomicU64::new(0);
    let start = Instant::now();
    let mismatches = model.verify(&algorithms, reading.mode(), reading.threads, &|n| {
        bytes.fetch_add(n, Ordering::Relaxed);
    })?;
    let seconds = start.elapsed().as_secs_f64();
    for mismatch in &mismatches {
        let chunk = mismatch
            .chunk
            .map_or_else(String::new, |chunk| format!(", chunk {chunk}"));
        writeln!(
            out,
            "mismatch: {}{chunk} against {}",
            shown(&mismatch.path),
            mismatch.algorithm
        )?;
    }
    let bytes = bytes.into_inner();
    let names: Vec<&str> = algorithms.iter().map(|algorithm| algorithm.id()).collect();
    writeln!(
        out,
        "{} files of {} read in {seconds:.1} s, {}/s, against {}: {}",
        model.table().len(),
        size(bytes),
        size(rate(bytes, seconds)),
        names.join(", "),
        if mismatches.is_empty() {
            "all match".to_owned()
        } else {
            format!("{} mismatches", mismatches.len())
        }
    )?;
    if let (true, Some(id)) = (
        mismatches.is_empty() && algorithms.contains(&Algorithm::Sha3),
        model.content_id(),
    ) {
        writeln!(
            out,
            "content id {id}: the files', as the sha3-512 table matches them; authentic if a \
             source you trust names it"
        )?;
    }
    Ok(mismatches.is_empty())
}

#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "a rate to show, not to compute with"
)]
fn rate(bytes: u64, seconds: f64) -> u64 {
    if seconds > 0.0 {
        (bytes as f64 / seconds) as u64
    } else {
        0
    }
}

/// A size in binary units, to one decimal.
#[allow(
    clippy::cast_precision_loss,
    reason = "a size to show, not to compute with"
)]
fn size(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["KiB", "MiB", "GiB", "TiB", "PiB"];
    if bytes < 1024 {
        return format!("{bytes} B");
    }
    let mut value = bytes as f64 / 1024.0;
    let mut unit = 0;
    while value >= 1024.0 && unit + 1 < UNITS.len() {
        value /= 1024.0;
        unit += 1;
    }
    format!("{value:.1} {}", UNITS[unit])
}

/// Parses a size with a unit, such as `35GB` or `24GiB`.
fn parse_size(text: &str) -> Result<u64, String> {
    let digits = text
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(text.len());
    let (number, unit) = text.split_at(digits);
    let number: u64 = number
        .parse()
        .map_err(|_| format!("{text:?} isn't a whole number with a unit"))?;
    let factor: u64 = match unit.trim() {
        "B" => 1,
        "KB" => 1_000,
        "MB" => 1_000_000,
        "GB" => 1_000_000_000,
        "TB" => 1_000_000_000_000,
        "KiB" => 1 << 10,
        "MiB" => 1 << 20,
        "GiB" => 1 << 30,
        "TiB" => 1 << 40,
        _ => {
            return Err(format!(
                "{text:?} needs a unit: B, KB, MB, GB, TB, KiB, MiB, GiB or TiB"
            ));
        }
    };
    number
        .checked_mul(factor)
        .ok_or_else(|| format!("{text:?} is too large"))
}

/// Counts and bytes, by key.
type Histogram<K> = BTreeMap<K, (u64, u64)>;

// Counts and bytes of the model's tensors by dtype, and of its payload
// files by tag.
fn histograms(model: &Model) -> (Histogram<Dtype>, Histogram<&str>) {
    let mut dtypes: Histogram<Dtype> = BTreeMap::new();
    for (_, tensor) in model.table().tensors() {
        let entry = dtypes.entry(tensor.dtype).or_default();
        *entry = (entry.0 + 1, entry.1 + tensor.size);
    }
    let mut tags: Histogram<&str> = BTreeMap::new();
    for entry in model.table().entries() {
        if let EntryKind::Payload(payload) = &entry.kind {
            for tag in &payload.tags {
                let counts = tags.entry(tag.as_str()).or_default();
                *counts = (counts.0 + 1, counts.1 + entry.size);
            }
        }
    }
    (dtypes, tags)
}

fn count(n: u64, one: &str, many: &str) -> String {
    format!("{n} {}", if n == 1 { one } else { many })
}

fn info(model: &Model, out: &mut dyn Write) -> io::Result<()> {
    let manifest = model.manifest();
    let table = model.table();
    let mut row = |key: &str, value: &str| writeln!(out, "{key:<12} {}", shown(value));
    row("model", &manifest.name)?;
    row(
        "arch",
        &format!(
            "{}, schema version {}",
            manifest.arch, manifest.schema_version
        ),
    )?;
    row("format", &format!("FuryModel {}", manifest.format_version))?;
    row(
        "content id",
        &model.content_id().map_or_else(
            || "unknown: the model has no sha3-512 table".to_owned(),
            |id| format!("{id}, as the sha3-512 table claims"),
        ),
    )?;
    let payloads = table
        .entries()
        .iter()
        .filter(|entry| matches!(entry.kind, EntryKind::Payload(_)))
        .count() as u64;
    row(
        "files",
        &format!(
            "{} of {}: {} and {}, and the manifest",
            table.len(),
            size(table.total_size()),
            count(payloads, "payload file", "payload files"),
            count(
                manifest.aux.len() as u64,
                "auxiliary file",
                "auxiliary files"
            ),
        ),
    )?;
    info_contents(model, out)?;
    info_storage(model, out)?;
    info_provenance(model, out)
}

fn info_contents(model: &Model, out: &mut dyn Write) -> io::Result<()> {
    let manifest = model.manifest();
    let table = model.table();
    let mut row = |key: &str, value: &str| writeln!(out, "{key:<12} {}", shown(value));
    let (dtypes, tags) = histograms(model);
    let tensors: Vec<String> = dtypes
        .iter()
        .map(|(dtype, (n, bytes))| format!("{dtype} {n} of {}", size(*bytes)))
        .collect();
    row(
        "tensors",
        &format!(
            "{}: {}",
            table.tensors().count(),
            if tensors.is_empty() {
                "none".to_owned()
            } else {
                tensors.join(", ")
            }
        ),
    )?;
    let tags: Vec<String> = tags
        .iter()
        .map(|(tag, (n, bytes))| {
            format!("{tag} {} of {}", count(*n, "file", "files"), size(*bytes))
        })
        .collect();
    row(
        "tags",
        &if tags.is_empty() {
            "none".to_owned()
        } else {
            tags.join(", ")
        },
    )?;
    for group in &manifest.groups {
        let axes: Vec<String> = group
            .axes
            .iter()
            .map(|axis| format!("{} {}", axis.value_count(), axis.name))
            .collect();
        let files: u64 = group
            .axes
            .iter()
            .map(furymodel::manifest::Axis::value_count)
            .product();
        row(
            "group",
            &format!(
                "{}: {} ({}) of {} each",
                group.name,
                count(files, "file", "files"),
                axes.join(" × "),
                size(group.size)
            ),
        )?;
    }
    Ok(())
}

fn info_storage(model: &Model, out: &mut dyn Write) -> io::Result<()> {
    let table = model.table();
    let mut row = |key: &str, value: &str| writeln!(out, "{key:<12} {}", shown(value));
    let names: Vec<&str> = model
        .hash_tables()
        .iter()
        .map(|table| table.algorithm().id())
        .collect();
    row(
        "hash tables",
        &format!(
            "{}; {}",
            if names.is_empty() {
                "none".to_owned()
            } else {
                names.join(", ")
            },
            model.verifier().map_or_else(
                |_| "none this build verifies chunks with".to_owned(),
                |algorithm| format!("chunks verified with {algorithm}"),
            )
        ),
    )?;
    for root in model.roots() {
        let files = (0..table.len())
            .filter(|&file| model.path(file).starts_with(root))
            .count() as u64;
        row(
            "root",
            &format!("{} ({})", root.display(), count(files, "file", "files")),
        )?;
    }
    if let Some(first) = model.parts().first() {
        let present: Vec<String> = model
            .parts()
            .iter()
            .map(|part| part.part.to_string())
            .collect();
        row(
            "parts",
            &format!("{} of {} present", present.join(", "), first.parts),
        )?;
    }
    Ok(())
}

fn info_provenance(model: &Model, out: &mut dyn Write) -> io::Result<()> {
    let manifest = model.manifest();
    let mut row = |key: &str, value: &str| writeln!(out, "{key:<12} {}", shown(value));
    for source in &manifest.provenance.sources {
        let mut origin = source.kind.clone();
        if let Some(repository) = &source.repository {
            write!(origin, " from {repository}").expect("writing to a String");
        }
        if let Some(revision) = &source.revision {
            write!(origin, " at {revision}").expect("writing to a String");
        }
        row(
            "source",
            &format!(
                "{origin}, {}",
                count(source.files.len() as u64, "file", "files")
            ),
        )?;
    }
    let tool = &manifest.provenance.tool;
    row("tool", &format!("{} {}", tool.name, tool.version))?;
    for transform in &manifest.provenance.transforms {
        row("transform", transform)?;
    }
    for note in &manifest.provenance.notes {
        row("note", note)?;
    }
    Ok(())
}

fn summary(model: &Model) -> serde_json::Value {
    let manifest = model.manifest();
    let table = model.table();
    let (dtypes, tags) = histograms(model);
    serde_json::json!({
        "name": manifest.name,
        "arch": manifest.arch,
        "schema_version": manifest.schema_version,
        "format_version": manifest.format_version,
        "content_id": model.content_id().map(|id| id.to_string()),
        "files": table.len(),
        "bytes": table.total_size(),
        "dtypes": dtypes
            .iter()
            .map(|(dtype, (tensors, bytes))| {
                (
                    dtype.id().to_owned(),
                    serde_json::json!({"tensors": tensors, "bytes": bytes}),
                )
            })
            .collect::<serde_json::Map<_, _>>(),
        "tags": tags
            .iter()
            .map(|(tag, (files, bytes))| {
                ((*tag).to_owned(), serde_json::json!({"files": files, "bytes": bytes}))
            })
            .collect::<serde_json::Map<_, _>>(),
        "hash_tables": model
            .hash_tables()
            .iter()
            .map(|table| table.algorithm().id())
            .collect::<Vec<_>>(),
        "verifier": model.verifier().ok().map(Algorithm::id),
        "roots": model.roots(),
        "parts": model.parts().iter().map(|part| part.part).collect::<Vec<_>>(),
        "provenance": manifest.provenance,
    })
}

fn print_diff(found: &diff::Diff, out: &mut dyn Write) -> io::Result<()> {
    for field in &found.fields {
        writeln!(out, "field differs: {}", shown(field))?;
    }
    for path in &found.removed {
        writeln!(out, "removed: {}", shown(path))?;
    }
    for path in &found.added {
        writeln!(out, "added: {}", shown(path))?;
    }
    for path in &found.changed {
        writeln!(out, "changed: {}", shown(path))?;
    }
    for path in &found.relabeled {
        writeln!(out, "tags or role differ: {}", shown(path))?;
    }
    for path in &found.uncompared {
        writeln!(out, "not compared, no common hash table: {}", shown(path))?;
    }
    for name in &found.removed_tensors {
        writeln!(out, "tensor removed: {}", shown(name))?;
    }
    for name in &found.added_tensors {
        writeln!(out, "tensor added: {}", shown(name))?;
    }
    for (name, old, new) in &found.changed_tensors {
        writeln!(
            out,
            "tensor changed: {}: {} -> {}",
            shown(name),
            shown(old),
            shown(new)
        )?;
    }
    let compared = found.algorithm.map_or_else(
        || "no common hash table".to_owned(),
        |algorithm| format!("compared with {algorithm}"),
    );
    if found.is_empty() {
        writeln!(out, "the models are the same, {compared}")
    } else {
        writeln!(out, "the models differ, {compared}")
    }
}

#[cfg(test)]
mod tests {
    use super::{parse_size, size};

    #[test]
    fn sizes() {
        assert_eq!(parse_size("35GB"), Ok(35_000_000_000));
        assert_eq!(parse_size("24GiB"), Ok(24 << 30));
        assert_eq!(parse_size("512B"), Ok(512));
        assert!(parse_size("35").is_err());
        assert!(parse_size("35G").is_err());
        assert!(parse_size("1.5GiB").is_err());
        assert!(parse_size("99999999999TiB").is_err());
        assert_eq!(size(512), "512 B");
        assert_eq!(size(11_894_784), "11.3 MiB");
        assert_eq!(size(48 << 30), "48.0 GiB");
    }
}
