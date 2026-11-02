<!--
SPDX-FileCopyrightText: 2026 Mika Tammi
SPDX-License-Identifier: MIT OR Apache-2.0
-->

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working
with code in this repository.

## Project overview

FuryModel v0.2: the specification and the Rust library of a model file
format laid out for streaming layers and MoE experts from disk through RAM
to the GPU. The format is generic: it knows no model architecture. Tensor
names and configuration schemas come from the runtime that defines the
architectures; this repository only stores, locates, verifies and reads
the bytes.

What the format promises, and this library implements:

- One file per (layer, expert) and per dense layer block; payloads have no
  headers and every tensor offset and file size is aligned to 16 KiB, so a
  file or tensor can be read with direct I/O straight into GPU-visible
  memory.
- A compact manifest: group templates describe repeated files (one entry for
  all experts), every tensor records its dtype, shape, layout and scheme, and
  unknown fields are errors unless they sit under `extensions`.
- Several hash tables per model, each in its own sidecar: fast checks per
  file and per 2 MiB read chunk (gxhash3-128, xxh3-128) so every chunk is
  verified as it lands, and cryptographic ones per file (sha2-512, sha3-512,
  blake3, sha2-256). Tables can be added later without touching payloads;
  the content id and the file table's fingerprint are SHA3-512 based.
- Several roots per model, and delivery parts that a root may hold.

The v0.1 format of the same project ("modeldir") is research material: read
it for ideas, don't copy its conventions blindly.

## Commands

```sh
nix develop                # rust-toolchain.toml's toolchain, cargo-nextest, reuse
nix build .#checks.aarch64-darwin.clippy   # warnings as errors
nix build .#checks.aarch64-darwin.nextest
nix build .#checks.aarch64-darwin.msrv     # nixpkgs' rustc and cargo
nix build .#furymodel-cli                 # its check: clippy, the tests
nix run .#furymodel-cli -- info ROOT...   # or verify, hash add, split, diff
nix fmt
nix flake check --all-systems --no-update-lock-file
```

Builds and tests go through Nix. `cargo build` inside `nix develop` works
too, with the dependencies from crates.io, as for a user of the crate
outside Nix; CI's `cargo` job keeps it working.

`nix flake check` evaluates every system but builds only the current
system's checks. The per-commit gate builds every system's checks and
packages; on a Mac the Linux ones go to its remote builder. Flakes only see
files git tracks, so `git add` new files first:

```sh
nix fmt && git diff --exit-code &&
  nix flake check --all-systems --no-update-lock-file &&
  for s in x86_64-linux aarch64-linux aarch64-darwin; do
    nix build --no-link --impure --expr \
      "let f = builtins.getFlake \"git+file://$PWD\";
       in builtins.attrValues (f.checks.$s // f.packages.$s)"
  done
```

### Releasing

```sh
nix run .#release -- 0.2.0-alpha.2     # Mika: the release commit and tag
nix run .#release -- --abort 0.2.0-alpha.2
nix run .#publish-releases -- --dry-run   # package and build as crates.io would
```

Versions go alpha.N, beta.N, rc.N, then the release, and change only in a
release commit, `<repo>: <version>`, the Rust convention: master keeps a
version until the next release commit, so a `[patch]` of the crates from
git still matches their dependents' requirements. `nix run .#release`
makes that commit on origin/master in a worktree of its own, checks that it
changes nothing but versions (instead of the per-commit gate, which master
has passed), signs it and its tag `v<version>`, and pushes both to GitLab
with a merge request set to merge when its pipeline passes. Once merged,
master's pipeline pushes the tag to GitHub and publishes the crates. The
scripts are the same in the project's other repositories; what is
particular to this one is in `nix/release.nix`.

## Architecture

- `crates/furymodel/SPEC.md`: the format's specification, normative, which
  the crate's documentation shows as its `spec` module. The code follows
  it; a change to the format changes it in the same commit, and bumps
  `format_version` where its section 9 says so.
- `flake.nix`: flake-parts on nixos-26.05; systems x86_64-linux,
  aarch64-linux and aarch64-darwin.
- `Cargo.toml`: the workspace, its shared package fields, dependencies and
  lints; `crates/furymodel`: the library, whose modules follow SPEC.md's
  sections (the crate documentation lists them), with the tests' fixture
  model in `src/testing.rs`; `crates/furymodel-cli`: the command line tool
  (info, verify, hash add, split, diff), tested end to end in `tests/`.
  Each crate's `README.md` is its page on crates.io, and its `LICENSE-MIT`
  and `LICENSE-APACHE` are links into `LICENSES/`, which cargo packages as
  the files themselves.
- `rust-toolchain.toml`: rust-overlay's stable as of `flake.lock`;
  `clippy.toml`: the disallowed macros and methods; `.cargo/config.toml`:
  AES instructions for the Linux targets, which gxhash needs at compile
  time (Apple's target has them). Built without them, the library leaves
  gxhash3-128 out and verifies with xxh3-128; built with them, it still
  uses gxhash3-128 only where the CPU has AES, as CPUID or the kernel's
  hwcaps say at run time.
- `nix/`: `rust.nix` (the toolchain and crane: one `buildDepsOnly` whose
  `cargoArtifacts` every Rust check reuses), `checks.nix` (clippy and
  rustdoc with warnings as errors, nextest, nextest again built without AES
  instructions, `msrv`: every target built with nixpkgs' own rustc and
  cargo, `reuse lint`), `package.nix` (`packages.furymodel-cli`, the
  default, whose check runs clippy with warnings as errors and nextest),
  `devshell.nix`, `treefmt.nix` (`nix fmt` and `checks.treefmt`), and
  `release.nix`: the `release` and `publish-releases` apps (`release.sh`,
  `publish-releases.sh`) and `checks.release`, which runs both against
  throwaway repositories and keys (`release-test.nix`).
- `.config/nextest.toml`: the leak timeout on macOS, and the `sandbox`
  profile that the checks and the package's check run.
- `.gitlab-ci.yml`: GitLab CI on x86_64-linux runners. `checks` runs
  `nix flake check --all-systems`, which builds x86_64-linux's checks;
  `packages` builds the packages of x86_64-linux and aarch64-linux, listed
  by hand, and so runs clippy and the tests on both; `cargo` runs `cargo build` inside `nix develop`, the build
  outside Nix; `publish-github`, on master once those passed, pushes master
  to the public GitHub repository; `release` then releases the release tags
  merged into master (`nix run .#publish-releases`): each must be signed
  with the release signing key and name its commit's version, and goes to
  GitHub, its crates to crates.io. Tags get no pipeline of their own.

## Conventions

- Every file has SPDX copyright and license comments
  (`2026 Mika Tammi`, `MIT OR Apache-2.0`). Files that can't hold comments are
  annotated in `REUSE.toml`.
- A test that needs what the Nix sandbox lacks, such as a GPU or the
  network, carries `outside_sandbox` in its name: nextest's `sandbox`
  profile, which every Nix build of the tests runs, leaves it out.
- Every file type has a formatter in `nix/treefmt.nix`, enabled in the
  commit that adds the first file of that type. Rust is formatted by the
  toolchain's rustfmt, TOML by taplo.
- Rust: edition 2024, MSRV 1.95 (nixos-26.05's rustc, which `checks.msrv`
  builds with; Cargo.toml's rust-version must stay its version), pedantic
  clippy with warnings as errors, documented public items. No `unsafe`
  without an explicit `allow` in the crate that needs it and a `SAFETY:`
  comment on every block. No `println!`/`eprintln!`/`dbg!`: output goes
  through `tracing` or an explicit writer. No `std::env::var*`:
  configuration comes from arguments and files.
- Nothing host-specific in the repository: no hostnames, absolute paths of
  anyone's machine, or names of private repositories and services. master
  is mirrored to a public GitHub repository in full.

## Git workflow

- One commit per reasonably sized feature. Docs (README, this file) change
  in the same commit as the feature they describe.
- Subjects in nixpkgs style, `<attr or area>: <summary>`. Short bodies: why,
  and facts that aren't obvious from the diff.
- Commit with `git commit -s`, and with an `Assisted-by:` trailer naming
  the AI assistant that helped, such as `Assisted-by: Claude Opus 5.5`,
  directly above the Signed-off-by. No other trailers.
- `flake.lock` changes only together with the input change that needs them.
- Every commit passes the per-commit gate (see Commands), but a release
  commit, which changes only versions and `nix run .#release` checks.
- Corrections to earlier commits go in as `git commit --fixup=<commit>` (or
  `--squash`) and are autosquashed before the merge request.
- v0.2 work happens on `initialv02` and reaches master through merge
  requests on the GitLab repository, which CI mirrors to GitHub. Never push
  to GitHub by hand. Mika signs the commits with GPG before the merge.

## Common mistakes to avoid

- `path:` flake inputs, or `flake.lock` updates riding along in unrelated
  commits.
- Committing `HANDOFF.md` or other session notes.
- Writing SPDX tag names followed by a colon in prose: `reuse lint` parses
  them anywhere in a file.
