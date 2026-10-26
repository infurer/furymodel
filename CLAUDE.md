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
  the content id is SHA2-512 based.
- Several roots per model, and delivery parts that a root may hold.

The v0.1 format of the same project ("modeldir") is research material: read
it for ideas, don't copy its conventions blindly.

## Commands

```sh
nix develop
nix fmt
nix flake check --all-systems --no-update-lock-file
```

`nix flake check` evaluates every system but builds only the current
system's checks. The per-commit gate builds them for all systems; on a Mac
the Linux ones go to its remote builder. Flakes only see files git tracks,
so `git add` new files first:

```sh
nix fmt && git diff --exit-code &&
  nix flake check --all-systems --no-update-lock-file &&
  for s in x86_64-linux aarch64-linux aarch64-darwin; do
    nix build --no-link --impure --expr \
      "builtins.attrValues (builtins.getFlake \"git+file://$PWD\").checks.$s"
  done
```

## Architecture

- `flake.nix`: flake-parts on nixos-26.05; systems x86_64-linux,
  aarch64-linux and aarch64-darwin.
- `nix/`: `checks.nix` (`reuse lint`), `devshell.nix` and `treefmt.nix`
  (`nix fmt` and `checks.treefmt`).
- `.gitlab-ci.yml`: GitLab CI on x86_64-linux runners. `publish-github`, on
  master, pushes master to the public GitHub repository.

## Conventions

- Every file has SPDX copyright and license comments
  (`2026 Mika Tammi`, `MIT OR Apache-2.0`). Files that can't hold comments are
  annotated in `REUSE.toml`.
- Every file type has a formatter in `nix/treefmt.nix`, enabled in the
  commit that adds the first file of that type.
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
- Every commit passes the per-commit gate (see Commands).
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
