# SPDX-FileCopyrightText: 2026 Mika Tammi
# SPDX-License-Identifier: MIT OR Apache-2.0
#
# The Rust toolchain from rust-overlay (rust-toolchain.toml) and crane. One
# buildDepsOnly produces the dependencies' cargoArtifacts, which every Rust
# check reuses, so a source change never rebuilds them. A second set, built
# with nixpkgs' own rustc and cargo, serves the check of the MSRV
# (checks.nix).
{inputs, ...}: {
  perSystem = {system, ...}: let
    pkgs = import inputs.nixpkgs {
      inherit system;
      overlays = [inputs.rust-overlay.overlays.default];
    };
    rustToolchain = pkgs.rust-bin.fromRustupToolchainFile ../rust-toolchain.toml;
    craneLib = (inputs.crane.mkLib pkgs).overrideToolchain rustToolchain;
    # nixpkgs' own rustc and cargo, which a nixpkgs build of the crates
    # uses: Cargo.toml's rust-version is theirs, or the check would test
    # another compiler than the MSRV.
    msrv = (builtins.fromTOML (builtins.readFile ../Cargo.toml)).workspace.package.rust-version;
    craneLibMsrv =
      if msrv == pkgs.lib.versions.majorMinor pkgs.rustc.version
      then inputs.crane.mkLib pkgs
      else throw "Cargo.toml's rust-version is ${msrv}, but nixpkgs' rustc is ${pkgs.rustc.version}: bring one to the other";
    # The Rust sources, and the specification the crate's documentation
    # includes.
    src = pkgs.lib.cleanSourceWith {
      src = ../.;
      name = "source";
      filter = path: type:
        craneLib.filterCargoSources path type
        || pkgs.lib.hasSuffix "/crates/furymodel/SPEC.md" path;
    };
    commonArgs = {
      inherit src;
      pname = "furymodel";
      version = "0.2.0";
      strictDeps = true;
    };
  in {
    _module.args = {
      inherit pkgs craneLib craneLibMsrv commonArgs rustToolchain;
      cargoArtifacts = craneLib.buildDepsOnly commonArgs;
      cargoArtifactsMsrv = craneLibMsrv.buildDepsOnly commonArgs;
    };
  };
}
