# SPDX-FileCopyrightText: 2026 Mika Tammi
# SPDX-License-Identifier: MIT OR Apache-2.0
#
# `nix develop`: the Rust toolchain of rust-toolchain.toml, everything the
# checks build with, and the tools the per-commit gate uses.
{
  perSystem = {
    config,
    craneLib,
    pkgs,
    ...
  }: {
    devShells.default = craneLib.devShell {
      inherit (config) checks;
      packages = [
        config.treefmt.build.wrapper
        pkgs.cargo-nextest
        pkgs.reuse
      ];
    };
  };
}
