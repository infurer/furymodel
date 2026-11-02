# SPDX-FileCopyrightText: 2026 Mika Tammi
# SPDX-License-Identifier: MIT OR Apache-2.0
#
# `nix fmt` and `checks.treefmt`. Every file type in the repository gets a
# formatter here in the same commit that adds the first file of that type.
{inputs, ...}: {
  imports = [
    inputs.treefmt-nix.flakeModule
  ];

  perSystem = {rustToolchain, ...}: {
    treefmt = {
      projectRootFile = "flake.nix";

      programs = {
        alejandra.enable = true;
        deadnix.enable = true;
        shfmt.enable = true;
        statix.enable = true;
        taplo.enable = true;
        # The toolchain's own rustfmt, so `nix fmt` and `cargo fmt` agree.
        rustfmt = {
          enable = true;
          package = rustToolchain;
          edition = "2024";
        };
        mdformat = {
          enable = true;
          # GitHub-flavoured Markdown: plain mdformat would mangle tables.
          plugins = ps: [ps.mdformat-gfm];
        };
        yamlfmt.enable = true;
      };

      # Files without a formatter.
      settings.global.excludes = [
        ".gitignore"
        "LICENSES/*"
        "Cargo.lock"
        "flake.lock"
      ];
    };
  };
}
