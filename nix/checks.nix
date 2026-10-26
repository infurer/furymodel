# SPDX-FileCopyrightText: 2026 Mika Tammi
# SPDX-License-Identifier: MIT OR Apache-2.0
#
# Repository-wide checks beyond treefmt's own.
{self, ...}: {
  perSystem = {
    cargoArtifacts,
    cargoArtifactsMsrv,
    commonArgs,
    craneLib,
    craneLibMsrv,
    pkgs,
    ...
  }: {
    checks = {
      # Pedantic clippy over every target, with warnings as errors.
      clippy = craneLib.cargoClippy (commonArgs
        // {
          inherit cargoArtifacts;
          cargoClippyExtraArgs = "--workspace --all-targets -- --deny warnings";
        });

      # The tests.
      nextest = craneLib.cargoNextest (commonArgs
        // {
          inherit cargoArtifacts;
          cargoNextestExtraArgs = "--workspace --no-tests=warn";
        });

      # Every target, the tests included, built with nixpkgs' own rustc
      # and cargo: the MSRV holds.
      msrv = craneLibMsrv.cargoBuild (commonArgs
        // {
          cargoArtifacts = cargoArtifactsMsrv;
          cargoExtraArgs = "--locked --workspace --all-targets";
          doInstallCargoArtifacts = false;
        });

      # The API docs, with rustdoc's warnings as errors.
      doc = craneLib.cargoDoc (commonArgs
        // {
          inherit cargoArtifacts;
          cargoDocExtraArgs = "--workspace --no-deps";
          RUSTDOCFLAGS = "--deny warnings";
        });

      # Every file carries SPDX copyright and license information.
      reuse =
        pkgs.runCommand "reuse-lint" {
          nativeBuildInputs = [pkgs.reuse];
        } ''
          cp -r ${self}/. src
          cd src
          reuse lint
          touch $out
        '';
    };
  };
}
