# SPDX-FileCopyrightText: 2026 Mika Tammi
# SPDX-License-Identifier: MIT OR Apache-2.0
#
# The furymodel-cli package, from the dependencies' shared cargoArtifacts.
# Its check runs clippy over the workspace, with warnings as errors, and the
# tests a build in the Nix sandbox can run (nextest's sandbox profile), so
# that building the package on a system tests it there.
{
  perSystem = {
    cargoArtifacts,
    commonArgs,
    craneLib,
    lib,
    pkgs,
    ...
  }: let
    furymodel-cli = craneLib.buildPackage (commonArgs
      // {
        inherit cargoArtifacts;
        pname = "furymodel-cli";
        cargoExtraArgs = "--locked --package furymodel-cli";
        nativeCheckInputs = [pkgs.cargo-nextest];
        checkPhaseCargoCommand = ''
          cargoWithProfile clippy --locked --workspace --all-targets -- --deny warnings
          cargo nextest run ''${CARGO_PROFILE:+--cargo-profile $CARGO_PROFILE} \
            --locked --workspace --profile sandbox --no-tests=warn
        '';
        meta = {
          description = "Describe, verify, extend, split and compare FuryModel models";
          license = with lib.licenses; [mit asl20];
          mainProgram = "furymodel-cli";
        };
      });
  in {
    packages = {
      inherit furymodel-cli;
      default = furymodel-cli;
    };
  };
}
