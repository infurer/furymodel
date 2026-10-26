# SPDX-FileCopyrightText: 2026 Mika Tammi
# SPDX-License-Identifier: MIT OR Apache-2.0
#
# The furymodel-cli package, from the dependencies' shared cargoArtifacts.
# Its tests run in checks.nextest, so the package doesn't run them again.
{
  perSystem = {
    cargoArtifacts,
    commonArgs,
    craneLib,
    lib,
    ...
  }: let
    furymodel-cli = craneLib.buildPackage (commonArgs
      // {
        inherit cargoArtifacts;
        pname = "furymodel-cli";
        cargoExtraArgs = "--locked --package furymodel-cli";
        doCheck = false;
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
