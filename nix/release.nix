# SPDX-FileCopyrightText: 2026 Mika Tammi
# SPDX-License-Identifier: MIT OR Apache-2.0
#
# Releases. `nix run .#release -- VERSION` starts one: a signed release
# commit and tag, pushed to GitLab with a merge request (release.sh).
# Master's pipeline finishes it with `nix run .#publish-releases`, which
# pushes the tag to GitHub and publishes the crates to crates.io
# (publish-releases.sh). The two scripts are the same in every repository
# of the project; what is particular to one is set here.
{
  perSystem = {
    pkgs,
    rustToolchain,
    ...
  }: let
    # Signing goes through git, with the user's own gpg (gpg.program).
    release = pkgs.writeShellApplication {
      name = "release";
      runtimeInputs = [
        pkgs.coreutils
        pkgs.curl
        pkgs.gawk
        pkgs.git
        pkgs.glab
        pkgs.jq
        pkgs.yq-go
        rustToolchain
      ];
      text = builtins.readFile ./release.sh;
    };
    publish-releases = pkgs.writeShellApplication {
      name = "publish-releases";
      runtimeInputs = [
        pkgs.coreutils
        pkgs.curl
        pkgs.gawk
        pkgs.git
        pkgs.gnupg
        pkgs.jq
        pkgs.openssh
        rustToolchain
      ];
      # No crate of ours among the dependencies: Cargo.lock is the one that
      # users get, and crates.io's copies build on the Linux runner, which
      # the publish's verification does.
      runtimeEnv.PUBLISH_FLAGS = "--locked";
      text = builtins.readFile ./publish-releases.sh;
    };
  in {
    apps = {
      release = {
        type = "app";
        program = pkgs.lib.getExe release;
        meta.description = "Start a release: a signed release commit and tag, and their merge request";
      };
      publish-releases = {
        type = "app";
        program = pkgs.lib.getExe publish-releases;
        meta.description = "Push the release tags merged into master to GitHub, and publish their crates";
      };
    };

    checks.release = pkgs.callPackage ./release-test.nix {
      inherit release publish-releases rustToolchain;
    };
  };
}
