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
    # cargo publish builds every package before it uploads it, here and in
    # its --check before a merge. Build scripts and blake3's C need a C
    # compiler and linker: the builds' own, stdenv.cc, which every package
    # and check here builds with. On macOS, Rust's standard library also
    # links libiconv, which the SDK gives the builds.
    publish-releases = pkgs.writeShellApplication {
      name = "publish-releases";
      runtimeInputs = [
        pkgs.stdenv.cc
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
      runtimeEnv =
        {
          PUBLISH_FLAGS = "--locked";
        }
        // pkgs.lib.optionalAttrs pkgs.stdenv.hostPlatform.isDarwin {
          LIBRARY_PATH = "${pkgs.libiconv}/lib";
        };
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
        meta.description = "Push the release tags merged into master to GitHub, and publish their crates (--check: all but the upload)";
      };
    };

    checks.release = pkgs.callPackage ./release-test.nix {
      inherit release publish-releases rustToolchain;
    };
  };
}
