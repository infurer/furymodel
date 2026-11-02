<!--
SPDX-FileCopyrightText: 2026 Mika Tammi
SPDX-License-Identifier: MIT OR Apache-2.0
-->

# furymodel

FuryModel: a model file format and its Rust library, laid out for streaming
layers and experts from disk.

The format is specified in [SPEC.md](SPEC.md), which the crate's
documentation shows as its `spec` module. The library writes models, opens
them from their roots, and reads them with every chunk verified against the
model's hash tables. Its command line tool is the `furymodel-cli` crate.

Licensed under the MIT license or the Apache License, Version 2.0, at your
option.
