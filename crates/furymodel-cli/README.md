<!--
SPDX-FileCopyrightText: 2026 Mika Tammi
SPDX-License-Identifier: MIT OR Apache-2.0
-->

# furymodel-cli

The command line tool of FuryModel, a model file format laid out for
streaming layers and experts from disk:

- `info` describes a model: its files, tensors, tags, hash tables,
  provenance and content id;
- `verify` checks a model's files against its hash tables, or that it is the
  model of a content id;
- `hash add` computes a hash table the model lacks;
- `split` splits a model into delivery parts, each within a size limit;
- `diff` compares two models by their manifests and hash tables.

The format and its library are the `furymodel` crate.

Licensed under the MIT license or the Apache License, Version 2.0, at your
option.
