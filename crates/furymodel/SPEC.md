<!--
SPDX-FileCopyrightText: 2026 Mika Tammi
SPDX-License-Identifier: MIT OR Apache-2.0
-->

# FuryModel format, version 1

FuryModel stores a model's tensors in files laid out for streaming: a
runtime reads a layer's or an expert's file with direct I/O, straight into
memory the GPU can use, verifying every 2 MiB chunk as it lands. The format
knows no model architecture. Tensor names, the architecture's identifier and
its configuration come from the runtime that defines the architecture; to
FuryModel they are opaque strings and JSON.

The key words "must", "must not" and "may" are normative. All integers in
binary structures are little-endian unless stated otherwise.

## 1. Files and roots

A model is a set of files, each named by a relative path:

- the manifest, `furymodel.json` (section 2);
- payload files, which hold the tensors (section 3);
- auxiliary files, such as the tokenizer or the chat template (section 2.6);
- hash tables, one per algorithm, under `hashes/` (section 5);
- in delivery parts only, `part.json` (section 7).

Paths are `/`-separated, relative, at most 1,024 bytes long, and made of
segments of the characters `A-Z a-z 0-9 . _ -`. A segment must not be
empty, `.` or `..`, and must be at most 255 bytes long. The paths
`furymodel.json`, `part.json` and `hashes`, and every path under
`hashes/`, are reserved: no payload or auxiliary file may use them, or
have them for a directory.

Many file systems hold `A.bin` and `a.bin` as one file, so paths are also
compared with the letters `A-Z` taken as `a-z`: the reserved paths are
reserved that way too, no two paths of a model may be equal that way, and
no path may be another's directory, that is, no path followed by `/` may
begin another path, either way.

The files live in one or more *roots*, directories that each hold a copy of
the manifest and some of the files (section 7).

## 2. The manifest

`furymodel.json` is a UTF-8 JSON object. Every field below is required
unless it says otherwise. A reader must reject a manifest with fields it
doesn't know, anywhere except inside `config`, the `extensions` map and
the values of that map, and a manifest in which an object has a key twice,
anywhere, `config` and `extensions` included: JSON parsers disagree on
which of the two counts.

The manifest is immutable: once a model is written, nothing rewrites it.
Hash tables and delivery parts are kept outside it, so that adding either
never changes the content id (section 6). A writer writes the manifest
last, once every other file of the model is durably in place, so that a
root holding a manifest holds a whole model.

| Field            | Type    | Meaning                                                    |
| ---------------- | ------- | ---------------------------------------------------------- |
| `format`         | string  | `"furymodel"`                                              |
| `format_version` | integer | `1`                                                        |
| `name`           | string  | display name, not empty                                    |
| `arch`           | string  | the runtime's architecture id, not empty                   |
| `schema_version` | integer | the runtime's tensor-schema version for `arch`, at least 1 |
| `config`         | object  | the runtime's typed configuration of the model             |
| `alignment`      | integer | `16384`                                                    |
| `files`          | array   | payload files described one by one (2.1)                   |
| `groups`         | array   | payload files described by a template (2.3)                |
| `aux`            | array   | auxiliary files (2.6)                                      |
| `provenance`     | object  | where the tensors came from (2.7)                          |
| `extensions`     | object  | optional: data outside this specification (2.8)            |

A reader checks `format` and `format_version` before anything else, and
rejects a manifest whose version it doesn't support.

### 2.1 Payload files

An entry of `files`:

| Field     | Type             | Meaning                                     |
| --------- | ---------------- | ------------------------------------------- |
| `path`    | string           | the file's path (section 1)                 |
| `size`    | integer          | the file's size in bytes                    |
| `tags`    | array of strings | the file's tags (2.5)                       |
| `tensors` | array            | the tensors the file holds, by offset (2.2) |

### 2.2 Tensors

| Field    | Type              | Meaning                                               |
| -------- | ----------------- | ----------------------------------------------------- |
| `name`   | string            | the runtime's canonical name, unique within the model |
| `dtype`  | string            | element type (4.1)                                    |
| `shape`  | array of integers | the logical shape; `[]` is a scalar                   |
| `layout` | string            | how logical indices map to elements (4.2)             |
| `scheme` | string            | how elements are encoded as bytes (4.3)               |
| `offset` | integer           | byte offset in the file, a multiple of `alignment`    |
| `size`   | integer           | the tensor's size in bytes, as its scheme computes it |

A tensor's name isn't empty. A file's tensors are listed by increasing
offset and must not overlap; each ends within the file.

### 2.3 Groups

A group describes many files of identical structure with one entry, such as
every expert of every layer. An entry of `groups`:

| Field     | Type             | Meaning                                                   |
| --------- | ---------------- | --------------------------------------------------------- |
| `name`    | string           | the group's name, unique among groups                     |
| `path`    | string           | a path template (2.4)                                     |
| `axes`    | array            | the template's axes, in order, at least one               |
| `size`    | integer          | every instance file's size                                |
| `tags`    | array of strings | every instance file's tags                                |
| `tensors` | array            | the tensors of every instance; `name` is a template (2.4) |

A group's name is made of `a-z 0-9 _ -` and isn't empty. An axis is
`{"name": NAME, "count": N}`, standing for the values `0, 1, …, N-1` (N at
least 1), or `{"name": NAME, "values": [V, …]}` with one or more strictly
increasing values. Axis names are made of `a-z 0-9 _`, start with a letter,
and are unique within the group. The group's instances are every
combination of axis values.

### 2.4 Templates

A template is a string in which `{NAME}` stands for the value of axis NAME
in decimal, and `{NAME:0W}` for the same value zero-padded to at least W
digits (W from 1 to 9). Braces have no other use. Between two of these
there must be a character other than a digit, such as `_`, `.` or `/`, so
that a template never renders two combinations of values the same. A
group's path template must use every axis of the group, and each of its
tensor name templates every axis with more than one value, so that no two
instances have a path or a tensor's name in common.

### 2.5 Tags

Tags are strings of `a-z 0-9 _ -`, not empty, unique within an entry. They
mark subsets of the files that a runtime loads or places together; the
towers of a model (for example `text`, `vision`, `drafter`) are tags. They
mean nothing to FuryModel itself.

### 2.6 Auxiliary files

Files the runtime needs besides tensors, stored as they came. An entry of
`aux`:

| Field  | Type    | Meaning                                                           |
| ------ | ------- | ----------------------------------------------------------------- |
| `path` | string  | the file's path, conventionally under `aux/`                      |
| `size` | integer | the file's size in bytes; no alignment is required                |
| `role` | string  | what the file is to the runtime, e.g. `tokenizer`; unique, opaque |

A role isn't empty.

### 2.7 Provenance

| Field        | Type             | Meaning                                                  |
| ------------ | ---------------- | -------------------------------------------------------- |
| `source`     | object           | the source the tensors were converted from               |
| `tool`       | object           | `{"name": …, "version": …}` of the program that wrote it |
| `transforms` | array of strings | every transform applied on the way, in order             |
| `notes`      | array of strings | anything else a reader of the provenance should know     |

`source` is `{"kind": KIND, "files": [...]}` with optional `repository`
and `revision` strings: KIND names the source format (for example
`safetensors`), and each entry of `files` is `{"path": …, "size": …}` with
an optional `sha2-256`, the SHA-256 of the source file in 64 lowercase hex
digits (the LFS object id on Hugging Face). The source's kind, a source
file's path and the tool's name and version aren't empty.

Transforms must only rearrange bytes: splitting, stacking, reordering
tensors or changing their layout. A conversion that changes values is not a
transform but a different model.

### 2.8 Extensions

`extensions` maps namespaced keys, two or more segments of `a-z 0-9 _ -`
joined by `.` (for example `example.speculative`), to any JSON values. It is
the only place for data this specification doesn't define. A reader ignores
extensions it doesn't know.

### 2.9 Limits

A few kilobytes of templates can describe millions of files, so a manifest
must keep the file table it describes within these limits, which a reader
checks before it writes out the instances of any group:

- at most 262,144 files, the manifest included, and at most 1,048,576
  tensors in all;
- group names of at most 64 bytes, at most 16 axes in a group, and tensor
  shapes of at most 16 dimensions;
- paths and tensor names of at most 1,024 bytes (section 1), as instances
  render them;
- a description of at most 256 MiB, every instance's own counted: 128
  bytes for each file and each tensor, 32 bytes more than its length for
  each path, tag, group name, role and tensor name, and 8 bytes for each
  dimension of a shape and each value of an instance's axes.

The description stands for what a reader holds of the file table, where
every instance of a group has its own copy of the group's tags, name and
tensors.

## 3. Payload files

A payload file is headerless: the bytes of its tensors at their offsets,
and zeros everywhere else. Its size and every tensor's offset are multiples
of `alignment`, 16384 bytes, so any of its tensors, or the whole file, can
be read with direct I/O into page-aligned memory.

## 4. Registries

The ids of dtypes, layouts and schemes come from the registries below. A
reader must reject an id it doesn't know. Registries only grow: an id never
changes its meaning. This is registry version 1.

### 4.1 Dtypes

`bf16` (bfloat16), `f16`, `f32` and `f64` (IEEE 754 binary16, binary32 and
binary64), and the integers `i8`, `u8`, `i16`, `u16`, `i32`, `u32`, `i64`
and `u64`: elements of 2, 2, 4, 8, 1, 1, 2, 2, 4, 4, 8 and 8 bytes.

### 4.2 Layouts

| Id          | Rank | Element of index `(i0, …, in-1)` in shape `(d0, …, dn-1)`       |
| ----------- | ---- | --------------------------------------------------------------- |
| `row_major` | any  | `((i0 · d1 + i1) · d2 + …) · dn-1 + in-1`                       |
| `out_in`    | 2    | `i · d1 + j` for the weight `W[i][j]` of a linear map `y = W x` |

`out_in` is `row_major` with a rank-2 shape `[out_features, in_features]`;
its own id lets a kernel declare that it reads a linear layer's weight in
exactly that order.

### 4.3 Schemes

| Id                                | Encoding                                 | Size in bytes         |
| --------------------------------- | ---------------------------------------- | --------------------- |
| the dtype ids of 4.1, e.g. `bf16` | each element as its dtype, little-endian | elements × dtype size |

A tensor's scheme must be the plain scheme of its dtype. Later schemes, such
as block quantisation, get ids of their own.

## 5. Hash tables

The files of a model, sorted by path, form its *file table*. It holds every
payload file, every auxiliary file and the manifest, `furymodel.json`, but
neither the hash tables nor `part.json`. Its order is the order of the
paths' bytes.

### 5.1 Algorithms

| Id            | Definition                                      | Digest   | Chunked |
| ------------- | ----------------------------------------------- | -------- | ------- |
| `gxhash3-128` | gxhash 3, `gxhash128(bytes, seed 0)`            | 16 bytes | yes     |
| `xxh3-128`    | XXH3-128 of xxHash 0.8 (seed 0, default secret) | 16 bytes | yes     |
| `sha2-512`    | SHA-512 (FIPS 180-4)                            | 64 bytes | no      |
| `sha3-512`    | SHA3-512 (FIPS 202)                             | 64 bytes | no      |
| `blake3`      | BLAKE3, 32-byte output                          | 32 bytes | no      |
| `sha2-256`    | SHA-256 (FIPS 180-4)                            | 32 bytes | no      |

A 128-bit digest is stored big-endian, as the hex form of the 128-bit value
reads, which for XXH3-128 is xxHash's canonical form.

gxhash3-128 and xxh3-128 are the fast checks. They detect corruption, not
tampering. gxhash's output is stable only within its major version 3, which
the id pins, and it needs AES instructions; xxh3-128 is the portable one.
sha3-512 is the cryptographic standard a model is identified with (section
6), and sha2-512 the other one it can be audited with. blake3 is the fast
cryptographic hash, and sha2-256 the one that other tools (Nix, Hugging
Face) speak.

The chunked algorithms divide each file into chunks of 2 MiB: chunk `k`
holds the bytes from `k · 2097152` up to `(k + 1) · 2097152` or the end of
the file, so a file of size `s` has `⌈s / 2097152⌉` chunks. A chunk's
digest is the algorithm's digest of its bytes, and the file's digest is the
algorithm's digest of its chunk digests, concatenated in order. The other
algorithms only have a file digest, the digest of the file's bytes.

### 5.2 Hash table files

The table of algorithm ID is the file `hashes/ID.bin`:

| Offset | Size | Field                                                         |
| ------ | ---- | ------------------------------------------------------------- |
| 0      | 8    | magic, the bytes `FMHASHES`                                   |
| 8      | 4    | table format version, `1`                                     |
| 12     | 4    | flags, `0`                                                    |
| 16     | 32   | the algorithm id in ASCII, padded with zero bytes             |
| 48     | 4    | the digest size D in bytes                                    |
| 52     | 4    | the chunk size, `2097152` for chunked algorithms, `0` if not  |
| 56     | 8    | the number of files F in the file table                       |
| 64     | 8    | the number of chunk digests C, `0` for algorithms not chunked |
| 72     | 32   | the file table's fingerprint (5.3)                            |
| 104    | 24   | reserved, zero bytes                                          |
| 128    | F·D  | the file digests, in file-table order                         |
|        | C·D  | the chunk digests, file by file in file-table order           |
|        | 16   | xxh3-128 of all the preceding bytes, big-endian               |

A reader must reject a table whose magic, version, flags, algorithm id,
digest size, chunk size, counts, fingerprint, reserved bytes or trailing
checksum aren't as described for the model's file table, or whose length
isn't exactly as the counts say.

A table can be added to a model at any time, since payloads never change,
and a model may have any subset of the tables; a writer of a new model
writes all six. A table added later is computed from the files as they are
read, each checked against a table the model has, the one a reader would
verify chunks with if there is one (section 8), so that the new table never
vouches for bytes the others don't. A table file is written whole or not
at all, under a name of the writer's own first and then renamed, so that
two writers of one table never write into one file.

A table a reader rejects, or one that differs between roots (section 7),
keeps the model from being used (section 8). It may be replaced: computed
from the files as a table added later is, each checked against another
table, and written over the old one in every root that holds it. A table
that gives a newer version than the writer's, with a trailing checksum
that holds, is left for a reader of that version.

A model with no table a reader can compute has none to check a new table
against. It may get a sha3-512 table computed from its files if the
content id that table gives (section 6) is one known for the model from a
source trusted for it: the files, the manifest among them, are then the
model's. A model with no table at all may also get tables computed from
its files as they are, which vouch for nothing but the bytes they held.

### 5.3 The file table's fingerprint

The first 32 bytes of the SHA3-512 of the ASCII string
`furymodel file table v1\n` followed, for each file in file-table order, by
the length of its path in bytes as a 4-byte integer, the path, and its size
as an 8-byte integer. It ties a hash table to the exact files and sizes it
was computed for.

## 6. The content id

A model's content id is the SHA3-512 of the ASCII string
`furymodel content id v1\n` followed, for each file in file-table order, by
the length of its path as a 4-byte integer, the path, its size as an 8-byte
integer and its 64-byte sha3-512 digest. It identifies the model's bytes,
manifest included, so two models with equal content ids hold equal files.
Adding a hash table or splitting a model into delivery parts doesn't change
it. Its text form is the lowercase hex of the 64 bytes; logs may show the
first 16 hex digits.

The content id and the file table's fingerprint rest on SHA-3 (FIPS 202)
rather than SHA-2 (FIPS 180-4). SHA-3 is the newer of the two standards and
is built on Keccak's sponge construction, independent of SHA-2's
Merkle-Damgård design, so that an attack found on SHA-2's construction
wouldn't carry over to a model's identity. The format itself therefore
needs no SHA-2: the sha2-512 and sha2-256 tables are optional like any
other.

## 7. Roots and delivery parts

A model may be spread over several roots: directories that each hold a copy
of `furymodel.json` and any subset of the other files, at their paths.

- The manifests of all roots must be byte for byte identical.
- A file present in several roots must have the same size in each; it is
  read from the first root, in the order the roots were given.
- A hash table present in several roots must be byte for byte identical.
- The model is complete when every file of the file table is present in
  some root with the size the manifest gives. A reader must not use an
  incomplete model.

A delivery part is a root made by splitting a model, so that each part fits
a size limit. It holds the manifest, every hash table, its share of the
files, and `part.json`:

| Field            | Type    | Meaning                                                         |
| ---------------- | ------- | --------------------------------------------------------------- |
| `format`         | string  | `"furymodel-part"`                                              |
| `format_version` | integer | `1`                                                             |
| `content_id`     | string  | the model's content id, in hex                                  |
| `part`           | integer | this part's number, from 1                                      |
| `parts`          | integer | the number of parts                                             |
| `files`          | array   | the part's files as `[start, end)` ranges of file-table indices |

As in the manifest, fields this specification doesn't define and objects
with a key twice are errors. The ranges of all parts together cover every
index of the file table exactly once; every part holds the manifest as
well, whichever part's range it falls in, and gets it last, as a writer
gets a model's (section 2). A reader that finds an incomplete model whose
roots are parts can name the missing parts.

## 8. Reading

A reader:

1. reads and checks the manifest, the same in every root, and builds the
   file table;
1. checks that the model is complete;
1. reads the hash tables present and checks them (5.2), and must not use a
   model with a table that fails the checks, whichever table it is, nor one
   with no table the reader can compute, which can't be verified;
1. picks the fastest table it supports for verification: gxhash3-128 when
   it has AES instructions, xxh3-128 otherwise; and
1. verifies every chunk it reads against that table before using the
   chunk's bytes. A chunk that doesn't match is read once more, from the
   storage rather than from a cache that may hold the same bytes, and if
   it still doesn't match, the read fails. Nothing ever computes on bytes
   that haven't been verified.

The manifest is verified like any file of the file table before its
contents are trusted. A full verification checks every file against every
table present.

The fast tables find corruption, not changes made on purpose: whoever
changes a file can compute its gxhash3-128 and xxh3-128 digests again.
Until the sha3-512 table has been checked against the files, the content
id it gives is only that table's claim; once it has, the content id is the
files', and they are the model a source trusted for its content id names
if the two ids are equal. A fast table is trusted to verify chunks with
once the same pass over the files has checked it too. So a model is checked
once, when it is installed, against its sha3-512 table and the fast table
it will be read with, together and with the content id expected of it;
from then on its chunks are verified against the fast table as they are
read.

## 9. Versioning

`format_version` changes with every change to the meaning of a manifest, a
payload file or a hash table, except for new registry ids (section 4), which
older readers reject on their own. Hash tables and `part.json` carry their
own versions.
