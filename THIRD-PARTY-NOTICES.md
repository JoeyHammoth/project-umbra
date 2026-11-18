# Third-party notices

UMBRA itself is MIT-licensed (see [LICENSE](LICENSE)). This file records
material that is **vendored into this repository** and therefore travels with
any copy of it, together with the terms it arrives under.

Dependencies fetched by `cargo` and `npm` are not vendored; their exact versions
and licence metadata are recorded in `Cargo.lock` and `web/package-lock.json`.

## Random123 known-answer vectors

- **File:** `crates/umbra-core/tests/kat_vectors`
- **Upstream:** <https://github.com/DEShawResearch/random123>, path `tests/kat_vectors`
- **Vendored as:** the complete file, unmodified — not edited, trimmed or
  regenerated. Its own header comment is retained, and the copy is checked by
  SHA-256 in review (`aab5ebabf40003f6…` for the revision vendored).
- **Used for:** test data only. `umbra-core` implements Philox-4x32 itself; no
  Random123 source code is compiled into any crate. The file supplies published
  reference answers that the RNG is checked against.
- **Licence:** BSD 3-Clause, reproduced in full below.

```text
Copyright 2010-2012, D. E. Shaw Research.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

* Redistributions of source code must retain the above copyright
  notice, this list of conditions, and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions, and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

* Neither the name of D. E. Shaw Research nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```
