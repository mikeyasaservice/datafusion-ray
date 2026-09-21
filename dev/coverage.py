#!/usr/bin/env python3
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Report line coverage of production code only.

Tests live in `#[cfg(test)] mod` blocks inside the files they cover, which is
what gives them access to private items. llvm-cov counts those blocks as
covered lines, so the headline percentage rises just by writing tests -- and a
CI gate on it could be satisfied by test code counting itself.

Every such module in this crate sits at the end of its file, so the exclusion
is exact: everything from the `#[cfg(test)]` line to EOF is test code.

    dev/coverage.py                 # report
    dev/coverage.py --fail-under 85 # report and gate
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import tempfile

REPO = pathlib.Path(__file__).resolve().parent.parent


def test_module_line(path: pathlib.Path) -> int | None:
    """First line of the file's `#[cfg(test)]` module, 1-indexed."""
    for i, line in enumerate(path.read_text().splitlines(), start=1):
        if line.strip() == "#[cfg(test)]":
            return i
    return None


def collect(export: dict) -> dict[str, tuple[int, int]]:
    """Map file -> (production lines covered, production lines total)."""
    out: dict[str, tuple[int, int]] = {}
    for data in export.get("data", []):
        for file_entry in data.get("files", []):
            path = pathlib.Path(file_entry["filename"])
            try:
                rel = path.relative_to(REPO)
            except ValueError:
                continue
            if rel.parts[0] != "src":
                continue

            cutoff = test_module_line(path)
            covered = total = 0
            # segments are [line, col, count, has_count, is_region_entry, ...]
            seen: dict[int, int] = {}
            for seg in file_entry.get("segments", []):
                line, _col, count, has_count = seg[0], seg[1], seg[2], seg[3]
                if not has_count:
                    continue
                if cutoff is not None and line >= cutoff:
                    continue
                # a line is covered if any segment on it executed
                seen[line] = max(seen.get(line, 0), count)
            for count in seen.values():
                total += 1
                if count > 0:
                    covered += 1
            out[str(rel)] = (covered, total)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--fail-under",
        type=float,
        default=None,
        help="exit non-zero if production coverage is below this percentage",
    )
    ap.add_argument(
        "--no-run",
        action="store_true",
        help="reuse an existing llvm-cov export instead of running the tests",
    )
    ap.add_argument("--json", type=pathlib.Path, default=None)
    args = ap.parse_args()

    export_path = args.json or pathlib.Path(tempfile.gettempdir()) / "dfray-cov.json"
    if not args.no_run:
        subprocess.run(
            [
                "cargo",
                "llvm-cov",
                "--no-default-features",
                "--json",
                "--output-path",
                str(export_path),
            ],
            cwd=REPO,
            check=True,
        )

    files = collect(json.loads(export_path.read_text()))
    if not files:
        print("no src/ files in the coverage export", file=sys.stderr)
        return 2

    width = max(len(f) for f in files)
    print(f"{'file':<{width}}  {'covered':>7}  {'lines':>6}  {'cover':>7}")
    print("-" * (width + 26))
    tc = tt = 0
    for name in sorted(files):
        covered, total = files[name]
        tc += covered
        tt += total
        pct = 100 * covered / total if total else 100.0
        print(f"{name:<{width}}  {covered:>7}  {total:>6}  {pct:>6.1f}%")
    print("-" * (width + 26))
    overall = 100 * tc / tt if tt else 100.0
    print(f"{'TOTAL (production only)':<{width}}  {tc:>7}  {tt:>6}  {overall:>6.1f}%")

    if args.fail_under is not None and overall < args.fail_under:
        print(
            f"\nproduction coverage {overall:.1f}% is below the required "
            f"{args.fail_under:.1f}%",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
