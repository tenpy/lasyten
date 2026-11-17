#!/usr/bin/env python3
"""Validate that a lasyten wheel is tagged cp312-abi3 and ships ``_core.abi3.so``.

Used by ``.github/workflows/publish.yml`` (issue #326 item 4 / leftover #2).
"""

from __future__ import annotations

import argparse
import re
import sys
import zipfile
from pathlib import Path


_ABI3_TAG = re.compile(r'-cp312-abi3-')
_CORE_ABI3 = re.compile(r'(^|/)lasyten/_core\.abi3\.(so|dylib|pyd)$')


def check_wheel(path: Path) -> list[str]:
    errors: list[str] = []
    name = path.name
    if not _ABI3_TAG.search(name):
        errors.append(f'{name}: expected cp312-abi3 in the wheel filename (got Stable ABI tag missing)')

    with zipfile.ZipFile(path) as zf:
        members = zf.namelist()
        core = [m for m in members if _CORE_ABI3.search(m)]
        if not core:
            # Also accept Windows-style / unusual layouts under lasyten/
            core = [m for m in members if '_core.abi3.' in m and m.startswith('lasyten/')]
        if not core:
            sample = [m for m in members if '_core' in m][:8]
            errors.append(f'{name}: missing lasyten/_core.abi3.* (found _core entries: {sample})')

    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('wheels', nargs='+', type=Path, help='Wheel paths to check')
    args = parser.parse_args(argv)

    all_errors: list[str] = []
    for wheel in args.wheels:
        if not wheel.is_file():
            all_errors.append(f'{wheel}: file not found')
            continue
        errs = check_wheel(wheel)
        if errs:
            all_errors.extend(errs)
        else:
            print(f'OK: {wheel.name}')

    if all_errors:
        for e in all_errors:
            print(e, file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
