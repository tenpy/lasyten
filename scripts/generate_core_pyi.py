#!/usr/bin/env python3
"""Generate ``lasyten/_core.pyi`` via ``nanobind.stubgen`` from a specific ``_core`` .so.

CMake used to run ``PYTHONPATH=build python -m nanobind.stubgen -m _core``. With
``LASYTEN_STABLE_ABI``, the build directory can contain both ``_core.abi3.so`` (current)
and a leftover ``_core.cpython-*.so`` (stale). CPython then prefers the tagged
extension for the running interpreter, so stubgen types ``py::array`` as
``typing.Any`` instead of ``numpy.ndarray``.

This script loads ``$LASYTEN_CORE_SO`` (or the newest matching build artifact)
explicitly as ``lasyten._core`` and invokes stubgen on that module.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _pick_core_so(repo: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        if not explicit.is_file():
            raise FileNotFoundError(f'LASYTEN_CORE_SO / --so not found: {explicit}')
        return explicit

    env = os.environ.get('LASYTEN_CORE_SO')
    if env:
        p = Path(env)
        if not p.is_file():
            raise FileNotFoundError(f'LASYTEN_CORE_SO not found: {p}')
        return p

    build = repo / 'build'
    candidates = list(build.glob('_core*.so')) + list(build.glob('_core*.pyd'))
    # Prefer Stable ABI artifact when present; else newest mtime.
    abi3 = [c for c in candidates if '.abi3.' in c.name or c.name.endswith('.abi3.so')]
    if abi3:
        return max(abi3, key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f'no _core*.so under {build}; pass --so or LASYTEN_CORE_SO')
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _load_as_lasyten_core(so: Path, repo: Path) -> ModuleType:
    """Load ``so`` as ``lasyten._core`` (package path = source ``lasyten/``)."""
    pkg_name = 'lasyten'
    mod_name = 'lasyten._core'
    if pkg_name not in sys.modules:
        pkg = ModuleType(pkg_name)
        pkg.__path__ = [str(repo / 'lasyten')]  # type: ignore[attr-defined]
        sys.modules[pkg_name] = pkg
    sys.modules.pop(mod_name, None)
    # Also drop a bare ``_core`` import so stubgen cannot pick a different SO.
    sys.modules.pop('_core', None)

    spec = importlib.util.spec_from_file_location(mod_name, so)
    if spec is None or spec.loader is None:
        raise ImportError(f'cannot create module spec for {so}')
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def generate_pyi(
    *,
    so: Path,
    output: Path,
    repo: Path | None = None,
    pattern_file: Path | None = None,
    include_docstrings: bool = True,
) -> Path:
    repo = repo or _repo_root()
    mod = _load_as_lasyten_core(so, repo)

    from nanobind.stubgen import StubGen, load_pattern_file

    patterns = load_pattern_file(str(pattern_file)) if pattern_file else []
    sg = StubGen(
        mod,
        include_docstrings=include_docstrings,
        patterns=patterns,
        quiet=False,
    )
    sg.put(mod)
    if patterns:
        unmatched = [p for p in patterns if p.matches == 0]
        for p in unmatched:
            print(f'  warning: stubgen pattern did not match: {p.query.pattern!r}')
        matched = sum(p.matches for p in patterns)
        print(f'  applied {matched} stubgen pattern match(es)')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(sg.get(), encoding='utf-8')
    return output


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        '-o',
        '--output',
        type=Path,
        default=None,
        help='Output .pyi path (default: lasyten/_core.pyi)',
    )
    ap.add_argument(
        '--so',
        type=Path,
        default=None,
        help='Compiled _core extension to stub (default: LASYTEN_CORE_SO or newest build/_core*.so)',
    )
    ap.add_argument(
        '-p',
        '--pattern-file',
        type=Path,
        default=None,
        help='Optional nanobind stubgen pattern file',
    )
    ap.add_argument(
        '--exclude-docstrings',
        action='store_true',
        help='Omit docstrings from the generated stub',
    )
    args = ap.parse_args()

    repo = _repo_root()
    so = _pick_core_so(repo, args.so)
    out = args.output or (repo / 'lasyten' / '_core.pyi')
    pattern = args.pattern_file
    if pattern is None:
        default_pat = repo / 'scripts' / 'stubgen_patterns.txt'
        if default_pat.is_file():
            pattern = default_pat

    print(f'loading {so}')
    generate_pyi(
        so=so,
        output=out,
        repo=repo,
        pattern_file=pattern,
        include_docstrings=not args.exclude_docstrings,
    )
    print(f'wrote {out}')
    return 0


if __name__ == '__main__':
    soft = os.environ.get('LASYTEN_STUBGEN_SOFT', '').lower() in ('1', 'true', 'yes')
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f'WARNING: generate_core_pyi failed: {exc}', file=sys.stderr)
        # CMake sets LASYTEN_STUBGEN_SOFT=1 so a stubgen failure does not fail the build.
        raise SystemExit(0 if soft else 1)
