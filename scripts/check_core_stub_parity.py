#!/usr/bin/env python3
"""Compare a generated ``lasyten._core`` stub against the live compiled extension.

Run after a normal C++ build / ``pip install -e .``::

    python scripts/check_core_stub_parity.py --generate

Also invoked as a local pre-commit hook (skips if the compiled extension is
not importable). To compare against an already-generated stub::

    python scripts/generate_core_stubs.py -o /tmp/_core_stub.py
    python scripts/check_core_stub_parity.py /tmp/_core_stub.py

Exit status is non-zero if public names, per-class members, or ``lasyten-cpp-ref``
markers drift.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def _load_stub(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location('_core_stub_under_test', path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _public_names(mod: ModuleType) -> set[str]:
    skip = {'annotations', 'enum'}
    return {n for n in dir(mod) if not n.startswith('_') and n not in skip}


def _own_members(cls: type) -> set[str]:
    return {n for n in cls.__dict__ if not n.startswith('_')}


def _has_cpp_ref(doc: str | None) -> bool:
    return bool(doc) and 'lasyten-cpp-ref' in doc


def compare(live: ModuleType, stub: ModuleType) -> list[str]:
    errors: list[str] = []
    live_names = _public_names(live)
    stub_names = _public_names(stub)
    if live_names != stub_names:
        missing = sorted(live_names - stub_names)
        extra = sorted(stub_names - live_names)
        if missing:
            errors.append(f'module names missing from stub ({len(missing)}): {missing[:20]}')
        if extra:
            errors.append(f'extra module names in stub ({len(extra)}): {extra[:20]}')

    for name in sorted(live_names & stub_names):
        live_obj = getattr(live, name)
        stub_obj = getattr(stub, name)
        if not isinstance(live_obj, type):
            # Module-level function / constant: check cpp-ref presence when live has it
            if callable(live_obj) and _has_cpp_ref(getattr(live_obj, '__doc__', None)):
                if not _has_cpp_ref(getattr(stub_obj, '__doc__', None)):
                    errors.append(f'{name}: missing lasyten-cpp-ref in stub docstring')
            continue
        if not isinstance(stub_obj, type):
            errors.append(f'{name}: live is a type, stub is {type(stub_obj).__name__}')
            continue
        live_m = _own_members(live_obj)
        stub_m = _own_members(stub_obj)
        missing = sorted(live_m - stub_m)
        if missing:
            errors.append(f'{name}: missing members {missing[:15]}' + (
                f' (+{len(missing) - 15} more)' if len(missing) > 15 else ''
            ))
        # Class docstring marker
        if _has_cpp_ref(getattr(live_obj, '__doc__', None)):
            if not _has_cpp_ref(getattr(stub_obj, '__doc__', None)):
                errors.append(f'{name}: missing lasyten-cpp-ref on class docstring')
        # Sample members that have markers on the live object. Skip data
        # attributes (e.g. ``FermionParity.even``) whose ``__doc__`` is the
        # instance type's class docstring rather than a binding-level doc.
        for mem in sorted(live_m & stub_m):
            live_mem = getattr(live_obj, mem, None)
            stub_mem = getattr(stub_obj, mem, None)
            if not callable(live_mem) and not isinstance(live_mem, property):
                continue
            live_doc = getattr(live_mem, '__doc__', None)
            if isinstance(live_mem, property):
                live_doc = live_mem.fget.__doc__ if live_mem.fget else live_doc
            if _has_cpp_ref(live_doc):
                stub_doc = getattr(stub_mem, '__doc__', None)
                if isinstance(stub_mem, property):
                    stub_doc = stub_mem.fget.__doc__ if stub_mem.fget else stub_doc
                if not _has_cpp_ref(stub_doc):
                    errors.append(f'{name}.{mem}: missing lasyten-cpp-ref in stub docstring')
    return errors


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        'stub',
        type=Path,
        nargs='?',
        default=None,
        help='Path to generated stub (default: regenerate to a temp path)',
    )
    ap.add_argument(
        '--generate',
        action='store_true',
        help='Regenerate the stub with scripts/generate_core_stubs.py before comparing',
    )
    args = ap.parse_args()

    live = _load_live_core()
    if live is None:
        print('SKIP: compiled lasyten._core not importable')
        return 0

    stub_path = args.stub
    if stub_path is None or args.generate:
        repo = Path(__file__).resolve().parent.parent
        stub_path = repo / 'build' / '_core_stub_parity.py'
        stub_path.parent.mkdir(parents=True, exist_ok=True)
        from generate_core_stubs import generate_stub

        stub_path.write_text(
            generate_stub(repo, from_doxygen=False, strict_docs=False),
            encoding='utf-8',
        )
        print(f'generated {stub_path}')

    stub = _load_stub(stub_path)
    errors = compare(live, stub)
    if errors:
        print(f'FAIL: {len(errors)} parity issue(s)')
        for err in errors[:50]:
            print(f'  - {err}')
        if len(errors) > 50:
            print(f'  ... ({len(errors) - 50} more)')
        return 1
    print('OK: stub matches live lasyten._core API surface')
    return 0


def _load_so(so: Path, *, repo: Path) -> ModuleType | None:
    """Load a compiled ``_core`` extension from ``so`` into ``sys.modules``."""
    try:
        spec = importlib.util.spec_from_file_location('lasyten._core', so)
        if spec is None or spec.loader is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        if 'lasyten' not in sys.modules:
            pkg = ModuleType('lasyten')
            pkg.__path__ = [str(repo / 'lasyten')]  # type: ignore[attr-defined]
            sys.modules['lasyten'] = pkg
        # Drop a previously imported (possibly stale installed) extension.
        sys.modules.pop('lasyten._core', None)
        sys.modules['lasyten._core'] = mod
        spec.loader.exec_module(mod)
        return mod
    except Exception as exc:
        print(f'SKIP: could not load {so} ({exc})')
        return None


def _load_live_core() -> ModuleType | None:
    """Import the compiled extension without preferring a pure-Python stub.

    Prefer ``build/_core*.so`` when present so local pre-commit (which runs
    ``cmake --build build`` just before this check) compares against the
    freshly built extension rather than a stale installed wheel/editable copy.
    """
    repo = Path(__file__).resolve().parent.parent

    build_candidates = sorted((repo / 'build').glob('_core*.so'))
    if build_candidates:
        mod = _load_so(build_candidates[-1], repo=repo)
        if mod is not None:
            return mod

    # Prefer an already-imported extension module.
    existing = sys.modules.get('lasyten._core')
    if existing is not None:
        origin = getattr(existing, '__file__', '') or ''
        if origin.endswith('.so') or origin.endswith('.pyd'):
            return existing

    # Avoid importing the lasyten package (editable installs may rebuild).
    try:
        import lasyten  # noqa: F401
    except Exception:
        pass

    try:
        import lasyten._core as live
    except Exception:
        live = None
    if live is not None:
        origin = getattr(live, '__file__', '') or ''
        if not (origin.endswith('.py') or origin.endswith('.pyc')):
            return live

    # Last resort: load .so from the scikit-build install tree.
    candidates = sorted((repo / 'build' / 'install' / 'platlib' / 'lasyten').glob('_core*.so'))
    if not candidates:
        return None
    return _load_so(candidates[-1], repo=repo)


if __name__ == '__main__':
    # Allow importing sibling scripts/
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.exit(main())
