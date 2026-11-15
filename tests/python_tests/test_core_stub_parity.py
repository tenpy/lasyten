"""Ensure generated ``lasyten._core`` stubs track the compiled extension API."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import re
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_SCRIPTS = _REPO / 'scripts'


def _is_compiled_core() -> bool:
    try:
        spec = importlib.util.find_spec('lasyten._core')
    except (ImportError, ModuleNotFoundError, ValueError):  # fmt: skip
        return False
    if spec is None or spec.loader is None:
        return False
    if isinstance(spec.loader, importlib.machinery.ExtensionFileLoader):
        return True
    origin = spec.origin or ''
    return origin.endswith('.so') or origin.endswith('.pyd')


@pytest.mark.skipif(not _is_compiled_core(), reason='compiled lasyten._core not available')
def test_core_stub_parity():
    sys.path.insert(0, str(_SCRIPTS))
    try:
        import lasyten._core as live
    except ImportError as exc:
        pytest.skip(f'compiled lasyten._core not importable: {exc}')

    origin = getattr(live, '__file__', '') or ''
    if origin.endswith('.py'):
        pytest.skip('lasyten._core is a pure-Python stub')

    from check_core_stub_parity import compare
    from generate_core_stubs import generate_stub

    stub_src = generate_stub(_REPO, from_doxygen=False, strict_docs=False)
    stub_path = _REPO / 'build' / '_core_stub_parity_pytest.py'
    stub_path.parent.mkdir(parents=True, exist_ok=True)
    stub_path.write_text(stub_src, encoding='utf-8')

    spec = importlib.util.spec_from_file_location('_core_stub_parity_pytest', stub_path)
    assert spec is not None and spec.loader is not None
    stub = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stub)

    errors = compare(live, stub)
    assert not errors, 'stub parity errors:\n' + '\n'.join(f'  - {e}' for e in errors[:40])


@pytest.mark.skipif(not _is_compiled_core(), reason='compiled lasyten._core not available')
def test_core_pyi_array_typing():
    """``py::array`` must appear as ``numpy.ndarray`` in live signatures / IDE stubs.

    Regression for the Stable-ABI stubgen footgun: ``PYTHONPATH=build`` +
    ``-m _core`` preferred a stale ``_core.cpython-*.so`` over ``_core.abi3.so``,
    so stubgen typed arrays as ``typing.Any``. Do not re-exec the extension here
    (nanobind forbids double registration); inspect the already-imported module.
    """
    try:
        import lasyten._core as live
    except ImportError as exc:
        pytest.skip(f'compiled lasyten._core not importable: {exc}')

    origin = getattr(live, '__file__', '') or ''
    if origin.endswith('.py'):
        pytest.skip('lasyten._core is a pure-Python stub')

    block_to_numpy = live.EigenBlockBackend.BlockCls.to_numpy
    sigs = [s[0] for s in block_to_numpy.__nb_signature__ if s[0]]
    assert any('numpy.ndarray' in s for s in sigs), (
        f'EigenBlockBackend.Block.to_numpy signatures lack numpy.ndarray: {sigs!r} '
        '(stale _core.cpython-*.so often shows typing.Any instead)'
    )
    assert not any('typing.Any' in s for s in sigs), sigs

    bb_from = live.BlockBackend.block_from_numpy.__nb_signature__[0][0]
    assert 'numpy.ndarray' in bb_from, bb_from

    pyi = _REPO / 'lasyten' / '_core.pyi'
    if pyi.is_file():
        text = pyi.read_text(encoding='utf-8')
        assert re.search(r'def to_numpy\(self\)\s*->\s*numpy\.ndarray', text)
        assert 'def block_from_numpy(self, a: numpy.ndarray' in text
        any_hits = len(re.findall(r'\bAny\b', text))
        assert any_hits < 50, (
            f'too many Any annotations in {pyi} ({any_hits}); regenerate with '
            'scripts/generate_core_pyi.py --so <fresh _core>'
        )
