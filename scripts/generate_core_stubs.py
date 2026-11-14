#!/usr/bin/env python3
"""Generate a bag-of-names ``lasyten/_core.py`` stub for Sphinx autodoc (no C++ build).

Parses ``pybind/**/*.cpp`` for class / function / enum / exception registrations and
resolves ``DOC(...)`` / ``doc_plus`` / ``doc_cpp_ref`` / ``R"pydoc"`` against a
docstring map from Doxygen (or existing ``pybind/docstrings/`` headers).

Usage::

    # Prefer existing CMake-generated DOC headers when present:
    python scripts/generate_core_stubs.py

    # Or rebuild DOC map via scoped Doxygen (Read the Docs):
    python scripts/generate_core_stubs.py --from-doxygen

    # Write to a custom path:
    python scripts/generate_core_stubs.py -o /tmp/_core.py
"""

from __future__ import annotations

import argparse
import enum
import re
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from doxygen_xml_to_docstrings import (  # noqa: E402
    build_doc_map,
    lookup_doc,
    parse_docstring_headers,
)

_REPO = _SCRIPTS.parent
_PYBIND = _REPO / 'pybind'
_TRAMPOLINE_RE = re.compile(r'^Py[A-Z]')

# ---------------------------------------------------------------------------
# Doc helpers (mirror pybind/doc_plus.h)
# ---------------------------------------------------------------------------

_CPP_REF_MARKER = '.. lasyten-cpp-ref::'


def _format_cpp_ref_marker(cpp_symbol: str, role: str = 'func') -> str:
    sym = cpp_symbol.strip()
    if '(' in sym:
        sym = sym[: sym.index('(')].rstrip()
    marker = f'{_CPP_REF_MARKER} {sym}\n'
    if role and role != 'func':
        marker += f'   :role: {role}\n'
    return marker


def doc_plus(shared: str, python_extra: str) -> str:
    if not python_extra:
        return shared
    if not shared:
        return python_extra
    pos = shared.rfind(_CPP_REF_MARKER)
    if pos < 0:
        base = shared if shared.endswith('\n') else shared + '\n'
        return base + '\n' + python_extra
    before = shared[:pos].rstrip()
    after = shared[pos:]
    extra = python_extra if python_extra.endswith('\n') else python_extra + '\n'
    return before + '\n\n' + extra + '\n' + after


def doc_cpp_ref(doc: str, cpp_symbol: str, role: str = 'func') -> str:
    marker = _format_cpp_ref_marker(cpp_symbol, role)
    if not doc:
        return marker
    pos = doc.rfind(_CPP_REF_MARKER)
    if pos < 0:
        base = doc if doc.endswith('\n') else doc + '\n'
        return base + '\n' + marker
    return doc[:pos].rstrip() + '\n\n' + marker


# ---------------------------------------------------------------------------
# C++ text utilities
# ---------------------------------------------------------------------------


def _strip_cpp_comments(text: str) -> str:
    """Remove // and /* */ comments without touching string contents (approx)."""
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        if text[i] == '"' or (text[i] == 'R' and i + 1 < n and text[i + 1] == '"'):
            # raw or normal string — copy through
            if text[i] == 'R':
                m = re.match(r'R"([^\s()]*)\(', text[i:])
                if m:
                    delim = m.group(1)
                    start = i + m.end()
                    end = text.find(f'){delim}"', start)
                    if end < 0:
                        out.append(text[i:])
                        break
                    out.append(text[i : end + len(delim) + 2])
                    i = end + len(delim) + 2
                    continue
            # normal / escaped string
            j = i + 1
            while j < n:
                if text[j] == '\\':
                    j += 2
                    continue
                if text[j] == '"':
                    j += 1
                    break
                j += 1
            out.append(text[i:j])
            i = j
            continue
        if text[i : i + 2] == '//':
            while i < n and text[i] != '\n':
                i += 1
            continue
        if text[i : i + 2] == '/*':
            end = text.find('*/', i + 2)
            i = n if end < 0 else end + 2
            continue
        out.append(text[i])
        i += 1
    return ''.join(out)


def _skip_string_or_char(text: str, i: int) -> int:
    """If ``text[i]`` starts a string/char/raw-string, return index past it; else ``i``."""
    n = len(text)
    if i >= n:
        return i
    # Raw string R"delim( ... )delim"
    if text[i] == 'R' and i + 1 < n and text[i + 1] == '"':
        m = re.match(r'R"([^\s()]*)\(', text[i:])
        if m:
            delim = m.group(1)
            start = i + m.end()
            end = text.find(f'){delim}"', start)
            if end < 0:
                raise ValueError('unclosed raw string')
            return end + len(delim) + 2
    # Character literal 'x' / '\n' / '\'' / '"'
    if text[i] == "'":
        j = i + 1
        if j < n and text[j] == '\\':
            j += 2
        elif j < n:
            j += 1
        if j < n and text[j] == "'":
            return j + 1
        return i + 1
    # Normal string
    if text[i] == '"':
        j = i + 1
        while j < n:
            if text[j] == '\\':
                j += 2
                continue
            if text[j] == '"':
                return j + 1
            j += 1
        return n
    return i


def _matching_paren(text: str, open_idx: int) -> int:
    """Return index of matching closing paren; ``open_idx`` points at ``(``."""
    depth = 0
    i = open_idx
    n = len(text)
    while i < n:
        nxt = _skip_string_or_char(text, i)
        if nxt != i:
            i = nxt
            continue
        ch = text[i]
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
            if depth == 0:
                return i
        i += 1
    raise ValueError('unbalanced parentheses')


def _matching_brace(text: str, open_idx: int) -> int:
    """Return index of matching closing brace; ``open_idx`` points at ``{``."""
    depth = 0
    i = open_idx
    n = len(text)
    while i < n:
        nxt = _skip_string_or_char(text, i)
        if nxt != i:
            i = nxt
            continue
        ch = text[i]
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return i
        i += 1
    raise ValueError('unbalanced braces')


def _split_args(arglist: str) -> list[str]:
    """Split a top-level comma-separated C++ argument list.

    Tracks ``()``, ``{}``, and template ``<>``. Angle brackets inside ``{}``
    are ignored so comparison operators in lambda bodies (``x < cutoff``) do
    not swallow the remaining ``m.def`` arguments.
    """
    args: list[str] = []
    buf: list[str] = []
    depth = 0
    angle = 0
    brace = 0
    i = 0
    n = len(arglist)
    while i < n:
        nxt = _skip_string_or_char(arglist, i)
        if nxt != i:
            buf.append(arglist[i:nxt])
            i = nxt
            continue
        ch = arglist[i]
        if ch == '{':
            brace += 1
        elif ch == '}' and brace:
            brace -= 1
        elif ch == '<' and brace == 0:
            angle += 1
        elif ch == '>' and angle and brace == 0:
            angle -= 1
        elif ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
        elif ch == ',' and depth == 0 and angle == 0 and brace == 0:
            args.append(''.join(buf).strip())
            buf = []
            i += 1
            continue
        buf.append(ch)
        i += 1
    tail = ''.join(buf).strip()
    if tail:
        args.append(tail)
    return args


def _parse_string_literal(expr: str) -> str | None:
    expr = expr.strip()
    m = re.match(r'R"([^\s()]*)\((.*)\)\1"', expr, re.DOTALL)
    if m:
        return m.group(2)
    if len(expr) >= 2 and expr[0] == '"' and expr[-1] == '"':
        # naive unescape of common sequences
        inner = expr[1:-1]
        return (
            inner.replace(r'\n', '\n')
            .replace(r'\t', '\t')
            .replace(r'\"', '"')
            .replace(r'\\', '\\')
        )
    return None


# ---------------------------------------------------------------------------
# Stub IR
# ---------------------------------------------------------------------------


@dataclass
class Member:
    name: str
    kind: str  # method | staticmethod | property | readonly | readwrite | classmethod
    doc: str = ''


@dataclass
class ClassStub:
    name: str
    bases: list[str] = field(default_factory=list)
    doc: str = ''
    members: dict[str, Member] = field(default_factory=dict)
    nested: dict[str, ClassStub] = field(default_factory=dict)
    is_enum: bool = False
    enum_base: str = 'enum.Enum'
    enum_values: list[str] = field(default_factory=list)
    is_exception: bool = False
    exception_base: str = 'Exception'


@dataclass
class ModuleStub:
    doc: str = 'LASyTen python bindings using pybind11'
    classes: dict[str, ClassStub] = field(default_factory=dict)
    functions: dict[str, Member] = field(default_factory=dict)
    constants: dict[str, str] = field(default_factory=dict)  # name -> repr/doc note
    # var name in C++ → python class name (for nested registration / chaining)
    handles: dict[str, str] = field(default_factory=dict)
    # Chronological handle bindings: (source_offset, handle, py_name)
    handle_history: list[tuple[int, str, str]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Doc expression evaluator
# ---------------------------------------------------------------------------


class DocResolver:
    def __init__(self, doc_map: dict[str, str], *, strict: bool = True):
        self.doc_map = doc_map
        self.strict = strict
        self.missing: list[str] = []

    def resolve(self, expr: str) -> str:
        expr = expr.strip()
        if not expr:
            return ''
        # DOC(lasyten, …)
        if expr.startswith('DOC('):
            close = _matching_paren(expr, expr.index('('))
            inner = expr[4:close]
            rest = expr[close + 1 :].strip()
            if rest:
                # DOC(...) is only part of a larger expression — unexpected
                pass
            args = _split_args(inner)
            parsed: list[str | int] = []
            for a in args:
                a = a.strip()
                if a.isdigit():
                    parsed.append(int(a))
                else:
                    parsed.append(a)
            doc = lookup_doc(self.doc_map, *parsed)
            if doc is None:
                key = ', '.join(str(p) for p in parsed)
                self.missing.append(key)
                if self.strict:
                    raise KeyError(f'missing DOC({key}) in docstring map')
                return ''
            return doc
        if expr.startswith('doc_plus('):
            close = _matching_paren(expr, expr.index('('))
            args = _split_args(expr[len('doc_plus(') : close])
            if len(args) < 2:
                return self.resolve(args[0]) if args else ''
            return doc_plus(self.resolve(args[0]), self.resolve(args[1]))
        if expr.startswith('doc_cpp_ref('):
            close = _matching_paren(expr, expr.index('('))
            args = _split_args(expr[len('doc_cpp_ref(') : close])
            doc = self.resolve(args[0]) if args else ''
            sym = _parse_string_literal(args[1]) if len(args) > 1 else ''
            role = _parse_string_literal(args[2]) if len(args) > 2 else 'func'
            return doc_cpp_ref(doc, sym or '', role or 'func')
        lit = _parse_string_literal(expr)
        if lit is not None:
            return lit
        return ''


# ---------------------------------------------------------------------------
# Binding parser
# ---------------------------------------------------------------------------

_SKIP_BASES = frozenset(
    {
        'py::smart_holder',
        'smart_holder',
    }
)


def _filter_bases(template_args: str) -> list[str]:
    """Extract Python-visible base class names from ``py::class_<T, Bases...>``."""
    parts = _split_args(template_args)
    bases: list[str] = []
    for i, p in enumerate(parts):
        p = p.strip()
        if i == 0:
            continue  # primary type
        # Drop template wrappers / holders / trampolines
        bare = p.split('<', 1)[0].split('::')[-1].strip()
        if bare in _SKIP_BASES or bare.startswith('py::'):
            continue
        if _TRAMPOLINE_RE.match(bare):
            continue
        # Nested C++ types like BlockBackend::Block are not Python bases
        if '::' in p and not p.startswith('py::'):
            # e.g. BlockBackend::Block — skip as base (nested class registration)
            continue
        bases.append(bare)
    return bases


def _python_name_from_string_arg(arg: str) -> str | None:
    return _parse_string_literal(arg.strip())


class BindingParser:
    def __init__(self, resolver: DocResolver):
        self.resolver = resolver
        self.mod = ModuleStub()
        self._doc_vars: dict[str, str] = {}
        # Shared bind helpers: name → members defined on the Class& parameter.
        # Bodies are skipped by ``_parse_chain_defs`` so members attach only via
        # call sites (e.g. ``bind_axis_common<>(pipe)`` → LegPipe).
        self._bind_helpers: dict[str, list[Member]] = {}
        self._helper_body_ranges: list[tuple[int, int]] = []

    def _bind_handle(self, name: str, py_name: str, pos: int) -> None:
        self.mod.handles[name] = py_name
        self.mod.handle_history.append((pos, name, py_name))

    def _in_helper_body(self, pos: int) -> bool:
        return any(start <= pos < end for start, end in self._helper_body_ranges)

    def _set_member(self, cls: ClassStub, name: str, kind: str, doc: str) -> None:
        """Record a member; keep an existing docstring when a later overload has none."""
        existing = cls.members.get(name)
        if existing is not None and existing.doc and not doc:
            return
        cls.members[name] = Member(
            name=name,
            kind=kind,
            doc=doc or (existing.doc if existing else ''),
        )

    def _doc_from_def_args(self, args: list[str]) -> str:
        """Last pybind argument that looks like a docstring expression."""
        for a in reversed(args[1:]):
            a = a.strip()
            if a.startswith('py::'):
                continue
            if a in self._doc_vars:
                return self._doc_vars[a]
            if (
                a.startswith('DOC(')
                or a.startswith('doc_plus(')
                or a.startswith('doc_cpp_ref(')
                or a.startswith('R"')
                or (len(a) >= 2 and a[0] == '"' and a[-1] == '"')
            ):
                return self.resolver.resolve(a)
        return ''

    def _handle_at(self, name: str, pos: int) -> str | None:
        """Python class name bound to C++ ``name`` just before ``pos``."""
        best: tuple[int, str] | None = None
        for off, h, py in self.mod.handle_history:
            if h == name and off <= pos and (best is None or off >= best[0]):
                best = (off, py)
        if best is not None:
            return best[1]
        return self.mod.handles.get(name)

    def parse_file(self, path: Path) -> None:
        text = _strip_cpp_comments(path.read_text(encoding='utf-8'))
        # Handles are C++ locals — reset per translation unit.
        self.mod.handles.clear()
        self.mod.handle_history.clear()
        self._doc_vars: dict[str, str] = {}
        self._bind_helpers.clear()
        self._helper_body_ranges.clear()
        self._parse_doc_vars(text)
        self._parse_classes(text)
        self._parse_handle_aliases(text)
        self._parse_native_enums(text)
        self._parse_exceptions(text)
        self._parse_module_defs(text)
        self._parse_module_attrs(text)
        self._parse_class_doc_assign(text)
        # Discover Class& bind helpers before chain defs so their bodies are skipped
        # and members are applied only at call sites below.
        self._parse_class_bind_helpers(text)
        self._parse_chain_defs(text)
        self._parse_enum_post_attrs(text)
        self._parse_template_call_sites(text)
        self._apply_bind_helper_calls(text)

    def _parse_handle_aliases(self, text: str) -> None:
        """Bind ``auto& alias = *g_handle;`` used after declare-then-define shells."""
        for m in re.finditer(
            r'(?:auto\s*&|py::class_\s*<[^>]+>\s*&)\s*([A-Za-z_][\w]*)\s*=\s*\*\s*([A-Za-z_][\w]*)\s*;',
            text,
        ):
            alias, source = m.group(1), m.group(2)
            py = self._handle_at(source, m.start())
            if py is None and source in self.mod.classes:
                py = source
            if py is not None:
                self._bind_handle(alias, py, m.start())

    def _parse_doc_vars(self, text: str) -> None:
        """Capture ``char const* name = DOC(...)/doc_plus(...)/R"pydoc"`` locals."""
        for m in re.finditer(
            r'(?:char\s+const\s*\*|const\s+char\s*\*|auto)\s+([A-Za-z_][\w]*)\s*=\s*',
            text,
        ):
            name = m.group(1)
            start = m.end()
            end = self._statement_end(text, start)
            expr = text[start:end].strip()
            if (
                expr.startswith('DOC(')
                or expr.startswith('doc_plus(')
                or expr.startswith('doc_cpp_ref(')
                or expr.startswith('R"')
            ):
                try:
                    self._doc_vars[name] = self.resolver.resolve(expr)
                except Exception:
                    pass

    def _ensure_class(
        self,
        name: str,
        *,
        bases: list[str] | None = None,
        doc: str = '',
        parent: str | None = None,
    ) -> ClassStub:
        if parent:
            parent_cls = self.mod.classes.setdefault(parent, ClassStub(name=parent))
            cls = parent_cls.nested.get(name)
            if cls is None:
                cls = ClassStub(name=name, bases=bases or [], doc=doc)
                parent_cls.nested[name] = cls
            else:
                if bases:
                    cls.bases = bases
                if doc and not cls.doc:
                    cls.doc = doc
            return cls
        cls = self.mod.classes.get(name)
        if cls is None:
            cls = ClassStub(name=name, bases=bases or [], doc=doc)
            self.mod.classes[name] = cls
        else:
            if bases:
                cls.bases = bases
            if doc and not cls.doc:
                cls.doc = doc
        return cls

    def _parse_classes(self, text: str) -> None:
        # py::class_<...>(parent_or_m, "Name" [, doc])
        # optionally assigned: py::class_<...> cls(...)
        for m in re.finditer(r'py::class_\s*<', text):
            lt = m.end() - 1  # at '<'
            # find matching '>' for template args (angle-aware, rough)
            depth = 0
            i = lt
            while i < len(text):
                if text[i] == '<':
                    depth += 1
                elif text[i] == '>':
                    depth -= 1
                    if depth == 0:
                        break
                elif text[i] == '"' or (text[i] == 'R' and i + 1 < len(text) and text[i + 1] == '"'):
                    # skip strings inside template (rare)
                    if text[i] == 'R':
                        rm = re.match(r'R"([^\s()]*)\(', text[i:])
                        if rm:
                            delim = rm.group(1)
                            start = i + rm.end()
                            end = text.find(f'){delim}"', start)
                            i = end + len(delim) + 2
                            continue
                    j = i + 1
                    while j < len(text) and not (text[j] == '"' and text[j - 1] != '\\'):
                        j += 1
                    i = j + 1
                    continue
                i += 1
            else:
                continue
            tmpl = text[lt + 1 : i]
            # Assignment form: auto cls = py::class_<...>(...)
            before = text[max(0, m.start() - 100) : m.start()]
            var_name = None
            assign_m = re.search(r'([A-Za-z_][\w]*)\s*=\s*$', before)
            if assign_m:
                var_name = assign_m.group(1)
            # find constructor call
            j = i + 1
            while j < len(text) and text[j].isspace():
                j += 1
            # pattern: > cls(m, "Name"  OR  >(m, "Name"
            if text[j : j + 1] != '(':
                vm = re.match(r'([A-Za-z_][\w]*)\s*', text[j:])
                if vm:
                    if var_name is None:
                        var_name = vm.group(1)
                    j += vm.end()
                while j < len(text) and text[j].isspace():
                    j += 1
            if j >= len(text) or text[j] != '(':
                continue
            close = _matching_paren(text, j)
            args = _split_args(text[j + 1 : close])
            if len(args) < 2:
                continue
            parent_expr = args[0].strip()
            py_name = _python_name_from_string_arg(args[1])
            if not py_name:
                continue
            doc = ''
            if len(args) >= 3:
                doc = self.resolver.resolve(args[2])
            bases = _filter_bases(tmpl)
            parent_py = None
            if parent_expr not in ('m', 'module', 'module_'):
                parent_py = self.mod.handles.get(parent_expr)
                if parent_py is None and parent_expr in self.mod.classes:
                    parent_py = parent_expr
            cls = self._ensure_class(py_name, bases=bases, doc=doc, parent=parent_py)
            if var_name:
                self._bind_handle(
                    var_name,
                    f'{parent_py}.{py_name}' if parent_py else py_name,
                    m.start(),
                )
            # Parse chained .def(...).def(...) until ';'; start after ctor ')'
            self._parse_member_chain_after(text, close + 1, cls)

    def _statement_end(self, text: str, start: int) -> int:
        """Index of ';' ending the statement that begins at ``start``."""
        i = start
        paren = 0
        brace = 0
        while i < len(text):
            nxt = _skip_string_or_char(text, i)
            if nxt != i:
                i = nxt
                continue
            ch = text[i]
            if ch == '(':
                paren += 1
            elif ch == ')':
                paren -= 1
            elif ch == '{':
                brace += 1
            elif ch == '}':
                brace -= 1
            elif ch == ';' and paren <= 0 and brace <= 0:
                return i
            i += 1
        return len(text)

    def _parse_member_chain_after(self, text: str, after_ctor: int, cls: ClassStub) -> None:
        """Parse ``.def(...).def_static(...)`` chain after a class_/enum ctor call."""
        end = self._statement_end(text, after_ctor)
        region = text[after_ctor:end]
        patterns = {
            'def_static': 'staticmethod',
            'def_property_readonly': 'property',
            'def_property': 'property',
            'def_readonly': 'readonly',
            'def_readwrite': 'readwrite',
            'def': 'method',
        }
        for cm in re.finditer(
            r'\.\s*(def_static|def_property_readonly|def_property|'
            r'def_readonly|def_readwrite|def)\s*\(',
            region,
        ):
            kind = patterns[cm.group(1)]
            open_idx = cm.end() - 1
            try:
                close = _matching_paren(region, open_idx)
            except ValueError:
                continue
            args = _split_args(region[open_idx + 1 : close])
            if not args:
                continue
            name = _python_name_from_string_arg(args[0])
            if not name:
                if 'py::init' in args[0]:
                    cls.members.setdefault('__init__', Member('__init__', 'method', ''))
                continue
            self._set_member(cls, name, kind, self._doc_from_def_args(args))

    def _parse_native_enums(self, text: str) -> None:
        for m in re.finditer(r'py::native_enum\s*<[^>]+>\s*', text):
            # optional var
            before = text[max(0, m.start() - 60) : m.start()]
            var_m = re.search(r'([A-Za-z_][\w]*)\s*$', before)
            j = m.end()
            while j < len(text) and text[j].isspace():
                j += 1
            var_name = None
            if text[j] != '(':
                vm = re.match(r'([A-Za-z_][\w]*)\s*', text[j:])
                if vm:
                    var_name = vm.group(1)
                    j += vm.end()
                while j < len(text) and text[j].isspace():
                    j += 1
            if var_m and var_name is None:
                # `py::native_enum<...> fusion_enum(` style already handled
                pass
            if j >= len(text) or text[j] != '(':
                continue
            close = _matching_paren(text, j)
            args = _split_args(text[j + 1 : close])
            if len(args) < 3:
                continue
            py_name = _python_name_from_string_arg(args[1])
            enum_base = _python_name_from_string_arg(args[2]) or 'enum.Enum'
            doc = self.resolver.resolve(args[3]) if len(args) > 3 else ''
            if not py_name:
                continue
            cls = self._ensure_class(py_name, doc=doc)
            cls.is_enum = True
            cls.enum_base = enum_base
            if var_name:
                self._bind_handle(var_name, py_name, m.start())
            hm = re.search(
                r'py::native_enum\s*<[^>]+>\s+([A-Za-z_][\w]*)\s*\(',
                text[m.start() : m.start() + 200],
            )
            if hm:
                self._bind_handle(hm.group(1), py_name, m.start())

        # Separate statements: fusion_enum.value("x", ...).export_values().finalize();
        for m in re.finditer(
            r'([A-Za-z_][\w]*)\s*\.\s*value\s*\(\s*"([^"]+)"',
            text,
        ):
            handle = m.group(1)
            py_name = self.mod.handles.get(handle)
            if not py_name or py_name not in self.mod.classes:
                continue
            cls = self.mod.classes[py_name]
            if not cls.is_enum:
                continue
            end = self._statement_end(text, m.start())
            region = text[m.start() : end]
            for vm in re.finditer(r'\.value\s*\(\s*"([^"]+)"', region):
                val = vm.group(1)
                if val not in cls.enum_values:
                    cls.enum_values.append(val)
            if '.export_values()' in region.replace(' ', ''):
                for val in cls.enum_values:
                    self.mod.constants[val] = val

    def _parse_exceptions(self, text: str) -> None:
        for m in re.finditer(r'py::register_exception\s*<[^>]+>\s*\(', text):
            open_idx = text.index('(', m.start())
            close = _matching_paren(text, open_idx)
            args = _split_args(text[open_idx + 1 : close])
            if len(args) < 2:
                continue
            py_name = _python_name_from_string_arg(args[1])
            if not py_name:
                continue
            base = 'Exception'
            if len(args) >= 3:
                b = args[2].strip()
                if b == 'PyExc_Exception':
                    base = 'Exception'
                elif b in self.mod.classes:
                    base = b
                else:
                    # variable holding previous exception
                    base = self.mod.handles.get(b, 'Exception')
            cls = self._ensure_class(py_name)
            cls.is_exception = True
            cls.exception_base = base
            # doc assignment: name.doc() = ...
            # find nearby .doc() =
            window = text[close : close + 300]
            dm = re.search(r'\.doc\s*\(\s*\)\s*=\s*', window)
            if dm:
                start = close + dm.end()
                # expression until ;
                end = start
                depth = 0
                while end < len(text) and not (text[end] == ';' and depth == 0):
                    if text[end] == '(':
                        depth += 1
                    elif text[end] == ')':
                        depth -= 1
                    end += 1
                cls.doc = self.resolver.resolve(text[start:end].strip())
            # handle: auto& symmetry_error = register_exception...
            before = text[max(0, m.start() - 80) : m.start()]
            bm = re.search(r'([A-Za-z_][\w]*)\s*=\s*$', before.strip())
            # `auto& symmetry_error =\n  py::register_exception`
            bm = re.search(
                r'([A-Za-z_][\w]*)\s*=\s*\n?\s*$',
                text[max(0, m.start() - 100) : m.start()],
            )
            if bm:
                self._bind_handle(bm.group(1), py_name, m.start())

    def _parse_module_defs(self, text: str) -> None:
        for m in re.finditer(r'\bm\.def\s*\(', text):
            close = _matching_paren(text, m.end() - 1)
            args = _split_args(text[m.end() : close])
            if not args:
                continue
            name = _python_name_from_string_arg(args[0])
            if not name:
                continue
            doc = self._doc_from_def_args(args)
            existing = self.mod.functions.get(name)
            if existing is not None and existing.doc and not doc:
                continue
            self.mod.functions[name] = Member(
                name=name, kind='method', doc=doc or (existing.doc if existing else '')
            )

    def _parse_module_attrs(self, text: str) -> None:
        for m in re.finditer(r'\bm\.attr\s*\(\s*"([^"]+)"\s*\)\s*=', text):
            name = m.group(1)
            if name.startswith('_') and name not in ('_valid_block_inds',):
                # keep public-ish
                pass
            self.mod.constants[name] = name

    def _parse_class_doc_assign(self, text: str) -> None:
        # ``cls.doc() = …``, ``(*g_cls).doc() = …``, ``g_cls->doc() = …``
        for m in re.finditer(
            r'(?:\(\s*\*\s*)?([A-Za-z_][\w]*)\s*(?:\)\s*)?(?:\.|->)\s*doc\s*\(\s*\)\s*=\s*',
            text,
        ):
            handle = m.group(1)
            start = m.end()
            end = start
            depth = 0
            while end < len(text) and not (text[end] == ';' and depth == 0):
                if text[end] == '(':
                    depth += 1
                elif text[end] == ')':
                    depth -= 1
                end += 1
            doc = self.resolver.resolve(text[start:end].strip())
            # Plain string literals (not DOC/doc_plus) also appear on declare shells.
            if not doc:
                lit = _parse_string_literal(text[start:end].strip())
                if lit is not None:
                    doc = lit
            py_name = self._handle_at(handle, m.start())
            if py_name and '.' not in py_name and py_name in self.mod.classes:
                if doc:
                    self.mod.classes[py_name].doc = doc
            elif py_name and '.' in py_name:
                parent, child = py_name.split('.', 1)
                nest = self.mod.classes.get(parent)
                if nest and child in nest.nested and doc:
                    nest.nested[child].doc = doc

    def _target_for_handle(self, handle: str, pos: int | None = None) -> ClassStub | None:
        if pos is not None:
            py = self._handle_at(handle, pos)
        else:
            py = self.mod.handles.get(handle)
        if not py:
            if handle in self.mod.classes:
                return self.mod.classes[handle]
            return None
        if '.' in py:
            parent, child = py.split('.', 1)
            p = self.mod.classes.get(parent)
            return p.nested.get(child) if p else None
        return self.mod.classes.get(py)

    def _resolve_cls(self, handle: str, pos: int) -> ClassStub | None:
        cls = self._target_for_handle(handle, pos)
        if cls is not None:
            return cls
        py = self._handle_at(handle, pos)
        if py and '.' not in py:
            return self._ensure_class(py)
        return None

    def _parse_chain_defs(self, text: str) -> None:
        # .def("name", …), .def_static, .def_property_readonly, .def_readonly, .def_readwrite
        # Associated with preceding handle via `cls.def` or chained `.def`
        # Strategy: find `handle.def(` / `handle.def_static(` etc., then also
        # standalone `.def(` that appear in a chain after a known handle start.

        # First: explicit handle.def...
        patterns = [
            ('def_static', 'staticmethod'),
            ('def_property_readonly', 'property'),
            ('def_property', 'property'),
            ('def_readonly', 'readonly'),
            ('def_readwrite', 'readwrite'),
            ('def', 'method'),
        ]
        for m in re.finditer(
            r'([A-Za-z_][\w]*)\s*\.\s*(def_static|def_property_readonly|def_property|'
            r'def_readonly|def_readwrite|def)\s*\(',
            text,
        ):
            if self._in_helper_body(m.start()):
                continue
            handle = m.group(1)
            kind_tok = m.group(2)
            kind = dict(patterns)[kind_tok]
            open_idx = m.end() - 1
            try:
                close = _matching_paren(text, open_idx)
            except ValueError:
                continue
            args = _split_args(text[open_idx + 1 : close])
            if not args:
                continue
            name = _python_name_from_string_arg(args[0])
            if not name or name in ('__enter__', '__exit__'):
                # still include dunders that autodoc might want; keep them
                if not name:
                    # py::init — skip
                    if 'py::init' in args[0] or args[0].strip().startswith('py::init'):
                        cls = self._resolve_cls(handle, m.start())
                        if cls and '__init__' not in cls.members:
                            cls.members['__init__'] = Member('__init__', 'method', '')
                    continue
            cls = self._resolve_cls(handle, m.start())
            if cls is None:
                continue
            if name.startswith('__') and name not in (
                '__init__',
                '__repr__',
                '__str__',
                '__len__',
                '__getitem__',
                '__setitem__',
                '__contains__',
                '__iter__',
                '__next__',
                '__call__',
                '__eq__',
                '__ne__',
                '__add__',
                '__sub__',
                '__mul__',
                '__rmul__',
                '__truediv__',
                '__neg__',
            ):
                pass
            self._set_member(cls, name, kind, self._doc_from_def_args(args))

        # Chained .def( without handle — walk from known handle starts
        for m in re.finditer(
            r'([A-Za-z_][\w]*)\s*(?:\.def|\.def_static|\.def_property|'
            r'\.def_property_readonly|\.def_readonly|\.def_readwrite)',
            text,
        ):
            if self._in_helper_body(m.start()):
                continue
            handle = m.group(1)
            cls = self._resolve_cls(handle, m.start())
            if cls is None:
                continue
            # scan forward for chained .def( until semicolon at depth 0... hard.
            # Already covered by handle.def above for first; for chains like
            #   cls.def(...).def(...).def(...)
            # the regex above only catches the first. Handle chains:
            region_start = m.start()
            end = self._statement_end(text, region_start)
            region = text[region_start:end]
            for cm in re.finditer(
                r'\.\s*(def_static|def_property_readonly|def_property|'
                r'def_readonly|def_readwrite|def)\s*\(',
                region,
            ):
                kind = dict(patterns)[cm.group(1)]
                open_idx = cm.end() - 1
                try:
                    close = _matching_paren(region, open_idx)
                except ValueError:
                    continue
                args = _split_args(region[open_idx + 1 : close])
                if not args:
                    continue
                name = _python_name_from_string_arg(args[0])
                if not name:
                    if 'py::init' in args[0]:
                        cls.members.setdefault('__init__', Member('__init__', 'method', ''))
                    continue
                self._set_member(cls, name, kind, self._doc_from_def_args(args))

    def _parse_enum_post_attrs(self, text: str) -> None:
        # Handles like: py::object tensor_cls = m.attr("Tensor");
        for m in re.finditer(
            r'([A-Za-z_][\w]*)\s*=\s*m\.attr\s*\(\s*"([^"]+)"\s*\)',
            text,
        ):
            self._bind_handle(m.group(1), m.group(2), m.start())

        # D.attr("name") = ... after native_enum finalize; also Class.attr("method")
        for m in re.finditer(
            r'([A-Za-z_][\w]*)\.attr\s*\(\s*"([^"]+)"\s*\)\s*=',
            text,
        ):
            handle, name = m.group(1), m.group(2)
            cls = self._resolve_cls(handle, m.start())
            if cls is None:
                continue
            kind = 'method'
            if name in ('from_hdf5', 'from_numpy_dtype'):
                kind = 'classmethod'
            elif name in (
                'is_real',
                'is_complex',
                'to_complex',
                'to_real',
                'python_type',
                'zero_scalar',
                'one_scalar',
                'eps',
            ):
                kind = 'property'
            start = m.end()
            end = self._statement_end(text, start)
            expr = text[start:end]
            doc = ''
            for dm in re.finditer(
                r'(?:DOC|doc_plus|doc_cpp_ref)\s*\(',
                expr,
            ):
                try:
                    open_idx = dm.end() - 1
                    close = _matching_paren(expr, open_idx)
                    doc = self.resolver.resolve(expr[dm.start() : close + 1])
                except Exception:
                    pass
            if not doc:
                for dm in re.finditer(r'R"pydoc\((.*?)\)pydoc"', expr, re.DOTALL):
                    doc = dm.group(1)
            if not doc:
                # trailing local like slice_leg_py_doc
                idents = re.findall(r'\b([A-Za-z_][\w]*)\b', expr)
                for ident in reversed(idents):
                    if ident in self._doc_vars:
                        doc = self._doc_vars[ident]
                        break
            cls.members[name] = Member(name=name, kind=kind, doc=doc)

        # as_property("name", ...) helper used in py_dtypes.cpp
        for m in re.finditer(r'\bas_property\s*\(\s*"([^"]+)"', text):
            cls = self.mod.classes.get('Dtype')
            if cls is None:
                continue
            cls.members[m.group(1)] = Member(m.group(1), 'property', '')

        # Identity unsupported factories: bind_unsupported("from_zero");
        for m in re.finditer(r'\bbind_unsupported\s*\(\s*"([^"]+)"\s*\)', text):
            cls = self.mod.classes.get('Identity')
            if cls is None:
                continue
            cls.members[m.group(1)] = Member(m.group(1), 'staticmethod', '')

    def _parse_class_bind_helpers(self, text: str) -> None:
        """Find ``void bind_*(Class& param) { param.def... }`` helpers.

        Used for shared axis bindings such as ``bind_axis_common`` that are
        invoked once per pybind class handle. Members are collected from the
        helper body and later attached at each call site.
        """
        for m in re.finditer(
            r'(?:template\s*<[^;{]*?>\s*)?void\s+([A-Za-z_][\w]*)\s*\(',
            text,
        ):
            func_name = m.group(1)
            open_paren = m.end() - 1
            try:
                close_paren = _matching_paren(text, open_paren)
            except ValueError:
                continue
            params = text[open_paren + 1 : close_paren]
            # Require a Class& / py::class_& parameter (the bind target).
            param_m = re.search(
                r'(?:Class|py::class_\s*<[^>]+>)\s*&\s*([A-Za-z_][\w]*)',
                params,
            )
            if param_m is None:
                continue
            param = param_m.group(1)
            j = close_paren + 1
            while j < len(text) and text[j].isspace():
                j += 1
            if j >= len(text) or text[j] != '{':
                continue
            try:
                close_brace = _matching_brace(text, j)
            except ValueError:
                continue
            body_start, body_end = j + 1, close_brace
            members = self._collect_handle_chain_members(
                text, param, body_start, body_end
            )
            if not members:
                continue
            self._bind_helpers[func_name] = members
            self._helper_body_ranges.append((body_start, body_end))

    def _collect_handle_chain_members(
        self, text: str, handle: str, start: int, end: int
    ) -> list[Member]:
        """Parse ``handle.def...`` / chained ``.def`` members inside ``text[start:end]``."""
        region = text[start:end]
        patterns = {
            'def_static': 'staticmethod',
            'def_property_readonly': 'property',
            'def_property': 'property',
            'def_readonly': 'readonly',
            'def_readwrite': 'readwrite',
            'def': 'method',
        }
        collected: dict[str, Member] = {}

        def _record(name: str, kind: str, doc: str) -> None:
            existing = collected.get(name)
            if existing is not None and existing.doc and not doc:
                return
            collected[name] = Member(
                name=name,
                kind=kind,
                doc=doc or (existing.doc if existing else ''),
            )

        for m in re.finditer(
            rf'{re.escape(handle)}\s*(?:\.def|\.def_static|\.def_property|'
            rf'\.def_property_readonly|\.def_readonly|\.def_readwrite)',
            region,
        ):
            stmt_start = start + m.start()
            stmt_end = self._statement_end(text, stmt_start)
            stmt = text[stmt_start:stmt_end]
            for cm in re.finditer(
                r'\.\s*(def_static|def_property_readonly|def_property|'
                r'def_readonly|def_readwrite|def)\s*\(',
                stmt,
            ):
                kind = patterns[cm.group(1)]
                open_idx = cm.end() - 1
                try:
                    close = _matching_paren(stmt, open_idx)
                except ValueError:
                    continue
                args = _split_args(stmt[open_idx + 1 : close])
                if not args:
                    continue
                name = _python_name_from_string_arg(args[0])
                if not name:
                    if 'py::init' in args[0]:
                        _record('__init__', 'method', '')
                    continue
                _record(name, kind, self._doc_from_def_args(args))
        return list(collected.values())

    def _apply_bind_helper_calls(self, text: str) -> None:
        """Attach helper members to each ``bind_foo<...>(handle)`` call-site class."""
        if not self._bind_helpers:
            return
        for func_name, members in self._bind_helpers.items():
            for m in re.finditer(
                rf'\b{re.escape(func_name)}\s*(?:<[^;{{]*?>)?\s*\(',
                text,
            ):
                if self._in_helper_body(m.start()):
                    continue
                open_idx = m.end() - 1
                try:
                    close = _matching_paren(text, open_idx)
                except ValueError:
                    continue
                args = _split_args(text[open_idx + 1 : close])
                if not args:
                    continue
                handle = args[0].strip()
                # Skip the function definition itself (``Class& cls``).
                if '&' in handle or re.match(r'^(?:Class|py::class_)', handle):
                    continue
                # Bare identifier only (the pybind class handle variable).
                if not re.fullmatch(r'[A-Za-z_][\w]*', handle):
                    continue
                cls = self._resolve_cls(handle, m.start())
                if cls is None:
                    continue
                for mem in members:
                    self._set_member(cls, mem.name, mem.kind, mem.doc)

    def _parse_template_call_sites(self, text: str) -> None:
        # bind_sparse_mapping<...>(m, "Name", doc)
        for m in re.finditer(r'bind_sparse_mapping\s*(?:<[^;]*?>)?\s*\(', text):
            open_idx = text.index('(', m.start())
            try:
                close = _matching_paren(text, open_idx)
            except ValueError:
                continue
            args = _split_args(text[open_idx + 1 : close])
            if len(args) < 2:
                continue
            py_name = _python_name_from_string_arg(args[1])
            if not py_name:
                continue
            doc = self.resolver.resolve(args[2]) if len(args) > 2 else ''
            cls = self._ensure_class(py_name, doc=doc)
            sparse_docs = {
                'from_identity': doc_cpp_ref(
                    'from_identity', 'lasyten::Mapping::from_identity()'
                ),
                'pre_compose': doc_cpp_ref('pre_compose', 'lasyten::Mapping::pre_compose()'),
                'nonzero_rows': doc_cpp_ref(
                    'nonzero_rows', 'lasyten::Mapping::nonzero_rows()'
                ),
                'nonzero_cols': doc_cpp_ref(
                    'nonzero_cols', 'lasyten::Mapping::nonzero_cols()'
                ),
                'prune': doc_cpp_ref('prune', 'lasyten::Mapping::prune()'),
            }
            for meth, kind in (
                ('from_identity', 'staticmethod'),
                ('pre_compose', 'method'),
                ('nonzero_rows', 'method'),
                ('nonzero_cols', 'method'),
                ('prune', 'method'),
                ('items', 'method'),
                ('keys', 'method'),
                ('values', 'method'),
                ('data', 'readwrite'),
                ('__init__', 'method'),
                ('__len__', 'method'),
                ('__contains__', 'method'),
                ('__getitem__', 'method'),
                ('__setitem__', 'method'),
            ):
                cls.members.setdefault(
                    meth, Member(meth, kind, sparse_docs.get(meth, ''))
                )
        for m in re.finditer(r'bind_identity_mapping\s*(?:<[^;]*?>)?\s*\(', text):
            open_idx = text.index('(', m.start())
            try:
                close = _matching_paren(text, open_idx)
            except ValueError:
                continue
            args = _split_args(text[open_idx + 1 : close])
            if len(args) < 2:
                continue
            py_name = _python_name_from_string_arg(args[1])
            if not py_name:
                continue
            cls = self._ensure_class(
                py_name, doc=doc_cpp_ref('IdMapping', 'lasyten::IdMapping')
            )
            id_docs = {
                'pre_compose': doc_cpp_ref(
                    'pre_compose', 'lasyten::IdMapping::pre_compose()'
                ),
                'prune': doc_cpp_ref('prune', 'lasyten::IdMapping::prune()'),
            }
            for meth, kind in (
                ('__init__', 'method'),
                ('keys', 'readwrite'),
                ('pre_compose', 'method'),
                ('nonzero_rows', 'method'),
                ('nonzero_cols', 'method'),
                ('prune', 'method'),
            ):
                cls.members.setdefault(meth, Member(meth, kind, id_docs.get(meth, '')))


# ---------------------------------------------------------------------------
# Codegen
# ---------------------------------------------------------------------------


def _py_str(doc: str) -> str:
    """Format a docstring as a Python string literal (safe escapes)."""
    if not doc:
        return '""'
    return repr(doc)


def _stub_body_for_dunder(name: str) -> str | None:
    """Return a safe stub body for special methods that must return a typed value."""
    bodies = {
        '__repr__': 'return f"{self.__class__.__qualname__}(...)"',
        '__str__': 'return f"{self.__class__.__qualname__}(...)"',
        '__bytes__': 'return b""',
        '__format__': 'return ""',
        '__bool__': 'return False',
        '__len__': 'return 0',
        '__hash__': 'return 0',
        '__index__': 'return 0',
        '__int__': 'return 0',
        '__float__': 'return 0.0',
        '__complex__': 'return 0j',
        '__length_hint__': 'return 0',
        '__iter__': 'return iter(())',
        '__reversed__': 'return iter(())',
        '__contains__': 'return False',
        '__eq__': 'return NotImplemented',
        '__ne__': 'return NotImplemented',
    }
    return bodies.get(name)


def _emit_member(lines: list[str], mem: Member, indent: str) -> None:
    doc = _py_str(mem.doc)
    stub_body = _stub_body_for_dunder(mem.name)
    if mem.kind == 'property' or mem.kind == 'readonly':
        lines.append(f'{indent}@property')
        lines.append(f'{indent}def {mem.name}(self):')
        lines.append(f'{indent}    {doc}')
        lines.append(f'{indent}    ...')
        lines.append('')
    elif mem.kind == 'readwrite':
        lines.append(f'{indent}@property')
        lines.append(f'{indent}def {mem.name}(self):')
        lines.append(f'{indent}    {doc}')
        lines.append(f'{indent}    ...')
        lines.append('')
        lines.append(f'{indent}@{mem.name}.setter')
        lines.append(f'{indent}def {mem.name}(self, value):')
        lines.append(f'{indent}    ...')
        lines.append('')
    elif mem.kind == 'staticmethod':
        lines.append(f'{indent}@staticmethod')
        lines.append(f'{indent}def {mem.name}(*args, **kwargs):')
        lines.append(f'{indent}    {doc}')
        lines.append(f'{indent}    {stub_body or "..."}')
        lines.append('')
    elif mem.kind == 'classmethod':
        lines.append(f'{indent}@classmethod')
        lines.append(f'{indent}def {mem.name}(cls, *args, **kwargs):')
        lines.append(f'{indent}    {doc}')
        lines.append(f'{indent}    {stub_body or "..."}')
        lines.append('')
    else:
        if mem.name == '__init__':
            lines.append(f'{indent}def __init__(self, *args, **kwargs):')
            lines.append(f'{indent}    {doc}')
            lines.append(f'{indent}    ...')
        else:
            lines.append(f'{indent}def {mem.name}(self, *args, **kwargs):')
            lines.append(f'{indent}    {doc}')
            lines.append(f'{indent}    {stub_body or "..."}')
        lines.append('')


def _emit_class(lines: list[str], cls: ClassStub, indent: str = '') -> None:
    if cls.is_enum:
        base = cls.enum_base
        if base.startswith('enum.'):
            lines.append(f'{indent}class {cls.name}({base}):')
        else:
            lines.append(f'{indent}class {cls.name}({base}):')
        lines.append(f'{indent}    {_py_str(cls.doc)}')
        if not cls.enum_values and not cls.members:
            lines.append(f'{indent}    ...')
        for i, val in enumerate(cls.enum_values):
            lines.append(f'{indent}    {val} = {i}')
        for mem in cls.members.values():
            if mem.name in cls.enum_values:
                continue
            _emit_member(lines, mem, indent + '    ')
        lines.append('')
        return

    if cls.is_exception:
        bases = cls.exception_base
        lines.append(f'{indent}class {cls.name}({bases}):')
        lines.append(f'{indent}    {_py_str(cls.doc)}')
        lines.append(f'{indent}    ...')
        lines.append('')
        return

    bases = ', '.join(cls.bases) if cls.bases else ''
    if bases:
        lines.append(f'{indent}class {cls.name}({bases}):')
    else:
        lines.append(f'{indent}class {cls.name}:')
    lines.append(f'{indent}    {_py_str(cls.doc)}')
    if not cls.members and not cls.nested:
        lines.append(f'{indent}    ...')
        lines.append('')
        return
    # nested classes first
    for nested in cls.nested.values():
        _emit_class(lines, nested, indent + '    ')
    # ensure __init__ exists for import-time friendliness (optional)
    if '__init__' not in cls.members:
        lines.append(f'{indent}    def __init__(self, *args, **kwargs):')
        lines.append(f'{indent}        ...')
        lines.append('')
    for mem in cls.members.values():
        _emit_member(lines, mem, indent + '    ')
    lines.append('')


def _topo_sort_classes(classes: dict[str, ClassStub]) -> list[ClassStub]:
    """Order classes so bases appear before subclasses when possible."""
    remaining = dict(classes)
    ordered: list[ClassStub] = []
    seen: set[str] = set()

    def ready(cls: ClassStub) -> bool:
        for b in cls.bases:
            if b in remaining and b not in seen:
                return False
            if cls.is_exception and cls.exception_base in remaining and cls.exception_base not in seen:
                return False
        return True

    while remaining:
        progress = False
        for name in list(remaining):
            cls = remaining[name]
            if ready(cls):
                ordered.append(cls)
                seen.add(name)
                del remaining[name]
                progress = True
        if not progress:
            ordered.extend(remaining.values())
            break
    return ordered


def emit_module(mod: ModuleStub) -> str:
    lines: list[str] = [
        '# This file is generated by scripts/generate_core_stubs.py — do not edit.',
        '# Bag-of-names stub for Sphinx autodoc when the compiled lasyten._core',
        '# extension is not available (e.g. Read the Docs).',
        '"""' + mod.doc + '"""',
        '',
        'from __future__ import annotations',
        '',
        'import enum',
        '',
    ]
    for cls in _topo_sort_classes(mod.classes):
        _emit_class(lines, cls)

    for fn in sorted(mod.functions.values(), key=lambda m: m.name):
        lines.append(f'def {fn.name}(*args, **kwargs):')
        lines.append(f'    {_py_str(fn.doc)}')
        lines.append('    ...')
        lines.append('')

    # Constants — dummy values
    for name in sorted(mod.constants):
        if name in mod.functions or name in mod.classes:
            continue
        if name in ('CONTRACT_SYMBOL', 'LEG_SELECT_SYMBOL', 'OPEN_LEG_SYMBOL'):
            lines.append(f'{name} = "*"')
        elif name == 'FORBIDDEN_LEG_LABEL_CHARS':
            lines.append(f'{name} = ""')
        elif name == 'ALL_SPECIES':
            lines.append(f'{name} = None')
        else:
            lines.append(f'{name} = None')
    if mod.constants:
        lines.append('')

    # Import-time helpers used by lasyten package
    if 'get_config' not in mod.functions:
        lines.append('def get_config(*args, **kwargs):')
        lines.append('    """Get the global configuration object."""')
        lines.append('    return None')
        lines.append('')

    return '\n'.join(lines).rstrip() + '\n'


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def load_doc_map(
    source_root: Path,
    *,
    from_doxygen: bool,
    doxygen: str,
) -> dict[str, str]:
    headers_dir = source_root / 'pybind' / 'docstrings'
    cmake_xml = source_root / 'build' / 'docstrings_doxygen'
    if not from_doxygen and headers_dir.is_dir() and any(headers_dir.rglob('*.h')):
        return parse_docstring_headers(headers_dir)
    reuse = cmake_xml if cmake_xml.is_dir() else None
    return build_doc_map(
        source_root,
        work_dir=source_root / 'build' / 'stub_docstrings_doxygen',
        doxygen=doxygen,
        reuse_cmake_xml=reuse if not from_doxygen else None,
    )


def generate_stub(
    source_root: Path,
    *,
    from_doxygen: bool = False,
    doxygen: str = 'doxygen',
    strict_docs: bool = True,
) -> str:
    doc_map = load_doc_map(source_root, from_doxygen=from_doxygen, doxygen=doxygen)
    resolver = DocResolver(doc_map, strict=strict_docs)
    parser = BindingParser(resolver)
    pybind = source_root / 'pybind'
    files = sorted(pybind.rglob('*.cpp'))
    # Include .cpp files that are #included into others (dtypes etc.) — still parse
    for path in files:
        parser.parse_file(path)
    # Also parse included .cpp fragments if any live as .cpp under pybind
    # (py_dtypes.cpp is included from py_block_backend.cpp — already in files)

    # Ensure dummy import-time attributes on Dtype / BlockBackend
    if 'Dtype' in parser.mod.classes:
        dt = parser.mod.classes['Dtype']
        dt.is_enum = True
        if not dt.enum_values:
            dt.enum_values = [
                'bool',
                'float32',
                'complex64',
                'float64',
                'complex128',
                'int64',
            ]
    if 'BlockBackend' in parser.mod.classes:
        bb = parser.mod.classes['BlockBackend']
        if 'BlockCls' not in bb.nested:
            bb.nested['BlockCls'] = ClassStub(name='BlockCls', doc='Abstract base for dense blocks.')
        if 'Scalar' not in bb.nested:
            bb.nested['Scalar'] = ClassStub(name='Scalar', doc='Scalar value with Dtype.')

    if resolver.missing and strict_docs:
        raise SystemExit(
            'missing DOC() keys:\n  ' + '\n  '.join(resolver.missing[:50])
            + (f'\n  ... ({len(resolver.missing)} total)' if len(resolver.missing) > 50 else '')
        )
    return emit_module(parser.mod)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        '-o',
        '--output',
        type=Path,
        default=None,
        help='Output path (default: <repo>/lasyten/_core.py)',
    )
    ap.add_argument(
        '--from-doxygen',
        action='store_true',
        help='Rebuild DOC map via scoped Doxygen instead of pybind/docstrings/',
    )
    ap.add_argument('--doxygen', default='doxygen', help='doxygen executable')
    ap.add_argument(
        '--allow-missing-docs',
        action='store_true',
        help='Do not fail when a DOC() key is missing from the map',
    )
    ap.add_argument(
        '--source-root',
        type=Path,
        default=_REPO,
        help='Repository root',
    )
    args = ap.parse_args()
    source_root = args.source_root.resolve()
    text = generate_stub(
        source_root,
        from_doxygen=args.from_doxygen,
        doxygen=args.doxygen,
        strict_docs=not args.allow_missing_docs,
    )
    out = args.output or (source_root / 'lasyten' / '_core.py')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding='utf-8')
    print(f'wrote {out} ({len(text)} bytes)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
