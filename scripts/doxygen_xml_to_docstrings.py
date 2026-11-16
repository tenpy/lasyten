#!/usr/bin/env python3
"""Generate pybind ``DOC()`` headers from Doxygen XML + source ``///`` comments.

Doxygen discovers symbols (names, overload order, namespaces). Comment bodies are
read from the C++ header (Doxygen ``@param`` / ``@returns`` + markdown) and
converted to NumPy-style sections for Sphinx Napoleon / Python ``__doc__``.
A trailing ``.. lasyten-cpp-ref::`` marker is appended so Sphinx can inject a
compact ``[C++]`` signature badge linking to the matching Breathe page.

Example::

    doxygen <scoped-Doxyfile>
    python scripts/doxygen_xml_to_docstrings.py \\
        --xml-dir build/docstrings_xml/xml \\
        --header-rel tensors/ops_algebra.h \\
        --source-root . \\
        -o pybind/docstrings/tensors/ops_algebra.h
"""

from __future__ import annotations

import argparse
import re
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

# @param[in] name1,name2 description…  /  @returns description…
_CMD_RE = re.compile(
    r'^@(?P<cmd>param|paramref|returns?|return|brief)\s*'
    r'(?:\[(?P<dir>[^\]]*)\]\s*)?'
    r'(?P<rest>.*)$'
)

_DOC_PREAMBLE = """\
#ifndef LASYTEN_MKDOC_DOC_MACROS
#define LASYTEN_MKDOC_DOC_MACROS
#define MKD_EXPAND(x)                                      x
#define MKD_COUNT(_1, _2, _3, _4, _5, _6, _7, COUNT, ...)  COUNT
#define MKD_VA_SIZE(...)                                   MKD_EXPAND(MKD_COUNT(__VA_ARGS__, 7, 6, 5, 4, 3, 2, 1, 0))
#define MKD_CAT1(a, b)                                     a##b
#define MKD_CAT2(a, b)                                     MKD_CAT1(a, b)
#define MKD_DOC1(n1)                                       mkd_doc_##n1
#define MKD_DOC2(n1, n2)                                   mkd_doc_##n1##_##n2
#define MKD_DOC3(n1, n2, n3)                               mkd_doc_##n1##_##n2##_##n3
#define MKD_DOC4(n1, n2, n3, n4)                           mkd_doc_##n1##_##n2##_##n3##_##n4
#define MKD_DOC5(n1, n2, n3, n4, n5)                       mkd_doc_##n1##_##n2##_##n3##_##n4##_##n5
#define MKD_DOC6(n1, n2, n3, n4, n5, n6)                   mkd_doc_##n1##_##n2##_##n3##_##n4##_##n5##_##n6
#define MKD_DOC7(n1, n2, n3, n4, n5, n6, n7)               mkd_doc_##n1##_##n2##_##n3##_##n4##_##n5##_##n6##_##n7
#define DOC(...)                                           MKD_EXPAND(MKD_EXPAND(MKD_CAT2(MKD_DOC, MKD_VA_SIZE(__VA_ARGS__)))(__VA_ARGS__))
#endif /* LASYTEN_MKDOC_DOC_MACROS */

#if defined(__GNUG__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
#endif

"""

_BANNER = """\
/*
  Generated from Doxygen XML + include/lasyten/{source_rel} /// comments.
  Not committed — produced at build time (requires doxygen).

  cmake --build <build-dir> --target lasyten_generate_docstrings
 */

"""

# Kinds we emit as DOC() entries (not typedefs / variables / enums alone).
_EMIT_KINDS = frozenset({'function', 'class', 'struct', 'enum'})


@dataclass(frozen=True)
class Symbol:
    kind: str
    name: str
    namespace_parts: tuple[str, ...]
    line: int
    file: str
    has_docs: bool
    # Comma-separated parameter *types* (no names/defaults), for overload links.
    param_types: str = ''


def _local_tag(tag: str | None) -> str:
    if tag is None:
        return ''
    if '}' in tag:
        return tag.rsplit('}', 1)[-1]
    return tag


def _text_nonempty(elem: ET.Element | None) -> bool:
    if elem is None:
        return False
    return bool(''.join(elem.itertext()).strip())


def _compound_namespace(compound: ET.Element) -> tuple[str, ...]:
    kind = compound.get('kind', '')
    name = (compound.findtext('compoundname') or '').strip()
    if kind == 'namespace' and name:
        return tuple(p for p in name.split('::') if p)
    if kind in ('class', 'struct') and name:
        return tuple(p for p in name.split('::') if p)
    return ()


def _file_matches(location_file: str, header_rel: str) -> bool:
    loc = location_file.replace('\\', '/')
    rel = header_rel.replace('\\', '/')
    return loc == rel or loc.endswith('/' + rel) or loc.endswith(rel)


def _param_types(member: ET.Element) -> str:
    """Return ``Type1, Type2, …`` from a ``memberdef`` (no names/defaults)."""
    types: list[str] = []
    for child in member:
        if _local_tag(child.tag) != 'param':
            continue
        type_el = None
        for sub in child:
            if _local_tag(sub.tag) == 'type':
                type_el = sub
                break
        if type_el is None:
            continue
        t = re.sub(r'\s+', ' ', ''.join(type_el.itertext()).strip())
        if t:
            types.append(t)
    return ', '.join(types)


def collect_symbols(xml_dir: Path, header_rel: str) -> list[Symbol]:
    symbols: list[Symbol] = []
    for path in sorted(xml_dir.glob('*.xml')):
        if path.name in {'index.xml', 'Doxyfile.xml'} or path.name.endswith('.xsd'):
            continue
        try:
            root = ET.parse(path).getroot()
        except ET.ParseError:
            continue
        for compound in root.iter():
            if _local_tag(compound.tag) != 'compounddef':
                continue
            ns_parts = _compound_namespace(compound)
            compound_kind = compound.get('kind', '')
            # Class/struct documented on the compound itself.
            if compound_kind in ('class', 'struct'):
                loc = None
                for child in compound:
                    if _local_tag(child.tag) == 'location':
                        loc = child
                        break
                if loc is not None and _file_matches(loc.get('file', ''), header_rel):
                    brief = None
                    detailed = None
                    for child in compound:
                        t = _local_tag(child.tag)
                        if t == 'briefdescription':
                            brief = child
                        elif t == 'detaileddescription':
                            detailed = child
                    qname = ns_parts
                    if qname:
                        symbols.append(
                            Symbol(
                                kind=compound_kind,
                                name=qname[-1],
                                namespace_parts=qname[:-1],
                                line=int(loc.get('line', '0') or 0),
                                file=loc.get('file', ''),
                                has_docs=_text_nonempty(brief) or _text_nonempty(detailed),
                            )
                        )
            for md in compound.iter():
                if _local_tag(md.tag) != 'memberdef':
                    continue
                kind = md.get('kind', '')
                if kind not in _EMIT_KINDS:
                    continue
                loc = None
                name = None
                brief = None
                detailed = None
                for child in md:
                    t = _local_tag(child.tag)
                    if t == 'location':
                        loc = child
                    elif t == 'name':
                        name = (child.text or '').strip()
                    elif t == 'briefdescription':
                        brief = child
                    elif t == 'detaileddescription':
                        detailed = child
                if loc is None or not name:
                    continue
                if not _file_matches(loc.get('file', ''), header_rel):
                    continue
                member_ns = ns_parts
                # Nested class members: compound is the class; keep full path.
                if compound_kind in ('class', 'struct') and ns_parts:
                    # methods: DOC(ns, Class, method) → namespace_parts = ns + Class
                    pass
                symbols.append(
                    Symbol(
                        kind=kind,
                        name=name,
                        namespace_parts=member_ns,
                        line=int(loc.get('line', '0') or 0),
                        file=loc.get('file', ''),
                        has_docs=_text_nonempty(brief) or _text_nonempty(detailed),
                        param_types=_param_types(md) if kind == 'function' else '',
                    )
                )
    # Dedupe identical (kind, path, name, line)
    uniq: dict[tuple, Symbol] = {}
    for s in symbols:
        key = (s.kind, s.namespace_parts, s.name, s.line)
        uniq[key] = s
    return sorted(uniq.values(), key=lambda s: (s.line, s.name))


def _is_decl_prefix_line(stripped: str) -> bool:
    """True for lines Doxygen does not count as the member location.

    Doxygen's ``<location line>`` is typically the line of the *name*. Return
    types, ``template <...>``, and ``[[nodiscard]]`` then sit between the
    ``///`` block and that line, e.g.::

        /// docs
        virtual std::tuple<A, B>
        foo();
    """
    if stripped == '' or stripped.startswith('[['):
        return True
    if stripped.startswith('///') or stripped.startswith('//'):
        return False
    if stripped in {'public:', 'private:', 'protected:'}:
        return False
    if stripped.endswith(';') or stripped.endswith('{') or stripped.endswith('}'):
        return False
    # Another declarator (function, macro) — do not walk through it.
    if '(' in stripped:
        return False
    return True


def extract_slash_comment(source_lines: list[str], decl_line: int) -> str | None:
    """Return the ``///`` block immediately above 1-based ``decl_line``, or None."""
    if decl_line < 1 or decl_line > len(source_lines):
        return None
    # Skip attribute / return-type / empty lines directly above the name line.
    i = decl_line - 2  # 0-based index of line above decl
    while i >= 0 and _is_decl_prefix_line(source_lines[i].strip()):
        i -= 1
    if i < 0 or not source_lines[i].lstrip().startswith('///'):
        return None
    block: list[str] = []
    while i >= 0 and source_lines[i].lstrip().startswith('///'):
        line = source_lines[i].lstrip()
        if line.startswith('/// '):
            block.append(line[4:])
        elif line.startswith('///'):
            block.append(line[3:])
        else:
            break
        i -= 1
    block.reverse()
    # Trim trailing blank comment lines
    while block and block[-1].strip() == '':
        block.pop()
    text = '\n'.join(block).rstrip() + '\n'
    return text if text.strip() else None


def _split_param_rest(rest: str) -> tuple[str, str]:
    """Split ``name[,name…] description`` after ``@param``."""
    rest = rest.strip()
    if not rest:
        return '', ''
    # First token is the name list (may contain commas, no spaces inside names).
    parts = rest.split(None, 1)
    names = parts[0]
    desc = parts[1] if len(parts) > 1 else ''
    return names, desc


def doxygen_comment_to_numpy(doc: str) -> str:
    """Convert Doxygen ``@param`` / ``@returns`` (+ markdown) to NumPy sections.

    Prose before the first command is kept as the summary/body. Markdown
    backticks are upgraded to reST double-backticks for Napoleon/Sphinx.
    """
    prose: list[str] = []
    params: list[tuple[str, list[str]]] = []
    returns: list[str] = []
    mode: str | None = None  # None | 'param' | 'returns'
    cur_names = ''
    cur_lines: list[str] = []

    def flush() -> None:
        nonlocal cur_names, cur_lines, mode
        if mode == 'param' and cur_names:
            # ``a,b`` → ``a, b`` for NumPy readability
            display = ', '.join(p.strip() for p in cur_names.split(',') if p.strip())
            params.append((display, list(cur_lines)))
        elif mode == 'returns' and cur_lines:
            returns.extend(cur_lines)
        cur_names = ''
        cur_lines = []
        mode = None

    for raw in doc.splitlines():
        # Normalize backslash commands (\param → @param, \returns → @returns).
        line = raw
        for old, new in (
            ('\\paramref', '@paramref'),
            ('\\param', '@param'),
            ('\\returns', '@returns'),
            ('\\return', '@return'),
            ('\\brief', '@brief'),
        ):
            if line.lstrip().startswith(old):
                line = line.replace(old, new, 1)
                break
        stripped = line.strip()
        m = _CMD_RE.match(stripped) if stripped.startswith('@') else None
        if m:
            cmd = m.group('cmd')
            rest = m.group('rest') or ''
            if cmd in ('param', 'paramref'):
                flush()
                names, desc = _split_param_rest(rest)
                mode = 'param'
                cur_names = names
                cur_lines = [desc] if desc else []
            elif cmd in ('return', 'returns'):
                flush()
                mode = 'returns'
                cur_lines = [rest] if rest else []
            elif cmd == 'brief':
                flush()
                if rest:
                    prose.append(rest)
                mode = None
            continue

        if mode == 'param':
            if stripped == '':
                if cur_lines:
                    cur_lines.append('')
            else:
                cur_lines.append(line.rstrip())
            continue
        if mode == 'returns':
            if stripped == '':
                if cur_lines:
                    cur_lines.append('')
            else:
                cur_lines.append(line.rstrip())
            continue

        prose.append(raw)

    flush()

    # Drop trailing blanks in prose
    while prose and prose[-1].strip() == '':
        prose.pop()

    out: list[str] = []
    if prose:
        out.extend(prose)
    if params:
        if out:
            out.append('')
        out.append('Parameters')
        out.append('----------')
        for names, desc_lines in params:
            desc_lines = _keep_literal_block_indent(desc_lines)
            # Trim trailing empty continuation lines
            while desc_lines and desc_lines[-1].strip() == '':
                desc_lines.pop()
            if not desc_lines:
                out.append(names)
                continue
            first, *rest = desc_lines
            out.append(f'{names}')
            if first:
                out.append(f'    {first}')
            for cont in rest:
                if cont.strip() == '':
                    out.append('')
                else:
                    out.append(f'    {cont}')
    if returns:
        returns = _keep_literal_block_indent(returns)
        while returns and returns[-1].strip() == '':
            returns.pop()
        if out:
            out.append('')
        out.append('Returns')
        out.append('-------')
        for line in returns:
            out.append(line if line.strip() == '' else line)

    text = '\n'.join(out).rstrip() + '\n'
    # Doxygen display math → Sphinx ``.. math::`` (before inline, so ``\f[`` is not
    # confused with ``\f$``).
    text = _display_math_to_sphinx(text)
    # Markdown `code` → reST ``code``. Leave Sphinx roles (:math:`, :meth:`, …) alone.
    text = _markdown_ticks_to_rest(text)
    # RST: inline literals cannot span lines; ``Foo``s is not a valid end-string.
    text = _join_wrapped_inline_literals(text)
    text = re.sub(r'``(\w+)``s\b', r'``\1``\\ s', text)
    # Doxygen inline math → Sphinx math role (after backtick upgrade).
    text = _inline_math_to_sphinx(text)
    return text


_DISPLAY_MATH_RE = re.compile(r'(?:@f\[|\\f\[)(.*?)(?:@f\]|\\f\])', re.DOTALL)
_INLINE_MATH_RE = re.compile(r'(?:@f\$|\\f\$)(.*?)(?:@f\$|\\f\$)', re.DOTALL)


def _markdown_ticks_to_rest(text: str) -> str:
    """Upgrade markdown `` `code` `` to reST ``code``, without touching Sphinx roles."""
    protected: list[str] = []

    def stash(m: re.Match[str]) -> str:
        protected.append(m.group(0))
        return f'\x00ROLE{len(protected) - 1}\x00'

    text = re.sub(r':[\w]+:`[^`\n]+`', stash, text)
    text = re.sub(r'(?<!`)`([^`\n]+)`(?!`)', r'``\1``', text)
    for i, role in enumerate(protected):
        text = text.replace(f'\x00ROLE{i}\x00', role)
    return text


def _keep_literal_block_indent(lines: list[str]) -> list[str]:
    """Lstrip wrapped prose; keep extra indent only in a ``::`` literal block.

    Doxygen continues ``@param`` / ``@returns`` with a 4-space wrap. That extra
    indent is invalid RST inside a NumPy Parameters/Returns section unless it
    follows ``::``.
    """
    out: list[str] = []
    in_literal = False
    lit_indent: int | None = None
    for line in lines:
        raw = line.rstrip()
        if in_literal:
            if not raw.strip():
                out.append('')
                continue
            indent = len(raw) - len(raw.lstrip(' '))
            if lit_indent is None:
                lit_indent = indent
            if indent >= max(lit_indent, 1):
                out.append(raw)
                continue
            in_literal = False
            lit_indent = None
        out.append(raw.strip())
        if raw.strip().endswith('::'):
            in_literal = True
            lit_indent = None
    return out


def _join_wrapped_inline_literals(text: str) -> str:
    """Join lines that split an inline ``...`` literal (RST cannot span those)."""
    out: list[str] = []
    buf: list[str] | None = None
    for line in text.split('\n'):
        if buf is None:
            if line.count('``') % 2 == 1:
                buf = [line]
            else:
                out.append(line)
            continue
        buf.append(line.strip())
        if ''.join(buf).count('``') % 2 == 0:
            out.append(' '.join(buf))
            buf = None
    if buf:
        out.extend(buf)
    return '\n'.join(out)


def _inline_math_to_sphinx(text: str) -> str:
    """Rewrite Doxygen @f$ ... @f$ / \\f$ ... \\f$ to :math:`...`.

    Surrounding spaces and newlines (from wrapped /// comments) are stripped.
    RST inline markup is not recognized when a backtick is adjacent to
    whitespace.
    """

    def repl(m: re.Match[str]) -> str:
        body = re.sub(r'\s+', ' ', m.group(1)).strip()
        if not body:
            return m.group(0)
        return f':math:`{body}`'

    return _INLINE_MATH_RE.sub(repl, text)


def _display_math_to_sphinx(text: str) -> str:
    """Rewrite Doxygen ``\\f[ ... \\f]`` / ``@f[ ... @f]`` to a Sphinx ``.. math::`` block."""

    def repl(m: re.Match[str]) -> str:
        body = m.group(1)
        lines = body.splitlines()
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()
        if not lines:
            return '.. math::\n'
        indents = [len(l) - len(l.lstrip()) for l in lines if l.strip()]
        pad = min(indents) if indents else 0
        out_lines: list[str] = []
        for l in lines:
            s = l[pad:] if len(l) >= pad else l
            out_lines.append(('    ' + s) if s.strip() else '')
        return '.. math::\n\n' + '\n'.join(out_lines) + '\n'

    return _DISPLAY_MATH_RE.sub(repl, text)


def _cpp_ref_marker(sym: Symbol, *, has_overloads: bool) -> str:
    """Trailing RST marker consumed by ``docs/sphinx/lasyten_cpp_link.py``.

    Targets are bare qualified names (``lasyten::compose``) matching Sphinx's
    ``cpp:function`` inventory keys. Overload disambiguation is not required for
    the badge (inventory stores one entry per name).
    """
    del has_overloads  # inventory keys do not include signatures
    qname = '::'.join([*sym.namespace_parts, sym.name])
    if sym.kind in ('class', 'struct'):
        role = 'class'
    elif sym.kind == 'enum':
        role = 'enum'
    elif sym.kind == 'function':
        role = 'func'
    else:
        return ''
    lines = [f'.. lasyten-cpp-ref:: {qname}']
    if role != 'func':
        lines.append(f'   :role: {role}')
    return '\n' + '\n'.join(lines) + '\n'


def _sanitize_ident(part: str) -> str:
    """Make a Doxygen/C++ symbol fragment safe as a C identifier piece."""
    # Common operators first (order matters for multi-char).
    replacements = [
        ('operator<=>', 'operator_spaceship'),
        ('operator==', 'operator_eq'),
        ('operator!=', 'operator_ne'),
        ('operator<=', 'operator_le'),
        ('operator>=', 'operator_ge'),
        ('operator<<', 'operator_lshift'),
        ('operator>>', 'operator_rshift'),
        ('operator+=', 'operator_iadd'),
        ('operator-=', 'operator_isub'),
        ('operator*=', 'operator_imul'),
        ('operator/=', 'operator_idiv'),
        ('operator%=', 'operator_imod'),
        ('operator&=', 'operator_iand'),
        ('operator|=', 'operator_ior'),
        ('operator^=', 'operator_ixor'),
        ('operator&&', 'operator_land'),
        ('operator||', 'operator_lor'),
        ('operator++', 'operator_inc'),
        ('operator--', 'operator_dec'),
        ('operator->', 'operator_arrow'),
        ('operator()', 'operator_call'),
        ('operator[]', 'operator_index'),
        ('operator+', 'operator_plus'),
        ('operator-', 'operator_minus'),
        ('operator*', 'operator_star'),
        ('operator/', 'operator_slash'),
        ('operator%', 'operator_percent'),
        ('operator^', 'operator_xor'),
        ('operator&', 'operator_bitand'),
        ('operator|', 'operator_bitor'),
        ('operator~', 'operator_bitnot'),
        ('operator!', 'operator_not'),
        ('operator=', 'operator_assign'),
        ('operator<', 'operator_lt'),
        ('operator>', 'operator_gt'),
        ('operator,', 'operator_comma'),
    ]
    for src, dst in replacements:
        if part == src:
            part = dst
            break
    out: list[str] = []
    for ch in part:
        if ch.isalnum() or ch == '_':
            out.append(ch)
        else:
            out.append('_')
    s = ''.join(out)
    if not s or s[0].isdigit():
        s = 'x_' + s
    return s


def _macro_name(parts: list[str], overload_suffix: int | None) -> str:
    base = '_'.join(_sanitize_ident(p) for p in parts)
    if overload_suffix is None:
        return f'mkd_doc_{base}'
    return f'mkd_doc_{base}_{overload_suffix}'


def _escape_raw_string(doc: str) -> str:
    # Avoid terminating R"doc( ... )doc"
    if ')doc"' in doc:
        raise SystemExit('docstring contains )doc" which breaks R"doc(...)doc"')
    return doc


def format_entry(macro: str, doc: str) -> str:
    doc = _escape_raw_string(doc)
    if '\n' not in doc.rstrip('\n'):
        # Single-line: keep compact like mkdoc for short briefs
        one = doc.rstrip('\n')
        return f'static const char* {macro} =\n  R"doc({one})doc";\n'
    # Multi-line: content starts on same line as R"doc(
    body = doc if doc.endswith('\n') else doc + '\n'
    return f'static const char* {macro} =\n  R"doc({body})doc";\n'


def generate_entries(
    xml_dir: Path, header_rel: str, source_root: Path
) -> list[tuple[list[str], int | None, str]]:
    """Return ``(macro_parts, overload_suffix, docstring)`` for one header.

    ``macro_parts`` matches ``DOC(part1, part2, ...)`` (e.g. ``['lasyten', 'compose']``).
    ``overload_suffix`` is ``None`` for the first overload, else ``2``, ``3``, …
    """
    source_path = source_root / 'include' / 'lasyten' / header_rel
    if not source_path.is_file():
        raise FileNotFoundError(f'source header not found: {source_path}')
    source_lines = source_path.read_text(encoding='utf-8').splitlines()

    symbols = collect_symbols(xml_dir, header_rel)
    by_key: dict[tuple[str, ...], list[Symbol]] = defaultdict(list)
    for s in symbols:
        key = (*s.namespace_parts, s.name)
        by_key[key].append(s)

    ordered: list[tuple[tuple[str, ...], Symbol, int | None, bool]] = []
    for key, group in by_key.items():
        group = sorted(group, key=lambda s: s.line)
        has_overloads = len(group) > 1
        for idx, s in enumerate(group):
            suffix = None if idx == 0 else idx + 1  # 2, 3, ...
            ordered.append((key, s, suffix, has_overloads))
    ordered.sort(key=lambda t: t[1].line)

    entries: list[tuple[list[str], int | None, str]] = []
    for key, sym, suffix, has_overloads in ordered:
        doc = extract_slash_comment(source_lines, sym.line)
        if doc is None:
            continue
        doc = doxygen_comment_to_numpy(doc)
        marker = _cpp_ref_marker(sym, has_overloads=has_overloads)
        if marker:
            doc = doc.rstrip() + '\n' + marker
            if not doc.endswith('\n'):
                doc += '\n'
        entries.append((list(key), suffix, doc))
    return entries


def entries_to_macro_map(
    entries: list[tuple[list[str], int | None, str]],
) -> dict[str, str]:
    """Map ``mkd_doc_…`` macro names (as produced by ``DOC(...)``) to docstrings."""
    return {_macro_name(parts, suffix): doc for parts, suffix, doc in entries}


def lookup_doc(doc_map: dict[str, str], *doc_args: str | int) -> str | None:
    """Resolve ``DOC(lasyten, name, …)`` args against a macro-name docstring map."""
    parts: list[str] = []
    suffix: int | None = None
    for arg in doc_args:
        if isinstance(arg, int):
            suffix = arg
        else:
            parts.append(str(arg))
    return doc_map.get(_macro_name(parts, suffix))


def generate(xml_dir: Path, header_rel: str, source_root: Path) -> str:
    entries = generate_entries(xml_dir, header_rel, source_root)
    if not entries:
        raise SystemExit(f'no documented symbols found for {header_rel} in {xml_dir}')

    out: list[str] = [_BANNER.format(source_rel=header_rel), _DOC_PREAMBLE]
    for parts, suffix, doc in entries:
        out.append(format_entry(_macro_name(parts, suffix), doc))
        out.append('\n')
    out.append('#if defined(__GNUG__)\n#pragma GCC diagnostic pop\n#endif\n')
    return ''.join(out)


# Headers that participate in DOC() generation (kept in sync with CMakeLists.txt).
LASYTEN_MKDOC_HEADERS: tuple[str, ...] = (
    'tensors/ops_algebra.h',
    'tensors/constructors.h',
    'tensors/ops_elementwise.h',
    'tensors/ops_legs.h',
    'tensors/decompositions.h',
    'tensors/helpers.h',
    'tensors/labels.h',
    'tensors/vector_like.h',
    'tensors/direct_sum.h',
    'tensors/tensor.h',
    'tensors/symmetric_tensor.h',
    'tensors/diagonal_tensor.h',
    'tensors/mask.h',
    'tensors/charged_tensor.h',
    'tensors/hidden_leg_tensor.h',
    'tensors/sparse.h',
    'tensors/planar.h',
    'tensors/krylov_based.h',
    'symmetries/factors/fermion_number.h',
    'symmetries/factors/fermion_parity.h',
    'symmetries/factors/fibonacci_anyon_category.h',
    'symmetries/factors/ising_anyon_category.h',
    'symmetries/factors/no_symmetry.h',
    'symmetries/factors/quantum_double_zn_anyon_category.h',
    'symmetries/factors/su2.h',
    'symmetries/factors/su2_k_anyon_category.h',
    'symmetries/factors/su3_3_anyon_category.h',
    'symmetries/factors/sun.h',
    'symmetries/factors/toric_code_category.h',
    'symmetries/factors/u1.h',
    'symmetries/factors/zn.h',
    'symmetries/factors/zn_anyon_category.h',
    'symmetries/factors/zn_anyon_category2.h',
    'symmetries/abelian_group.h',
    'symmetries/base_symmetry.h',
    'symmetries/exceptions.h',
    'symmetries/group.h',
    'symmetries/sector.h',
    'symmetries/spaces.h',
    'symmetries/styles.h',
    'symmetries/symmetry.h',
    'symmetries/symmetry_factor.h',
    'symmetries/trees.h',
    'backends/abelian.h',
    'backends/backend_factory.h',
    'backends/block_inds.h',
    'backends/fusion_tree_backend.h',
    'backends/fusion_tree_mapping.h',
    'backends/no_symmetry.h',
    'backends/tensor_backend.h',
    'block_backend/array_api.h',
    'block_backend/block_backend.h',
    'block_backend/dtypes.h',
    'block_backend/eigen.h',
    'block_backend/numpy.h',
    'block_backend/torch.h',
    'tools/cost_polynomials.h',
    'tools/mappings.h',
    'models/couplings.h',
    'models/degrees_of_freedom.h',
    'models/sites.h',
    'tools.h',
)


def run_scoped_doxygen(
    header_rel: str,
    source_root: Path,
    xml_out: Path,
    *,
    doxygen: str = 'doxygen',
) -> Path:
    """Run a scoped Doxygen XML build for one header; return the ``xml/`` directory."""
    import subprocess
    import tempfile

    src_hdr = source_root / 'include' / 'lasyten' / header_rel
    if not src_hdr.is_file():
        raise FileNotFoundError(src_hdr)
    xml_out.mkdir(parents=True, exist_ok=True)
    doxyfile = tempfile.NamedTemporaryFile('w', suffix='.doxyfile', delete=False, encoding='utf-8')
    try:
        # Keep PREDEFINED in sync with CMakeLists.txt docstring Doxyfile and
        # docs/Doxyfile so #if LASYTEN_HAS_PYTHON APIs (e.g. Scalar::to_numpy)
        # appear in the DOC() map used by RTD stub generation (--from-doxygen).
        doxyfile.write(
            f"""PROJECT_NAME           = lasyten
OUTPUT_DIRECTORY       = {xml_out}
INPUT                  = {src_hdr}
RECURSIVE              = NO
GENERATE_HTML          = NO
GENERATE_LATEX         = NO
GENERATE_XML           = YES
EXTRACT_ALL            = YES
QUIET                  = YES
WARNINGS               = NO
MARKDOWN_SUPPORT       = NO
AUTOLINK_SUPPORT       = NO
ENABLE_PREPROCESSING   = YES
MACRO_EXPANSION        = NO
PREDEFINED             = LASYTEN_HAS_PYTHON=1 LASYTEN_WITH_NUMPY_BACKEND=1 LASYTEN_WITH_TORCH_BACKEND=1
"""
        )
        doxyfile.close()
        subprocess.run([doxygen, doxyfile.name], check=True, capture_output=True)
    finally:
        Path(doxyfile.name).unlink(missing_ok=True)
    xml_dir = xml_out / 'xml'
    if not xml_dir.is_dir():
        raise RuntimeError(f'Doxygen did not write XML under {xml_out}')
    return xml_dir


def build_doc_map(
    source_root: Path,
    *,
    headers: tuple[str, ...] | list[str] = LASYTEN_MKDOC_HEADERS,
    work_dir: Path | None = None,
    doxygen: str = 'doxygen',
    reuse_cmake_xml: Path | None = None,
) -> dict[str, str]:
    """Build a full ``mkd_doc_…`` → docstring map without configuring CMake.

    If ``reuse_cmake_xml`` points at a CMake ``docstrings_doxygen`` tree (with
    per-header subdirs ``…/<safe_rel>/xml``), reuse that XML. Otherwise run
    scoped Doxygen into ``work_dir``.
    """
    import shutil

    source_root = source_root.resolve()
    if work_dir is None:
        work_dir = source_root / 'build' / 'stub_docstrings_doxygen'
    work_dir = work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    merged: dict[str, str] = {}
    for header_rel in headers:
        safe = header_rel.replace('/', '_').replace('.', '_')
        if reuse_cmake_xml is not None:
            xml_dir = Path(reuse_cmake_xml) / safe / 'xml'
            if not xml_dir.is_dir():
                raise FileNotFoundError(f'missing CMake docstring XML: {xml_dir}')
        else:
            xml_root = work_dir / safe
            if xml_root.exists():
                shutil.rmtree(xml_root)
            xml_dir = run_scoped_doxygen(header_rel, source_root, xml_root, doxygen=doxygen)
        entries = generate_entries(xml_dir, header_rel, source_root)
        merged.update(entries_to_macro_map(entries))
    return merged


def parse_docstring_headers(docstrings_dir: Path) -> dict[str, str]:
    """Parse already-generated ``pybind/docstrings/**/*.h`` into a macro → doc map.

    Faster than re-running Doxygen when CMake has already produced the headers.
    """
    doc_re = re.compile(
        r'static const char\*\s+(mkd_doc_[A-Za-z0-9_]+)\s*=\s*'
        r'R"doc\((.*?)\)doc"\s*;',
        re.DOTALL,
    )
    out: dict[str, str] = {}
    for path in sorted(docstrings_dir.rglob('*.h')):
        text = path.read_text(encoding='utf-8')
        for m in doc_re.finditer(text):
            body = m.group(2)
            if '\n' in body and not body.endswith('\n'):
                body = body + '\n'
            out[m.group(1)] = body
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--xml-dir', type=Path, required=True, help='Doxygen XML output directory')
    ap.add_argument(
        '--header-rel',
        required=True,
        help='Path relative to include/lasyten/ (e.g. tensors/ops_algebra.h)',
    )
    ap.add_argument(
        '--source-root',
        type=Path,
        default=Path('.'),
        help='Repository root containing include/lasyten/',
    )
    ap.add_argument('-o', '--output', type=Path, required=True, help='Output docstring header')
    args = ap.parse_args()
    if not args.xml_dir.is_dir():
        raise SystemExit(f'XML dir not found: {args.xml_dir}')
    text = generate(args.xml_dir, args.header_rel, args.source_root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding='utf-8')
    return 0


if __name__ == '__main__':
    sys.exit(main())
