# java_normalizer.py
"""
Java source normalization:
- Whitespace normalization
- Comment removal
- Literal normalization (strings/char -> <STR>, numbers -> <NUM>)
- Scope + type–aware renaming for method/ctor parameters and local variables

Install:
  pip install tree_sitter tree_sitter_languages

API:
  from java_normalizer import normalize_java_source
  text = normalize_java_source(java_code)

CLI:
  python -m java_normalizer file1.java file2.java
"""

from __future__ import annotations
import re
import warnings
from dataclasses import dataclass
from typing import Dict

# ---------------- Whitespace ----------------
def normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\t", "    ")
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text)
    return (text.strip() + ("\n" if text.strip() else ""))

# ---------------- Comments ----------------
_LINE = re.compile(r"//.*?$", re.M)
_BLOCK = re.compile(r"/\*.*?\*/", re.S)

def remove_comments_java(text: str) -> str:
    # Replace block comments with spaces of equal length to preserve offsets if needed.
    def _repl_blocks(m): return " " * (m.end() - m.start())
    text = _BLOCK.sub(_repl_blocks, text)
    text = _LINE.sub("", text)
    return text

# ---------------- Literals ----------------
_STR_RX = re.compile(r"(\"(?:\\.|[^\"\\])*\")|('(?:\\.|[^'\\])*')", re.S)

# Java numeric literals incl. hex floats and suffixes; underscores allowed.
_NUM_RX = re.compile(
    r"""
    \b(
        0[xX](?:[0-9A-Fa-f_]+(?:\.[0-9A-Fa-f_]+)?)(?:[pP][+\-]?\d[\d_]*)?[fFdD]?[lL]? |  # hex int/float
        0[bB][01_]+[lL]? |                                                               # binary int
        0[oO][0-7_]+[lL]? |                                                              # octal int
        \d[\d_]*(?:\.\d[\d_]*)?(?:[eE][+\-]?\d[\d_]*)?[fFdD]?[lL]?                       # dec int/float
    )\b
    """,
    re.VERBOSE,
)

def normalize_literals_java(text: str, *, mask_strings: bool = True, mask_numbers: bool = True) -> str:
    if mask_strings:
        text = _STR_RX.sub("<STR>", text)
    if mask_numbers:
        text = _NUM_RX.sub("<NUM>", text)
    return text

# ---------------- Scope + Type renaming (tree-sitter) ----------------
@dataclass
class _Span:
    start: int
    end: int

_TYPE_PREFIX = {
    "byte": "num", "short": "num", "int": "num", "long": "num",
    "float": "num", "double": "num", "boolean": "bool", "char": "chr",
    "String": "str", "CharSequence": "str",
}

def _bucket_for_type(tname: str) -> str:
    base = tname.rstrip("[]").split(".")[-1]
    is_arr = tname.endswith("[]")
    prefix = _TYPE_PREFIX.get(base, "obj")
    return f"{prefix}{'_arr' if is_arr else ''}"

_PARSER = None
_LANG = None
_LANG_LIB = None

def _ensure_parser():
    global _PARSER, _LANG, _LANG_LIB
    if _PARSER is not None:
        return
    from tree_sitter import Language, Parser
    from tree_sitter_languages import get_language

    lang = None
    try:
        lang = get_language("java")
    except Exception:
        # tree_sitter>=0.25 removed the (path, name) initializer; fall back to manual loading.
        lang = None
    if lang is None:
        import ctypes
        try:
            from importlib import resources
        except ImportError:  # pragma: no cover
            import importlib_resources as resources  # type: ignore
        import tree_sitter_languages

        lib_path = resources.files(tree_sitter_languages) / "languages.so"
        _LANG_LIB = ctypes.CDLL(str(lib_path))
        getter = getattr(_LANG_LIB, "tree_sitter_java")
        getter.restype = ctypes.c_void_p
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="int argument support is deprecated", category=DeprecationWarning)
            lang = Language(getter())

    parser = Parser()
    if hasattr(parser, "set_language"):
        parser.set_language(lang)
    else:
        parser.language = lang
    _LANG = lang
    _PARSER = parser

def _node_text(src_bytes: bytes, node) -> str:
    return src_bytes[node.start_byte:node.end_byte].decode("utf-8", "ignore")

def _type_name(src: bytes, type_node) -> str:
    if type_node is None:
        return "obj"
    # Flatten array types to "base[]"
    if type_node.type == "array_type":
        base = _type_name(src, type_node.child_by_field_name("element"))
        return base + "[]"
    return _node_text(src, type_node)

def _method_like(node) -> bool:
    return node.type in ("method_declaration", "constructor_declaration")

def _first_identifier(node):
    if node is None:
        return None
    if node.type == "identifier":
        return node
    for child in node.named_children:
        found = _first_identifier(child)
        if found is not None:
            return found
    return None


def _register_decl(src, counters, mapping, type_node, name_node):
    ident = _first_identifier(name_node)
    if ident is None:
        return
    tname = _type_name(src, type_node)
    bucket = _bucket_for_type(tname)
    counters[bucket] = counters.get(bucket, 0) + 1
    mapping[_node_text(src, ident)] = f"{bucket}{counters[bucket]}"


def _gather_parameters(src, method_node, counters, mapping):
    params = method_node.child_by_field_name("parameters")
    if params is None:
        return
    for child in params.named_children:
        if child.type in ("formal_parameter", "spread_parameter", "receiver_parameter"):
            _register_decl(
                src,
                counters,
                mapping,
                child.child_by_field_name("type"),
                child.child_by_field_name("name") or child,
            )


def _gather_locals(src, node, counters, mapping):
    stack = [node]
    while stack:
        cur = stack.pop()
        if cur.type == "local_variable_declaration":
            ltype = cur.child_by_field_name("type")
            for child in cur.named_children:
                if child.type == "variable_declarator":
                    _register_decl(src, counters, mapping, ltype, child.child_by_field_name("name") or child)
        elif cur.type == "enhanced_for_statement":
            _register_decl(
                src,
                counters,
                mapping,
                cur.child_by_field_name("type"),
                cur.child_by_field_name("name"),
            )
        children = list(cur.named_children)
        if children:
            stack.extend(reversed(children))


def _collect_scope_maps(src: bytes, root):
    scopes = []
    stack = [root]
    while stack:
        n = stack.pop()
        if _method_like(n):
            body = n.child_by_field_name("body")
            span = _Span(n.start_byte, (body.end_byte if body else n.end_byte))
            counters: Dict[str, int] = {}
            mapping: Dict[str, str] = {}
            _gather_parameters(src, n, counters, mapping)
            if body is not None:
                _gather_locals(src, body, counters, mapping)
            if mapping:
                scopes.append((span, mapping))
        stack.extend(n.children)
    return scopes

def _collect_identifier_occurrences(src: bytes, root, scopes):
    # Replace only identifiers that are inside a method's span and present in that scope map.
    reps = []

    def _is_field_member(node) -> bool:
        parent = node.parent
        if parent is None or parent.type != "field_access":
            return False
        for idx, child in enumerate(parent.children):
            if child.start_byte == node.start_byte and child.end_byte == node.end_byte:
                role = parent.field_name_for_child(idx)
                return role == "field"
        return False

    def _walk(n):
        if n.type == "identifier":
            if _is_field_member(n):
                # Skip identifiers that are clearly fields (e.g., this.foo or obj.foo).
                return
            name = _node_text(src, n)
            s, e = n.start_byte, n.end_byte
            for span, mapping in scopes:
                if span.start <= s <= e <= span.end and name in mapping:
                    reps.append((s, e, mapping[name]))
                    break
        for c in n.children:
            _walk(c)
    _walk(root)
    return reps

def scope_type_variable_normalize_java(code: str) -> str:
    _ensure_parser()
    src = code.encode("utf-8", "ignore")
    tree = _PARSER.parse(src)
    root = tree.root_node

    scopes = _collect_scope_maps(src, root)
    if not scopes:
        return code

    reps = _collect_identifier_occurrences(src, root, scopes)
    if not reps:
        return code

    out = bytearray(src)
    for s, e, val in sorted(reps, key=lambda x: x[0], reverse=True):
        out[s:e] = val.encode("utf-8")
    return out.decode("utf-8", "ignore")

# ---------------- Public API ----------------
def normalize_java_source(
    code: str,
    *,
    strip_comments: bool = True,
    mask_strings: bool = True,
    mask_numbers: bool = True,
    scope_type_rename: bool = True,
) -> str:
    text = normalize_whitespace(code)
    if strip_comments:
        text = remove_comments_java(text)
    # Important: rename before literal masking to keep valid Java for the parser.
    if scope_type_rename:
        try:
            text = scope_type_variable_normalize_java(text)
        except Exception:
            pass
    if mask_strings or mask_numbers:
        text = normalize_literals_java(text, mask_strings=mask_strings, mask_numbers=mask_numbers)
    text = normalize_whitespace(text)
    return text

# ---------------- CLI ----------------
if __name__ == "__main__":
    import argparse, sys
    ap = argparse.ArgumentParser(description="Normalize Java source code.")
    ap.add_argument("paths", nargs="*", help="Files to process; stdin if empty.")
    ap.add_argument("--keep-comments", action="store_true")
    ap.add_argument("--keep-strings", action="store_true")
    ap.add_argument("--keep-numbers", action="store_true")
    ap.add_argument("--no-rename", action="store_true", help="Disable scope+type renaming.")
    args = ap.parse_args()

    def run(text: str) -> str:
        return normalize_java_source(
            text,
            strip_comments=not args.keep_comments,
            mask_strings=not args.keep_strings,
            mask_numbers=not args.keep_numbers,
            scope_type_rename=not args.no_rename,
        )

    if args.paths:
        outs = []
        for p in args.paths:
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                outs.append(run(f.read()))
        sys.stdout.write("\n".join(outs))
    else:
        sys.stdout.write(run(sys.stdin.read()))
