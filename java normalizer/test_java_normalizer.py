# test_java_normalizer.py
# pytest unit tests for java_normalizer.normalize_java_source
# Run: pytest -q

import importlib
import sys
import textwrap
import types
import pytest

jn = importlib.import_module("java_normalizer")

# ---- optional dependency guard ----
_tree_sitter_ok = True
try:
    # parser is lazy-initialized; ensure import availability
    import tree_sitter  # noqa: F401
    import tree_sitter_languages  # noqa: F401
except Exception:
    _tree_sitter_ok = False


def norm(src, **kw):
    return jn.normalize_java_source(src, **kw)


def test_whitespace_normalization_trailing_and_tabs():
    src = "class A{\tint x;\t \n\n\n}\r\n"
    out = norm(src, strip_comments=False, mask_strings=False, mask_numbers=False, scope_type_rename=False)
    assert out == "class A{    int x;\n\n}\n"


def test_comment_removal_preserves_string_like_sequences():
    src = r'''
class A {
  // leading comment
  String s = "not // a comment /* keep */";
  char c = '/';
  /* block
     comment */
  int x = 1; // trailing
}
'''
    out = norm(src, strip_comments=True, mask_strings=True, mask_numbers=True, scope_type_rename=False)
    # comments removed
    assert "//" not in out and "/*" not in out
    # literals masked
    assert '<STR>' in out
    assert '<NUM>' in out
    # char literal is also matched by _STR_RX
    assert out.count('<STR>') >= 1


def test_literal_masking_numbers_and_strings():
    src = r'class A { void f(){ int x=42; long y=0xFFL; float z=1.0e-3f; String s="hi"; }}'
    out = norm(src, strip_comments=False, mask_strings=True, mask_numbers=True, scope_type_rename=False)
    assert out.count("<NUM>") >= 3
    assert '<STR>' in out


def test_whitespace_collapses_large_blank_runs_and_trailing_lines():
    src = "class B {\n\n\n  int x = 1;\n\n\n\n  void f() {}\n}\n\n\n"
    out = norm(src, strip_comments=False, mask_strings=False, mask_numbers=False, scope_type_rename=False)
    assert out == "class B {\n\n  int x = 1;\n\n  void f() {}\n}\n"


def test_whitespace_only_input_returns_empty_string():
    src = " \t \n\r\n \n"
    out = norm(src, strip_comments=False, mask_strings=False, mask_numbers=False, scope_type_rename=False)
    assert out == ""


def test_comment_preservation_when_disabled():
    src = "class C {\n  // keep this\n  int x = 1; /* inline */\n}\n"
    out = norm(src, strip_comments=False, mask_strings=False, mask_numbers=False, scope_type_rename=False)
    assert "// keep this" in out
    assert "/* inline */" in out


def test_literal_masking_strings_only():
    src = 'class D { String s="hi"; int n=7; }'
    out = norm(src, strip_comments=False, mask_strings=True, mask_numbers=False, scope_type_rename=False)
    assert '<STR>' in out
    assert '7' in out


def test_literal_masking_numbers_only():
    src = 'class E { String s="hi"; int n=7; }'
    out = norm(src, strip_comments=False, mask_strings=False, mask_numbers=True, scope_type_rename=False)
    assert '<NUM>' in out
    assert '"hi"' in out


def test_literal_masking_complex_numeric_forms():
    src = "class F { double a = 0x1.ap2d; long b = 0b1010_0101L; int c = 123_456; }"
    out = norm(src, strip_comments=False, mask_strings=False, mask_numbers=True, scope_type_rename=False)
    assert out.count("<NUM>") == 3


def test_literal_masking_disabled_preserves_literals():
    src = 'class G { String s="hi"; double z=1.5; }'
    out = norm(src, strip_comments=False, mask_strings=False, mask_numbers=False, scope_type_rename=False)
    assert '"hi"' in out and '1.5' in out


def test_normalize_idempotent_when_already_clean():
    src = "class H {\n  String s = \"hi\";\n}\n"
    first = norm(src, scope_type_rename=False)
    second = norm(first, scope_type_rename=False)
    assert first == second


@pytest.mark.skipif(not _tree_sitter_ok, reason="tree-sitter not available")
def test_scope_and_type_renaming_parameters_and_locals():
    src = """
class Demo {
  public int sum(int a, int b) {
    int result = a + b;
    return result;
  }
}
"""
    out = norm(src)  # defaults: strip comments, mask literals, rename
    # parameters become num1, num2; local becomes num3; literals masked
    assert "int num1, int num2" in out
    assert "int num3 = num1 + num2;" in out
    assert "return num3;" in out


@pytest.mark.skipif(not _tree_sitter_ok, reason="tree-sitter not available")
def test_scope_reset_between_methods_and_type_buckets():
    src = """
class Demo {
  public int sum(int a, int b) { int r = a + b; return r; }
  public String join(String x, String y) { String z = x + y; return z; }
}
"""
    out = norm(src)
    # int bucket uses num*, String bucket uses str*
    assert "int num1, int num2" in out
    assert "int num3 = num1 + num2" in out
    assert "String str1, String str2" in out
    assert "String str3 = str1 + str2" in out
    # numbering restarts per method
    assert "public int sum" in out and "public String join" in out
    assert out.count("int num1, int num2") == 1
    assert out.count("String str1, String str2") == 1


@pytest.mark.skipif(not _tree_sitter_ok, reason="tree-sitter not available")
def test_nested_blocks_and_for_loop_vars():
    src = """
class C {
  void g(int n){
    for (int i=0; i<n; i++){
      int t = i * 2;
      System.out.println(t);
    }
  }
}
"""
    out = norm(src)
    # n -> num1, i -> num2, t -> num3; numbers masked
    assert "void g(int num1)" in out
    assert "for (int num2=<NUM>; num2<num1; num2++)" in out
    assert "int num3 = num2 * <NUM>;" in out
    assert "System.out.println(num3);" in out


@pytest.mark.skipif(not _tree_sitter_ok, reason="tree-sitter not available")
def test_array_and_reference_type_buckets():
    src = """
class Arr {
  void h(int[] a, java.util.List<String> lst) {
    int[] b = a;
    java.util.List<String> c = lst;
  }
}
"""
    out = norm(src)
    # arrays get num_arr*, references get obj*
    assert "int[] num_arr1" in out
    assert "java.util.List<String> obj1" in out
    # locals increment counters within bucket
    assert "int[] num_arr2 = num_arr1;" in out
    assert "java.util.List<String> obj2 = obj1;" in out


@pytest.mark.skipif(not _tree_sitter_ok, reason="tree-sitter not available")
def test_scope_boolean_and_char_buckets():
    src = textwrap.dedent("""
class Flags {
  boolean flip(boolean active, char symbol) {
    boolean toggled = !active;
    char copy = symbol;
    return toggled && (copy == symbol);
  }
}
""")
    out = norm(src, mask_numbers=False)
    assert "boolean flip(boolean bool1, char chr1)" in out
    assert "boolean bool2 = !bool1" in out
    assert "char chr2 = chr1;" in out


@pytest.mark.skipif(not _tree_sitter_ok, reason="tree-sitter not available")
def test_scope_enhanced_for_loop_and_locals():
    src = textwrap.dedent("""
class ForEach {
  void walk(java.util.List<String> inputs) {
    for (String item : inputs) {
      String upper = item.toUpperCase();
    }
  }
}
""")
    out = norm(src, mask_numbers=False)
    assert "void walk(java.util.List<String> obj1)" in out
    assert "for (String str1 : obj1)" in out
    assert "String str2 = str1.toUpperCase();" in out


@pytest.mark.skipif(not _tree_sitter_ok, reason="tree-sitter not available")
def test_scope_shadowed_locals_share_mapping_for_duplicate_names():
    src = textwrap.dedent("""
class Shadow {
  void demo(int value) {
    int tmp = value;
    {
      int tmp = value + 1;
      System.out.println(tmp);
    }
    System.out.println(tmp);
  }
}
""")
    out = norm(src)
    assert "int num3 = num1;" in out
    assert "int num3 = num1 + <NUM>;" in out
    assert out.count("System.out.println(num3);") == 2


@pytest.mark.skipif(not _tree_sitter_ok, reason="tree-sitter not available")
def test_scope_varargs_and_foreach_interaction():
    src = textwrap.dedent("""
class Logger {
  void log(String prefix, String... messages) {
    for (String message : messages) {
      System.out.println(prefix + message);
    }
  }
}
""")
    out = norm(src, mask_numbers=False)
    assert "void log(String str1, String... obj1)" in out
    assert "for (String str2 : obj1)" in out
    assert "System.out.println(str1 + str2);" in out


@pytest.mark.skipif(not _tree_sitter_ok, reason="tree-sitter not available")
def test_scope_lambda_parameters_remain_untouched():
    src = textwrap.dedent("""
class Lambda {
  void run() {
    java.util.function.Function<String, String> fn = s -> {
      String local = s.trim();
      return local;
    };
  }
}
""")
    out = norm(src)
    assert "Function<String, String> obj1 = s -> {" in out
    assert "String str1 = s.trim();" in out
    assert "return str1;" in out


@pytest.mark.skipif(not _tree_sitter_ok, reason="tree-sitter not available")
def test_scope_does_not_overwrite_field_members():
    src = textwrap.dedent("""
class Color {
  final String hex;
  Color(String hex) { this.hex = hex; }
  String describe() { return hex; }
}
""")
    out = norm(src, mask_numbers=False)
    assert "Color(String str1) { this.hex = str1; }" in out
    assert "return hex;" in out


def test_disable_renaming_and_masks_independently():
    src = 'class X{ String s="hi"; int x=7; }'
    out = norm(src, scope_type_rename=False, mask_strings=True, mask_numbers=False, strip_comments=True)
    assert '"hi"' not in out and '<STR>' in out
    assert '7' in out  # numbers kept
    assert 'String s' in out and 'int x' in out  # identifiers unchanged


def test_parser_failure_fallback_does_not_crash(monkeypatch):
    # Force parser failure path and ensure function still returns a string.
    if _tree_sitter_ok:
        monkeypatch.setattr(jn, "scope_type_variable_normalize_java", lambda x: (_ for _ in ()).throw(RuntimeError("fail")))
    src = "class A { void f(int a){ int b=a; } }"
    out = norm(src, scope_type_rename=True)
    assert isinstance(out, str) and len(out) > 0
