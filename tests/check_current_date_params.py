#!/usr/bin/env python3
"""Check for strategy selector calls missing current_date."""
from __future__ import annotations

import ast
import os
import sys
from dataclasses import dataclass


SRC_DIR = "src"


@dataclass
class SelectorSignature:
    file_path: str
    line_no: int
    positional_params: list[str]
    kwonly_params: list[str]

    @property
    def has_current_date(self) -> bool:
        return "current_date" in self.positional_params or "current_date" in self.kwonly_params

    @property
    def current_date_position(self) -> int | None:
        try:
            return self.positional_params.index("current_date")
        except ValueError:
            return None


@dataclass
class FileAnalysis:
    alias_map: dict[str, str]
    calls: list[tuple[ast.Call, str]]


def _iter_python_files(root: str) -> list[str]:
    python_files = []
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if filename.endswith(".py"):
                python_files.append(os.path.join(dirpath, filename))
    return sorted(python_files)


def _load_ast(file_path: str) -> ast.AST:
    with open(file_path, "r", encoding="utf-8") as f:
        return ast.parse(f.read(), filename=file_path)


def _build_selector_signatures(root: str) -> dict[str, SelectorSignature]:
    signatures: dict[str, SelectorSignature] = {}

    for file_path in _iter_python_files(root):
        tree = _load_ast(file_path)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if not node.name.startswith("select_") or not node.name.endswith("_stocks"):
                continue

            positional_params = [arg.arg for arg in node.args.posonlyargs + node.args.args]
            kwonly_params = [arg.arg for arg in node.args.kwonlyargs]
            signatures[node.name] = SelectorSignature(
                file_path=file_path,
                line_no=node.lineno,
                positional_params=positional_params,
                kwonly_params=kwonly_params,
            )

    return signatures


def _get_called_selector_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        func_name = node.func.id
    elif isinstance(node.func, ast.Attribute):
        func_name = node.func.attr
    else:
        return None

    if func_name.startswith("select_") and func_name.endswith("_stocks"):
        return func_name
    return None


def _collect_file_analysis(tree: ast.AST) -> FileAnalysis:
    alias_map: dict[str, str] = {}
    calls: list[tuple[ast.Call, str]] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                imported_name = alias.name
                if imported_name.startswith("select_") and imported_name.endswith("_stocks"):
                    alias_map[alias.asname or imported_name] = imported_name

        if isinstance(node, ast.Call):
            func_name = _get_called_selector_name(node)
            if func_name is not None:
                calls.append((node, func_name))

    return FileAnalysis(alias_map=alias_map, calls=calls)


def _call_passes_current_date(node: ast.Call, signature: SelectorSignature | None) -> bool:
    if any(keyword.arg == "current_date" for keyword in node.keywords if keyword.arg is not None):
        return True

    if signature is None or not signature.has_current_date:
        return False

    current_date_position = signature.current_date_position
    if current_date_position is None:
        return False

    positional_arg_count = len(node.args)
    return positional_arg_count > current_date_position


def _format_call(node: ast.Call) -> str:
    try:
        return ast.unparse(node)
    except Exception:
        return "<unable to render call>"


def main() -> int:
    selector_signatures = _build_selector_signatures(SRC_DIR)
    violations: list[tuple[str, int, str, str]] = []

    for file_path in _iter_python_files(SRC_DIR):
        tree = _load_ast(file_path)
        file_analysis = _collect_file_analysis(tree)
        for node, raw_func_name in file_analysis.calls:
            func_name = file_analysis.alias_map.get(raw_func_name, raw_func_name)
            signature = selector_signatures.get(func_name)
            if signature is None:
                continue
            if _call_passes_current_date(node, signature):
                continue

            violations.append((file_path, node.lineno, func_name, _format_call(node)))

    if violations:
        print("❌ Missing current_date parameter detected in:")
        for file_path, line_no, func_name, rendered_call in violations:
            print(f"   {file_path}:{line_no} -> {func_name}")
            print(f"      {rendered_call}")
        return 1

    print("✅ All strategy calls include current_date parameter")
    return 0


if __name__ == "__main__":
    sys.exit(main())
