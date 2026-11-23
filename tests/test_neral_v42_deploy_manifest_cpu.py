"""CPU-only deployment closure regression tests; no GPU, network or owner service."""
from __future__ import annotations

import ast
import importlib
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OWNER = ROOT / "scripts" / "neral_v41_owner.ps1"
ENTRY_POINTS = ("neral_evidence_agent", "neral_page_reader")


def deployment_manifest() -> list[str]:
    source = OWNER.read_text(encoding="utf-8")
    matches = re.findall(r"(?ms)^\$files\s*=\s*@\((.*?)\)", source)
    if len(matches) != 1:
        raise AssertionError("Expected exactly one owner deploy/rollback $files manifest")
    return re.findall(r"'([^']+)'", matches[0])


def local_python_dependencies(module: str) -> set[str]:
    path = ROOT / (module + ".py")
    if not path.is_file():
        raise AssertionError(f"Missing local import target: {module}")
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names = (node.module,)
        else:
            continue
        for name in names:
            root = name.split(".", 1)[0]
            # Only local root neral_*.py modules need individual file staging.
            # Local packages (e.g. neral_cpu_foundation/) belong to the
            # existing frozen release baseline, not this module closure.
            if root.startswith("neral_") and (ROOT / (root + ".py")).is_file():
                found.add(root)
    return found


def evidence_import_closure() -> set[str]:
    pending = list(ENTRY_POINTS)
    visited = set()
    while pending:
        module = pending.pop()
        if module in visited:
            continue
        visited.add(module)
        pending.extend(local_python_dependencies(module) - visited)
    return {name + ".py" for name in visited}


class OwnerEvidenceManifestCPU(unittest.TestCase):
    def test_every_transitive_local_import_is_deployed_and_rolled_back(self):
        files = deployment_manifest()
        self.assertEqual(len(files), len(set(files)), "Duplicate owner deploy paths")
        required = evidence_import_closure()
        missing = required - set(files)
        self.assertFalse(missing, f"Unstaged NERAL evidence dependencies: {sorted(missing)}")
        self.assertIn("neral_web_search.py", required)
        self.assertIn("neral_quantity_plans.py", required)
        self.assertIn("neral_v32_verifier.py", required)
        self.assertIn("neral_z3_json_compiler.py", required)
        self.assertIn("neral_knowledge_verifier.py", required)

    def test_regression_missing_web_search_cannot_pass(self):
        required = evidence_import_closure()
        manifest_without_search = set(deployment_manifest()) - {"neral_web_search.py"}
        self.assertEqual({"neral_web_search.py"} & required - manifest_without_search,
                         {"neral_web_search.py"})

    def test_real_evidence_imports_are_cpu_safe(self):
        for name in ENTRY_POINTS:
            with self.subTest(module=name):
                self.assertIsNotNone(importlib.import_module(name))

    def test_owner_checks_isolated_import_from_staged_root(self):
        source = OWNER.read_text(encoding="utf-8")
        self.assertIn('& $python -I -c "import sys;', source)
        self.assertIn("import neral_page_reader; import neral_evidence_agent", source)
        self.assertIn("$py=@($files | Where-Object { $_ -like '*.py' })", source)


if __name__ == "__main__":
    unittest.main()
