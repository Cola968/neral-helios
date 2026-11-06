"""Reproduces the math-v2 Windows evaluator failure: control model_catalog shadows trainer."""
import importlib.machinery
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
from scripts.neral_math_v2_eval import ensure_pinned_model_runtime


class PinnedRuntimeImports(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root=Path(self.tmp.name)
        self.control=root/'control'
        self.trainer=root/'trainer'
        self.control.mkdir()
        self.trainer.mkdir()
        for name in ('model_catalog.py','qwen_local_runtime.py','neral_adapter_candidate.py'):
            (self.trainer/name).write_text('PINNED = True\n',encoding='utf8')
        (self.control/'model_catalog.py').write_text('PINNED = False\n',encoding='utf8')

    def test_pinned_model_catalog_wins_even_from_control_cwd(self):
        with patch.object(sys,'path',[str(self.control)]+sys.path[:]):
            ensure_pinned_model_runtime(self.trainer)
            spec=importlib.machinery.PathFinder.find_spec('model_catalog',sys.path)
            self.assertEqual(Path(spec.origin).resolve(),(self.trainer/'model_catalog.py').resolve())

    def test_stale_cached_unpinned_model_catalog_refused(self):
        fake=types.ModuleType('model_catalog')
        fake.__file__=str(self.control/'model_catalog.py')
        with patch.dict(sys.modules,{'model_catalog':fake}):
            with self.assertRaises(ValueError):
                ensure_pinned_model_runtime(self.trainer)

    def test_missing_model_identity_source_refused(self):
        (self.trainer/'model_catalog.py').unlink()
        with self.assertRaises(ValueError):
            ensure_pinned_model_runtime(self.trainer)

if __name__=='__main__':unittest.main()