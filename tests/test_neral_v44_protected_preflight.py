"""Synthetic CPU checks for the original-owner snapshot comparison."""
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from scripts import neral_v44_protected_preflight as preflight

class RestoredOwnerTest(unittest.TestCase):
    def test_exact_bytes_only(self):
        with tempfile.TemporaryDirectory() as root:
            a,b=Path(root)/"owner",Path(root)/"backup"
            a.mkdir();b.mkdir()
            data=b"original\r\n"
            (a/"example.py").write_bytes(data)
            (b/"example.py").write_bytes(data)
            with patch.object(preflight,"FILES",("example.py",)):
                self.assertEqual(preflight.check_restored_owner(a,b),{"example.py":hashlib.sha256(data).hexdigest()})
                (a/"example.py").write_bytes(b"original\n")
                with self.assertRaisesRegex(ValueError,"ORIGINAL_OWNER_BACKUP_MISMATCH"):
                    preflight.check_restored_owner(a,b)
    def test_missing_backup_fails(self):
        with tempfile.TemporaryDirectory() as root:
            a,b=Path(root)/"owner",Path(root)/"backup"
            a.mkdir();b.mkdir()
            (a/"example.py").write_bytes(b"data")
            with patch.object(preflight,"FILES",("example.py",)):
                with self.assertRaisesRegex(ValueError,"ORIGINAL_OWNER_BACKUP_MISMATCH"):
                    preflight.check_restored_owner(a,b)

if __name__=="__main__":
    unittest.main()
