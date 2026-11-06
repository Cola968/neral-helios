import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from scripts.neral_math_v2_restore import restore

class WindowsPinRestoration(unittest.TestCase):
    def test_only_exact_crlf_translation_is_restored(self):
        with TemporaryDirectory() as t:
            p=Path(t)/'dataset.jsonl'
            original=b'{"a":1}\n{"b":2}\n'
            digest=hashlib.sha256(original).hexdigest()
            p.write_bytes(original.replace(b'\n',b'\r\n'))
            self.assertTrue(restore(p,digest))
            self.assertEqual(p.read_bytes(),original)
            self.assertFalse(restore(p,digest))
    def test_wrong_bytes_never_overwritten(self):
        with TemporaryDirectory() as t:
            p=Path(t)/'source.py';p.write_bytes(b'counterfeit\r\n')
            with self.assertRaisesRegex(ValueError,'non-CRLF drift'):
                restore(p,'f'*64)
            self.assertEqual(p.read_bytes(),b'counterfeit\r\n')
    def test_missing_file_refused(self):
        with TemporaryDirectory() as t:
            with self.assertRaisesRegex(ValueError,'Missing'):
                restore(Path(t)/'lost','0'*64)
if __name__=='__main__':unittest.main()
