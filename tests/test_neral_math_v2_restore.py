import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from scripts.neral_math_v2_restore import restore, restore_review_sources

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
    def test_composed_review_sources_allow_only_exact_lf_pins(self):
        with TemporaryDirectory() as t:
            root=Path(t); source=root/'historical.py'; source.write_bytes(b'a=1\r\nb=2\r\n')
            review=root/'final.review.json'
            review.write_text(json.dumps({'sources':{'historical.py':hashlib.sha256(b'a=1\nb=2\n').hexdigest()}}))
            report=restore_review_sources(root,review)
            self.assertEqual(report['crlf_only_repaired'],1)
            self.assertEqual(source.read_bytes(),b'a=1\nb=2\n')
            source.write_bytes(b'changed\n')
            with self.assertRaisesRegex(ValueError,'non-CRLF drift'):
                restore_review_sources(root,review)

    def test_missing_file_refused(self):
        with TemporaryDirectory() as t:
            with self.assertRaisesRegex(ValueError,'Missing'):
                restore(Path(t)/'lost','0'*64)
if __name__=='__main__':unittest.main()
