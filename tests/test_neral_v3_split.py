"""CPU contracts for deterministic, provenance-preserving Math-v3 A/B split."""
import hashlib
import json
import unittest

from scripts.neral_v3_split import (
    SECTION_SPECS, EXPECTED_V3_DATA_SHA256, EXPECTED_V3_REVIEW_SHA256,
    split_indices, slice_review, verify_original_rows
)


class V3SplitTests(unittest.TestCase):
    def setUp(self):
        self.lines = [
            json.dumps({'messages': [
                {'role': 'system', 'content': 'NERAL'},
                {'role': 'user', 'content': f'Unique task {i}'},
                {'role': 'assistant', 'content': str(i)}]}, separators=(',', ':'))
            for i in range(384)
        ]
        self.rows = [{'line': i + 1, 'row_sha256': hashlib.sha256(line.encode()).hexdigest(),
                      'evidence': ['deterministic-test'], 'finding': 'checked'}
                     for i,line in enumerate(self.lines)]
        self.review = {'schema': 1,
                       'dataset_sha256': hashlib.sha256(('\n'.join(self.lines)+'\n').encode()).hexdigest(),
                       'sources': {'scripts/frozen.py': '0'*64}, 'rows': self.rows}

    def test_stratified_320_train_64_validation_and_no_overlap(self):
        train, val = split_indices(self.lines)
        self.assertEqual(len(train), 320)
        self.assertEqual(len(val), 64)
        self.assertFalse(set(train) & set(val))
        self.assertEqual(sorted(train+val), list(range(384)))
        self.assertEqual([(n, total, take) for n,total,take in SECTION_SPECS],
                         [('core',86,72),('legacy',40,34),('equation_old',70,58),
                          ('equation_new',100,84),('other_math',88,72)])

    def test_split_is_identical_between_runs_and_changed_row_moves_hash(self):
        a = split_indices(self.lines)
        self.assertEqual(a, split_indices(list(self.lines)))
        modified = list(self.lines)
        modified[0] = modified[0].replace('Unique task 0','Unique task changed')
        self.assertNotEqual(
            hashlib.sha256(('\n'.join(modified)+'\n').encode()).hexdigest(),
            self.review['dataset_sha256'])

    def test_manifest_preserves_provenance_with_correct_line_numbers(self):
        train, val = split_indices(self.lines)
        for selected in (train, val):
            out_lines, new_review = slice_review(self.lines, self.review, selected)
            raw = ('\n'.join(out_lines)+'\n').encode()
            self.assertEqual(len(out_lines), len(selected))
            self.assertEqual(new_review['dataset_sha256'], hashlib.sha256(raw).hexdigest())
            self.assertEqual([x['line'] for x in new_review['rows']], list(range(1,len(selected)+1)))
            self.assertTrue(all(x['source_line'] in range(1,385) for x in new_review['rows']))
            self.assertEqual(new_review['sources'], self.review['sources'])

    def test_modified_answer_is_rejected_by_row_checksum(self):
        verify_original_rows(self.lines, self.review)
        bad = list(self.lines)
        bad[15] = bad[15].replace('"15"','"999"')
        with self.assertRaises(ValueError):
            verify_original_rows(bad, self.review)

    def test_expected_input_hashes_have_full_sha256(self):
        self.assertEqual(len(EXPECTED_V3_DATA_SHA256),64)
        self.assertEqual(len(EXPECTED_V3_REVIEW_SHA256),64)


if __name__=='__main__':
    unittest.main()
