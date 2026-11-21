import json
import tempfile
import unittest
from pathlib import Path
from scripts.neral_qlora_verified_v39 import load_verified_dataset
from neral_verified_notes import exact_note_lookup

SEED=Path(__file__).resolve().parents[1]/'datasets'/'neral_v39_seed97.jsonl'

class VerifiedNotesCPU(unittest.TestCase):
    def test_public_benchmark_exact_match_only_and_correct_provenance(self):
        rows=load_verified_dataset(SEED)
        first=next(x for x in rows if x['meta']['category']=='mathematics')
        note=exact_note_lookup(first['instruction'],SEED)
        self.assertEqual(note['text'],first['response'])
        self.assertFalse(note['verifier_ok'])
        self.assertEqual(note['source'],'v39_benchmark')
        self.assertIsNone(exact_note_lookup(first['instruction']+' Bitte ausführlicher.',SEED))

    def test_refuses_stale_web_answers(self):
        first=next(x for x in load_verified_dataset(SEED) if x['meta']['category']=='web')
        self.assertIsNone(exact_note_lookup(first['instruction'],SEED))

    def test_rejects_unverified_and_privacy_sensitive_local_examples(self):
        rows=load_verified_dataset(SEED)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'data.jsonl'
            row={**rows[0], 'meta':{**rows[0]['meta'],'source':'live_verified','verifier_ok':False}}
            path.write_text(json.dumps(row)+'\n',encoding='utf-8')
            self.assertIsNone(exact_note_lookup(row['instruction'],path))
            row['meta']['verifier_ok']=True
            row['meta']['verification_kind']='independent_runtime_proof'
            row['meta']['category']='offline_verified'
            path.write_text(json.dumps(row)+'\n',encoding='utf-8')
            self.assertIsNotNone(exact_note_lookup(row['instruction'],path))
            row['response']=r'C:\Users\Alice\Private\password.txt'
            path.write_text(json.dumps(row)+'\n',encoding='utf-8')
            self.assertIsNone(exact_note_lookup(row['instruction'],path))

if __name__=='__main__':unittest.main()
