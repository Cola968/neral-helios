import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from scripts.neral_v310_freeze_holdout100 import generate,main
from scripts.neral_qlora_verified_v39 import load_verified_dataset

SEED=Path(__file__).resolve().parents[1]/'datasets'/'neral_v39_seed97.jsonl'

class HoldoutFreezeCPU(unittest.TestCase):
    def test_100_unique_questions_not_in_training(self):
        train=load_verified_dataset(SEED)
        train_ids={x['meta']['instruction_sha256'] for x in train}
        for seed in ('ab'*32,'21'*32):
            cases=generate(seed,train_ids)
            self.assertEqual(len(cases),100)
            self.assertEqual(len({x['question_sha256'] for x in cases}),100)
            self.assertFalse(train_ids & {x['question_sha256'] for x in cases})
            self.assertEqual({c:sum(x['category']==c for x in cases)
                for c in ('mathematics','logic','programming','knowledge','web')},
                {c:20 for c in ('mathematics','logic','programming','knowledge','web')})

    def test_freeze_once_with_digest_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'holdout100.jsonl'
            self.assertEqual(main(['--train',str(SEED),'--out',str(path)]),0)
            self.assertEqual(len(path.read_text(encoding='utf-8').splitlines()),100)
            report=json.loads(path.with_suffix('.manifest.json').read_text())
            self.assertEqual(report['sha256'],hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertTrue(report['no_train_question_overlap'])
            before=path.read_bytes()
            with self.assertRaises(FileExistsError):
                main(['--train',str(SEED),'--out',str(path)])
            self.assertEqual(before,path.read_bytes())

if __name__=='__main__':unittest.main()
