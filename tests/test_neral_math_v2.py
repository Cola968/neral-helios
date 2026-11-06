"""CPU-only math-v2 data, quality and trainer safety contracts."""
from __future__ import annotations
import hashlib
import json
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from scripts import neral_math_v2_builder as b
from scripts import neral_math_v2_patcher as t
from scripts import neral_math_v2_eval as e


def fake_pair(root:Path,name:str,count:int):
    data=root/(name+'.jsonl');review=root/(name+'.review.json')
    lines=[]
    for i in range(count):
        lines.append(b.json_line({'messages':[{'role':'system','content':'You are NERAL.'},
                   {'role':'user','content':f'Unique audited {name} user problem {i}?'},
                   {'role':'assistant','content':f'Reviewed answer for {name} {i}.'}]}))
    raw=('\n'.join(lines)+'\n').encode('utf8')
    data.write_bytes(raw)
    manifest={'schema':1,'dataset_sha256':b.sha(raw),'sources':{f'{name}_source.py':b.sha(name.encode())},
       'rows':[{'line':i+1,'row_sha256':b.sha(line.encode()),'evidence':['fixture source checked'],
                'finding':'Fixture reviewed'} for i,line in enumerate(lines)]}
    review.write_text(json.dumps(manifest),encoding='utf8')
    return data,review


class MathV2Contracts(unittest.TestCase):
    def test_train_count_and_independent_rational_arithmetic(self):
        rows=b.generate(b.TRAIN_SEED)
        self.assertEqual(len(rows),560)
        self.assertEqual(len({b.norm(x['prompt']) for x in rows}),560)
        self.assertTrue(all(b.validate_expected(x) for x in rows))
        self.assertEqual({x['family'] for x in rows},set(b.FAMILIES))
        self.assertTrue(all(b.to_qlora(x)['messages'][-1]['content']==x['expected'] for x in rows))

    def test_train_and_new_dev_have_disjoint_questions(self):
        train={b.norm(x['prompt']) for x in b.generate(b.TRAIN_SEED)}
        dev=e.cases()
        self.assertEqual(len(dev),80)
        self.assertEqual(len({x['id'] for x in dev}),80)
        self.assertFalse(train & {b.norm(x['prompt']) for x in dev})

    def test_structured_arithmetic_validator_rejects_false_label(self):
        item=b.generate(b.TRAIN_SEED)[7]
        self.assertTrue(b.validate_expected(item))
        self.assertFalse(b.validate_expected({**item,'expected':'123456789'}))

    def test_strict_number_only_scoring(self):
        self.assertEqual(e.evaluate_text(' 5/2 ', '2.5'),{'correct':True,'format_valid':True})
        self.assertFalse(e.evaluate_text('Ergebnis: 2.5', '5/2')['format_valid'])
        self.assertFalse(e.evaluate_text('4 + 2 = 6', '6')['format_valid'])
        self.assertFalse(e.evaluate_text('12.\nErklärung','12')['format_valid'])

    def test_all_models_required_and_no_promotion_claim(self):
        rs={k:[{'id':x['id'],'correct':i%3==0,'format_valid':i%2==0}
               for i,x in enumerate(e.cases())]
            for k in ('base','old_rank32','new_rank16_v2')}
        report=e.score(rs)
        self.assertEqual(report['question_count'],80)
        self.assertFalse(report['general_improvement_proven'])
        self.assertFalse(report['weights_promoted'])
        self.assertFalse(report['independent_holdout'])
        self.assertEqual(report['scores']['base']['total'],80)
        with self.assertRaises(ValueError):e.score({**rs,'unexpected':[]})

    def test_patcher_must_reject_unpinned_trainer_and_existing_destination(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'old.py';out=Path(d)/'new.py'
            p.write_text('learning_rate=2e-4, bf16=True\n')
            with self.assertRaisesRegex(ValueError,'Pinned QLoRA trainer'):
                t.patch(p,out)
            self.assertFalse(out.exists())
            out.write_text('untouched')
            with self.assertRaisesRegex(ValueError,'Unsafe scratch trainer'):
                t.patch(p,out)
            self.assertEqual(out.read_text(),'untouched')

    def test_tampered_review_or_row_refused(self):
        with tempfile.TemporaryDirectory() as d:
            p,r=fake_pair(Path(d),'old',98)
            b.read_and_verify(p,r)
            rj=json.loads(r.read_text());rj['rows'][0]['row_sha256']='0'*64
            r.write_text(json.dumps(rj))
            with self.assertRaisesRegex(ValueError,'Row hash mismatch'):
                b.read_and_verify(p,r)

    def test_744_rows_are_built_locally_without_any_training(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            old,oldr=fake_pair(root,'old',98)
            reviewed,reviewr=fake_pair(root,'project',86)
            source=root/'neral_math_v2_builder.py';source.write_bytes(Path(b.__file__).read_bytes())
            out=root/'new'
            with patch.object(b,'LEGACY_SHA',b.sha(old.read_bytes())),patch.object(b,'LEGACY_REVIEW_SHA',b.sha(oldr.read_bytes())):
                summary=b.build(old,oldr,reviewed,reviewr,out,source,set())
            self.assertEqual(summary['total'],744)
            self.assertEqual(summary['independently_checked_numeric'],560)
            self.assertEqual(summary['legacy_math'],98)
            self.assertEqual(summary['reviewed_project'],86)
            self.assertFalse(summary['training_performed'])
            self.assertFalse(summary['approved_for_training'])
            lines=(out/'train.jsonl').read_text(encoding='utf8').splitlines()
            self.assertEqual(len(lines),744)
            self.assertEqual(len(json.loads((out/'train.review.json').read_text())['rows']),744)
            with patch.object(b,'LEGACY_SHA',b.sha(old.read_bytes())),patch.object(b,'LEGACY_REVIEW_SHA',b.sha(oldr.read_bytes())):
                with self.assertRaisesRegex(ValueError,'Output exists'):
                    b.build(old,oldr,reviewed,reviewr,out,source,set())

    def test_pinned_holdout_is_eight_jsonl_rows_not_one_json_object(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'holdout.jsonl'
            path.write_text(''.join(json.dumps({'prompt':f'unseen protected question {i}'})+'\n' for i in range(8)))
            self.assertEqual(len(b.load_holdout_prompts(path)),8)
            with self.assertRaisesRegex(ValueError,'exactly eight'):
                path.write_text(json.dumps({'prompt':'only one object'}))
                b.load_holdout_prompts(path)

    def test_frozen_holdout_prompt_blocks_without_data_output(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            old,oldr=fake_pair(root,'old',98)
            reviewed,reviewr=fake_pair(root,'project',86)
            source=root/'builder.py';source.write_bytes(Path(b.__file__).read_bytes())
            q={'unique audited old user problem 0?'}
            with patch.object(b,'LEGACY_SHA',b.sha(old.read_bytes())),patch.object(b,'LEGACY_REVIEW_SHA',b.sha(oldr.read_bytes())):
                with self.assertRaisesRegex(ValueError,'holdout conflict'):
                    b.build(old,oldr,reviewed,reviewr,root/'refuse',source,{b.norm(x) for x in q})
            self.assertFalse((root/'refuse').exists())

if __name__=='__main__':unittest.main()
