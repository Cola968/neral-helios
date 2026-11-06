"""Fail-closed tests for reviewed v3 algebra two-line conversion."""
import unittest
from scripts.neral_v3_two_line_equations import (
    replacement_for_source_line, adapt_row, source_section, SOURCE_PINNED,
)
from scripts.neral_math_v2_builder import generate, TRAIN_SEED, NUMBER_ONLY_SYSTEM
from scripts.neral_math_v3_balanced_pack import new_equations


class TwoLineEquationTests(unittest.TestCase):
    def test_all_170_exact_source_equations_convert_with_independent_proof(self):
        old=[r for r in generate(TRAIN_SEED) if r['family']=='equation']
        new=new_equations()
        self.assertEqual((len(old),len(new)),(70,100))
        for line in range(127,297):
            if line<=196:
                item=old[line-127]
                completion=item['expected']
            else:
                item=new[line-197]
                completion=item['completion']
            row={'messages':[{'role':'system','content':NUMBER_ONLY_SYSTEM},
                             {'role':'user','content':item['prompt']},
                             {'role':'assistant','content':completion}]}
            changed, was_changed=adapt_row(row,line)
            self.assertTrue(was_changed)
            self.assertNotEqual(changed['messages'][1]['content'],item['prompt'])
            text=changed['messages'][2]['content']
            self.assertEqual(len(text.splitlines()),2)
            self.assertEqual(text.splitlines()[-1], 'x = '+item['expected'])
            self.assertIn('=',text.splitlines()[0])

    def test_non_equation_core_is_unchanged(self):
        original={'messages':[{'role':'system','content':'NERAL'},
                              {'role':'user','content':'Who are you?'},
                              {'role':'assistant','content':'I am NERAL.'}]}
        for line in (1,86,87,126,297,384):
            row,modified=adapt_row(original,line)
            self.assertFalse(modified)
            self.assertEqual(row,original)

    def test_bad_origin_prompt_or_answer_fails_closed(self):
        r=new_equations()[0];row={'messages':[
            {'role':'system','content':NUMBER_ONLY_SYSTEM},
            {'role':'user','content':r['prompt']},
            {'role':'assistant','content':r['completion']}]}
        copy={'messages':[dict(x) for x in row['messages']]}
        copy['messages'][2]['content']='x=999'
        with self.assertRaises(ValueError):adapt_row(copy,197)
        copy['messages'][2]['content']=r['completion']
        copy['messages'][1]['content']='x = 123'
        with self.assertRaises(ValueError):adapt_row(copy,197)

    def test_stratification_source_sections_and_digest_pins(self):
        self.assertEqual([source_section(n) for n in (1,127,196,197,296,297)],
                         ['other','equation_old','equation_old','equation_new',
                          'equation_new','other'])
        self.assertEqual({k:v['count'] for k,v in SOURCE_PINNED.items()},
                         {'train':320,'validation':64})
        self.assertTrue(all(len(v['sha256'])==64 and len(v['review_sha256'])==64
                            for v in SOURCE_PINNED.values()))


if __name__=='__main__':
    unittest.main()
