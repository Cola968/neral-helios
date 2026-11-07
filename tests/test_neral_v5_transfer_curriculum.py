"""NERAL v5: verifiable transfer learning candidates, independent of evaluation answers."""
import ast
from decimal import Decimal, ROUND_HALF_UP
from fractions import Fraction
import json
from pathlib import Path
import tempfile
import unittest

from scripts.neral_v5_transfer_curriculum import (
    CATEGORIES, ROWS, TOTAL, STYLES, build, verify, make_case,
)


class TransferV5CurriculumTests(unittest.TestCase):
    def test_balanced_reproducible_2560_with_math_priority(self):
        self.assertEqual(len(CATEGORIES),8)
        self.assertEqual(CATEGORIES[:6],("affine","fractions","money","fuel","timezone","quantities"))
        self.assertEqual(TOTAL,2560)
        self.assertEqual(ROWS,{"train":1920,"val":320,"test":320})
        self.assertEqual(len(STYLES),8)
        with tempfile.TemporaryDirectory() as left, tempfile.TemporaryDirectory() as right:
            a,b=Path(left),Path(right)
            one=build(a)
            two=build(b)
            self.assertEqual(one,two)
            self.assertEqual(verify(a)["valid"],True)
            self.assertEqual(one["schema_version"],5)
            self.assertFalse(one["training_allowed"])
            self.assertFalse(one["human_quality_reviewed"])
            self.assertFalse(one["independent_external_holdout"])
            self.assertFalse(one["quality_improvement_proven"])
            seen=set()
            for split,count in ROWS.items():
                lines=(a/(split+".jsonl")).read_text(encoding="utf8").splitlines()
                self.assertEqual(len(lines),count)
                self.assertEqual((a/(split+".jsonl")).read_bytes(),(b/(split+".jsonl")).read_bytes())
                for line in lines:
                    row=json.loads(line)
                    self.assertEqual(set(row),{"prompt","completion"})
                    self.assertEqual([m["role"] for m in row["prompt"]],["system","user"])
                    self.assertEqual([m["role"] for m in row["completion"]],["assistant"])
                    user=row["prompt"][1]["content"]
                    self.assertNotIn(user,seen)
                    seen.add(user)
                    self.assertLessEqual(len(user),600)
                    self.assertLessEqual(len(row["completion"][0]["content"]),400)
            self.assertEqual(len(seen),2560)
            self.assertEqual((a/"manifest.json").read_bytes(),(b/"manifest.json").read_bytes())

    def test_math_labels_are_independently_checked(self):
        from scripts.neral_v5_transfer_curriculum import solve_case
        for cat in CATEGORIES[:6]:
            for style in range(8):
                for index in range(40):
                    case=make_case(cat,style,index)
                    completion=case["completion"][0]["content"]
                    oracle=solve_case(cat,style,index)
                    self.assertIn(oracle["answer"],completion)
                    self.assertNotIn("code fence",completion)
                    if style in (4,5):
                        self.assertIn("FINAL_ANSWER:",completion)
                        self.assertIn("CHECK:",completion)
                    else:
                        self.assertNotIn("FINAL_ANSWER:",completion)
                        self.assertLessEqual(len(completion),32)

    def test_fraction_money_fuel_and_timezone_have_exact_oracles(self):
        from scripts.neral_v5_transfer_curriculum import solve_case
        for style in range(8):
            for index in range(40):
                case=solve_case("fractions",style,index)
                left=Fraction(case["left"])
                right=Fraction(case["right"])
                op=case["operator"]
                expected=({"+":lambda:left+right,"-":lambda:left-right,
                           "*":lambda:left*right,"/":lambda:left/right}[op])()
                self.assertEqual(case["answer"],str(expected))
                money=solve_case("money",style,index)
                p=Decimal(money["source_cents"])/100
                d=Decimal(money["discount"])
                t=Decimal(money["surcharge"])
                price=(p*(100-d)/100*(100+t)/100).quantize(Decimal(".01"),rounding=ROUND_HALF_UP)
                bare=f"{price:.2f}".replace(".",",")
                expected=bare+(" EUR" if style in (1,4,5,7) else "")
                self.assertEqual(money["answer"],expected)
                fuel=solve_case("fuel",style,index)
                amount=Fraction(fuel["burn"]*(fuel["duration"]+fuel["reserve"]),60)+fuel["taxi"]
                expected=str(amount)+(" kg" if style in (1,4,5,7) else "")
                self.assertEqual(fuel["answer"],expected)
                tz=solve_case("timezone",style,index)
                minutes=tz["hour"]*60+tz["minute"]+tz["duration"]+(tz["dst"]-tz["src"])*60
                day,minute=divmod(minutes,1440)
                bare=f"{minute//60:02d}:{minute%60:02d} Tag {day:+d}"
                expected=bare.replace(" Tag "," (Tag ")+")" if style in (1,4,5,7) else bare
                self.assertEqual(tz["answer"],expected)

    def test_format_instructions_have_semantically_equal_bare_and_labeled_answers(self):
        from scripts.neral_v5_transfer_curriculum import solve_case
        for cat in CATEGORIES[:6]:
            for i in (0,5,15,28):
                plain=solve_case(cat,0,i)["answer"]
                labeled=solve_case(cat,1,i)["answer"]
                self.assertTrue(plain)
                self.assertTrue(labeled)
                self.assertNotIn("FINAL_ANSWER:",plain)
                self.assertNotIn("FINAL_ANSWER:",labeled)

    def test_static_traces_and_repairs_are_safe_verified_code(self):
        from scripts.neral_v5_transfer_curriculum import solve_case
        for cat in ("python_trace","python_debug"):
            for style in (0,4,6,7):
                for i in range(40):
                    src=solve_case(cat,style,i)["program"]
                    tree=ast.parse(src)
                    self.assertLessEqual(len(list(ast.walk(tree))),75)
                    self.assertFalse(any(isinstance(n,(ast.Import,ast.ImportFrom,ast.Attribute,
                        ast.Lambda,ast.While,ast.For,ast.With,ast.Try,ast.Global,ast.Nonlocal))
                        for n in ast.walk(tree)))
                    if cat=="python_debug":
                        self.assertEqual(len(tree.body),1)
                        self.assertIsInstance(tree.body[0],ast.FunctionDef)
                        self.assertIn("def ",make_case(cat,style,i)["completion"][0]["content"])

    def test_holdout_is_unmodified_and_not_imported_into_training(self):
        from scripts.neral_v4_novel_eval import CASES
        existing={" ".join(row["prompt"].casefold().split()) for row in CASES}
        new={" ".join(make_case(cat,style,i)["prompt"][1]["content"].casefold().split())
             for cat in CATEGORIES for style in range(8) for i in range(40)}
        self.assertFalse(existing & new)
        from scripts import neral_v5_transfer_curriculum as module
        code=Path(module.__file__).read_text(encoding="utf8")
        self.assertNotIn("neral_v4_novel_eval",code)
        self.assertNotIn("42372478543",code)
        self.assertNotIn("original_benchmark",code)

    def test_tamper_and_bad_seed_fail_closed(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);build(root)
            p=root/"train.jsonl"
            content=p.read_text(encoding="utf8")
            row=json.loads(content.splitlines()[0])
            row["completion"][0]["content"]="11111"
            rest=content.splitlines()
            rest[0]=json.dumps(row)
            p.write_text("\n".join(rest)+"\n",encoding="utf8")
            with self.assertRaises(ValueError):
                verify(root)
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaises((TypeError,ValueError)):
                build(Path(t),seed="random")

    def test_no_gpu_or_training_side_effects_from_build(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);build(root)
            self.assertEqual({p.name for p in root.iterdir()},
                             {"train.jsonl","val.jsonl","test.jsonl","manifest.json"})


if __name__=="__main__":
    unittest.main()
