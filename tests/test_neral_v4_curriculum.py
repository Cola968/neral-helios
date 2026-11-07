"""CPU-only regression contract for a reviewed-before-training NERAL v4 dataset."""
import json
import tempfile
import unittest
from pathlib import Path

from scripts.neral_v4_curriculum import build, verify, CATEGORIES, TOTAL


class VerifiedCurriculumV4Tests(unittest.TestCase):
    def test_fixed_seed_produces_isolated_balanced_splits(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            built = build(root, seed=20261107)
            checked = verify(root)
            self.assertEqual(built["rows"], checked["rows"])
            self.assertEqual(TOTAL, 2560)
            self.assertEqual(built["rows"], {"train": 1920, "val": 320, "test": 320})
            self.assertEqual(len(CATEGORIES), 8)
            self.assertEqual(built["categories"], list(CATEGORIES))
            self.assertFalse(built["training_allowed"])
            self.assertFalse(built["quality_improvement_proven"])
            self.assertFalse(built["independent_external_holdout"])
            self.assertTrue(checked["answers_verified"])

    def test_reproducible_byte_identical_output(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            left, right = Path(a), Path(b)
            build(left, seed=20261107)
            build(right, seed=20261107)
            for name in ("train.jsonl", "val.jsonl", "test.jsonl", "manifest.json"):
                self.assertEqual((left / name).read_bytes(), (right / name).read_bytes())

    def test_modified_label_rejected_even_when_schema_still_valid(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            build(root)
            path = root / "train.jsonl"
            rows = path.read_text(encoding="utf8").splitlines()
            first = json.loads(rows[0])
            first["completion"][0]["content"] = "x = 999999"
            rows[0] = json.dumps(first, ensure_ascii=False, sort_keys=True)
            path.write_text("\n".join(rows) + "\n", encoding="utf8")
            with self.assertRaises(ValueError):
                verify(root)

    def test_missing_example_and_cross_split_duplicate_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            build(root)
            p = root / "test.jsonl"
            rows = p.read_text(encoding="utf8").splitlines()
            rows[0] = (root / "train.jsonl").read_text(encoding="utf8").splitlines()[0]
            p.write_text("\n".join(rows) + "\n", encoding="utf8")
            with self.assertRaises(ValueError):
                verify(root)

    def test_no_shell_inference_download_or_training_as_build_side_effect(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            build(root)
            self.assertEqual(sorted(x.name for x in root.iterdir()),
                             ["manifest.json", "test.jsonl", "train.jsonl", "val.jsonl"])
            for split in ("train", "val", "test"):
                for line in (root / (split + ".jsonl")).read_text(encoding="utf8").splitlines():
                    row = json.loads(line)
                    self.assertEqual(set(row), {"completion", "prompt"})
                    self.assertEqual([m["role"] for m in row["prompt"]],
                                     ["system", "user"])
                    self.assertEqual([m["role"] for m in row["completion"]],
                                     ["assistant"])
                    self.assertNotIn("github_pat_", line)
                    self.assertNotIn("sk_live_", line)

    def test_rejects_invalid_seed(self):
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaises(ValueError):
                build(Path(t), seed="random")


    def test_debug_examples_cover_eight_different_correct_behaviors(self):
        import ast
        from scripts.neral_v4_curriculum import make_case
        for style in (0, 6, 7):
            for index in range(40):
                row=make_case("python_debug",style,index)
                source=row["completion"][0]["content"]
                tree=ast.parse(source)
                self.assertEqual(len(tree.body),1)
                fn=tree.body[0]
                self.assertIsInstance(fn,ast.FunctionDef)
                self.assertEqual(len(fn.body),1)
                self.assertFalse(any(isinstance(n,(ast.Import,ast.ImportFrom,ast.Attribute))
                                     for n in ast.walk(tree)))
                # Only execute these static, locally generated templates. Never
                # execute model-generated output or user-supplied Python here.
                allowed={"sum":sum,"range":range,"min":min,"max":max}
                scope={"__builtins__":allowed}
                exec(compile(tree,"<trusted-v4-test>","exec"),scope)
                func=scope[fn.name]
                variant=index%8
                if variant==0:
                    m,o=fn.body[0].value.left.right.value,fn.body[0].value.right.value
                    for v in (-5,0,3):
                        self.assertEqual(func(v),v*m+o)
                elif variant==1:
                    for n in (0,1,2,9):
                        self.assertEqual(func(n),sum(range(n)))
                elif variant==2:
                    for v in (-20,0,10,90):
                        self.assertEqual(func(v,-3,12),max(-3,min(v,12)))
                elif variant==3:
                    for v in ([],[4],[5,6,7]):
                        self.assertEqual(func(v),v[0] if v else None)
                elif variant==4:
                    for v,t in ((1,1),(0,1),(2,1),(-4,-4)):
                        self.assertEqual(func(v,t),v>=t)
                elif variant==5:
                    for v in ([],[-2,-7],[3,5,1],[0]):
                        self.assertEqual(func(v),max(v) if v else 0)
                elif variant==6:
                    for v in ("","a","aa","abcba","abca"):
                        self.assertEqual(func(v),v==v[::-1])
                else:
                    divisor=fn.body[0].value.left.right.value
                    for v in (-divisor,0,1,2*divisor,2*divisor+1):
                        self.assertEqual(func(v),v%divisor==0)

    def test_money_data_uses_explicit_cent_rounding(self):
        import re
        from decimal import Decimal, ROUND_HALF_UP
        from scripts.neral_v4_curriculum import make_case
        observed_fractional_cent=False
        for i in range(40):
            row=make_case("percent",0,i)
            user=row["prompt"][1]["content"]
            result=row["completion"][0]["content"]
            self.assertIn("auf Cent",user)
            m=re.search(r"Artikelpreis (\d+,\d{2}) EUR; erst (\d+)% Rabatt, "
                        r"dann (\d+)% Zuschlag",user)
            self.assertIsNotNone(m)
            price,discount,surcharge=m.groups()
            exact=(Decimal(price.replace(",","."))*
                   (Decimal(100)-Decimal(discount))/Decimal(100)*
                   (Decimal(100)+Decimal(surcharge))/Decimal(100))
            expected=exact.quantize(Decimal("0.01"),rounding=ROUND_HALF_UP)
            self.assertEqual(result,f"{expected:.2f} EUR".replace(".",","))
            observed_fractional_cent |= exact!=expected
        self.assertTrue(observed_fractional_cent,
                        "Dataset must contain genuinely nontrivial rounding examples")


if __name__ == "__main__":
    unittest.main()
