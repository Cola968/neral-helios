from __future__ import annotations
import unittest
from frontier350.score_chat350_capabilities import score_one,score,compare

class CapabilityTests(unittest.TestCase):
    def test_integer_last_number(self):
        self.assertTrue(score_one("37 + 58 = 95.",{"type":"integer","value":95}))
        self.assertTrue(score_one("Das Ergebnis ist 95,",{"type":"integer","value":95}))
        self.assertTrue(score_one("x = -7.",{"type":"integer","value":-7}))
        self.assertFalse(score_one("37 + 58 = 94.",{"type":"integer","value":95}))

    def test_integer_rejects_decimal_and_thousands_substrings(self):
        integer={"type":"integer","value":2}
        for response in ("0.2", "3.2", "0,2", "12.2", "1,234", "abc2", "12.02"):
            with self.subTest(response=response):
                self.assertFalse(score_one(response,integer))
        self.assertTrue(score_one("Das Ergebnis ist 2.",integer))
        self.assertTrue(score_one("x = +2,",integer))

    def test_contains(self):
        self.assertTrue(score_one("Die Schwerkraft zieht ihn.",{"type":"contains_any","values":["gravitation","schwerkraft"]}))
        self.assertTrue(score_one("The train leaves at nine.",{"type":"contains_all","values":["train","nine"]}))

    def test_report_and_compare(self):
        cases=[{"id":"a","category":"x","scoring":{"type":"exact","value":"OK"}},{"id":"b","category":"y","scoring":{"type":"integer","value":7}}]
        b=score({"rows":[{"id":"a","text":"NO"},{"id":"b","text":"7"}]},cases)
        c=score({"rows":[{"id":"a","text":"OK"},{"id":"b","text":"7"}]},cases)
        self.assertTrue(compare(b,c)["capability_no_regression"])

if __name__=="__main__":unittest.main()
