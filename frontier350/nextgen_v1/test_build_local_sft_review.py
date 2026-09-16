import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import build_local_sft_review as viewer


class ReviewPageTests(unittest.TestCase):
    def test_escape_untrusted_dialogue_and_never_approve(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            train=root/'train.jsonl'; valid=root/'valid.jsonl'
            bad="</pre><script>alert('x')</script>"
            train.write_text(json.dumps({'id':'a','source_id':'s','user':'What?', 'assistant':bad})+'\n',encoding='utf8')
            valid.write_text(json.dumps({'id':'b','source_id':'s','user':'Another?', 'assistant':'Safe.'})+'\n',encoding='utf8')
            flags=root/'flags.json'; stage=root/'stage.json'
            flags.write_text(json.dumps({'approved_for_training':False,'training_started':False,
                                        'training_rows':1,'validation_rows':1,
                                        'flagged_ids_no_text':[{'id':'a','reason':'requires_review'}]}),encoding='utf8')
            stage.write_text(json.dumps({'approved_for_training':False}),encoding='utf8')
            def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
            with patch.object(viewer,'TRAIN_SHA',sha(train)),patch.object(viewer,'VALID_SHA',sha(valid)):
                html=viewer.construct(train,valid,flags,stage)
                self.assertIn('&lt;script&gt;',html)
                self.assertNotIn('<script>',html)
                self.assertIn('FLAG: requires_review',html)
                self.assertIn('NICHT FREIGEGEBEN',html)
                flags.write_text(json.dumps({'approved_for_training':True,'training_started':False,
                                            'training_rows':1,'validation_rows':1,
                                            'flagged_ids_no_text':[]}),encoding='utf8')
                with self.assertRaisesRegex(ValueError,'review-only identity mismatch'):
                    viewer.construct(train,valid,flags,stage)


if __name__=='__main__':unittest.main()
