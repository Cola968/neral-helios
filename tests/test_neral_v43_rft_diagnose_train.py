import hashlib, json, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
from scripts import neral_v43_rft_train as t
from scripts import neral_v43_rft_diagnose as d

class TestTrain(unittest.TestCase):
    def test_checkpoint_inventory_rejects_adapter(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp); ck=p/'ck';ck.mkdir(); expected={}
            for name in ('config.json','tokenizer.json','model.safetensors.index.json','adapter_config.json'):
                file=ck/name;file.write_bytes(b'{}');expected[name]=hashlib.sha256(file.read_bytes()).hexdigest()
            m=p/'manifest.json';m.write_text(json.dumps(expected));sha=hashlib.sha256(m.read_bytes()).hexdigest()
            with self.assertRaises(ValueError):t.checkpoint_check(ck,m,sha)
    def test_cpu_default_imports_no_gpu_libraries(self):
        import subprocess,sys
        r=subprocess.run([sys.executable,'-c','from scripts import neral_v43_rft_train; import sys; assert not any(x in sys.modules for x in ("torch","peft","trl","transformers"))'],capture_output=True)
        self.assertEqual(r.returncode,0,r.stderr.decode())
    def test_monitor_cap_fails_closed(self):
        w=t.Watch();w.failed=True;w.samples=1
        with self.assertRaises(RuntimeError):w.check()
    def test_monitor_missing_sample_fails_closed(self):
        with self.assertRaises(RuntimeError):t.Watch().check()
    def test_gpu_gate_requires_explicit_approval(self):
        from types import SimpleNamespace
        with self.assertRaises(ValueError):t.train_smoke(SimpleNamespace(confirm=None),[],{}, {})
    def test_wrong_host_refuses_gpu(self):
        with patch.object(t.os,'name','posix'):
            with self.assertRaises(RuntimeError):t.owner_stopped()


class TestSmokeTelemetry(unittest.TestCase):
    def test_start_loss_is_first_microbatch_token_mean_and_duration_is_one_step(self):
        ticks=iter([100.0,112.5])
        m=t.SmokeTelemetry(clock=lambda:next(ticks))
        m.begin()
        m.record_first_loss(.75,supervised_tokens=5,normalizer=20)
        for _ in range(9):m.substep()
        m.end()
        result=m.report()
        self.assertEqual(result['start_loss'],3.0)
        self.assertEqual(result['start_loss_definition'],'first_pre_update_microbatch_mean_supervised_token_ce')
        self.assertEqual(result['step_duration_s'],12.5)
        self.assertEqual(result['gradient_accumulation_microbatches'],10)

    def test_missing_or_nonfinite_measurement_fails_closed(self):
        for invalid in (float('nan'),float('inf'),-1.0):
            m=t.SmokeTelemetry(clock=lambda:1.0);m.begin()
            with self.assertRaises(ValueError):m.record_first_loss(invalid,1,1)
        m=t.SmokeTelemetry(clock=lambda:1.0);m.begin()
        with self.assertRaises(ValueError):m.record_first_loss(1.0,0,1)
        with self.assertRaises(RuntimeError):m.report()

    def test_repeated_step_and_incomplete_accumulation_are_refused(self):
        m=t.SmokeTelemetry(clock=lambda:1.0);m.begin()
        with self.assertRaises(RuntimeError):m.begin()
        m.record_first_loss(1.0,1,1)
        with self.assertRaises(RuntimeError):m.record_first_loss(1.0,1,1)
        with self.assertRaises(RuntimeError):m.end()
        for _ in range(9):m.substep()
        m.end()
        with self.assertRaises(RuntimeError):m.end()
        with self.assertRaises(RuntimeError):m.substep()

class TestDiagnosis(unittest.TestCase):
    def test_ast_diagnostic_no_text_leak(self):
        tags=d.structural_code('def f(x: int):\n    return [y for y in x]')
        self.assertEqual(set(tags),{'annotations','comprehensions'})
    def test_diagnose_counts_without_model_outputs(self):
        cases=[{'id':str(i),'prompt':f'private{i}','category':'programming' if i%2 else 'math_logic','family':'test'} for i in range(40)]
        status=['verified']*24+['counterexample_unresolved']*6+['verification_incomplete']*10
        result=[]
        for engine in ['A','C']:
            for i,c in enumerate(cases):
                result.append({'id':c['id'],'engine':engine,'commit':d.PIN,'prompt_sha256':hashlib.sha256(c['prompt'].encode()).hexdigest(),'verification_status':status[i] if engine=='C' else 'verification_not_requested','first_draft':'PRIVATE','text':'PRIVATE','generation_rounds':1})
        r=d.analyse(cases,result)
        self.assertEqual(r['groups']['counterexample_unresolved']['count'],6)
        self.assertEqual(r['groups']['verification_incomplete']['count'],10)
        self.assertNotIn('PRIVATE',json.dumps(r))
        self.assertFalse(r['historical_repair_text_available'])
if __name__=='__main__':unittest.main()
