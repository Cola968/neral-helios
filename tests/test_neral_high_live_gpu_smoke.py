"""Real RTX 4070 opt-in HIGH acceptance reports MUST stay scalar-only."""
import json
import unittest

from scripts.neral_high_live_gpu_smoke import SmokeError, evaluate_response_metadata, make_report

def reply(model,mode,profile="legacy",*,text="A safe completed answer.",generated=True):
    return {"text":text, "mode":mode,"model":model,
            "inference_profile":profile,"model_origin":"local_foundation_not_neral_adapter",
            "adapter_loaded":False,"model_generated":generated}

class HighLiveSmokeContracts(unittest.TestCase):
    def test_valid_low_high_low_and_direct_tool_provenance(self):
        inputs=[
            ("LOW","qwen3:8b",reply("qwen3:8b","LOW")),
            ("HIGH","qwen3.5:9b",reply("qwen3.5:9b","HIGH","measured_nonthinking_2048")),
            ("LOW","qwen3:8b",reply("qwen3:8b","LOW")),
            ("HIGH","qwen3.5:9b",reply("qwen3.5:9b","HIGH","measured_nonthinking_2048",text="221",generated=False)),
        ]
        rows=[evaluate_response_metadata(mode,model,payload,1.5)
              for mode,model,payload in inputs]
        result=make_report(rows,{"peak_used_mib":7600,"total_mib":12282,
                                 "sample_count":200})
        self.assertEqual(result["status"],"NERAL_HIGH_LIVE_RTX4070_OPT_IN_SMOKE_ONLY")
        self.assertEqual(result["requests_completed"],4)
        self.assertEqual(result["models"],["qwen3:8b","qwen3.5:9b"])
        self.assertFalse(result["release_validated"])
        self.assertFalse(result["neural_improvement_proven"])
        self.assertFalse(result["model_promotion_authorized"])
        self.assertEqual(result["sampled_peak_vram_gib"],round(7600/1024,3))
        self.assertNotIn("A safe completed answer",json.dumps(result))
        self.assertNotIn("221",json.dumps(result))

    def test_wrong_models_adapter_and_empty_or_incomplete_refused(self):
        bad=[
            ("HIGH","qwen3.5:9b",reply("qwen3:8b","HIGH","measured_nonthinking_2048")),
            ("HIGH","qwen3.5:9b",reply("qwen3.5:9b","HIGH")),
            ("LOW","qwen3:8b",{**reply("qwen3:8b","LOW"),"adapter_loaded":True}),
            ("LOW","qwen3:8b",{**reply("qwen3:8b","LOW"),"model_origin":"local_neral_lora_experimental"}),
            ("HIGH","qwen3.5:9b",{**reply("qwen3.5:9b","HIGH","measured_nonthinking_2048"),"text":""}),
        ]
        for mode,model,payload in bad:
            with self.subTest(payload=payload),self.assertRaises(SmokeError):
                evaluate_response_metadata(mode,model,payload,2.0)
        with self.assertRaises(SmokeError):
            evaluate_response_metadata("LOW","qwen3:8b",reply("qwen3:8b","LOW"),-1.0)

    def test_missing_actual_vram_samples_refuse(self):
        good=[
            evaluate_response_metadata("LOW","qwen3:8b",reply("qwen3:8b","LOW"),1),
            evaluate_response_metadata("HIGH","qwen3.5:9b",reply("qwen3.5:9b","HIGH","measured_nonthinking_2048"),2),
            evaluate_response_metadata("LOW","qwen3:8b",reply("qwen3:8b","LOW"),1),
            evaluate_response_metadata("HIGH","qwen3.5:9b",reply("qwen3.5:9b","HIGH","measured_nonthinking_2048",text="221",generated=False),1),
        ]
        with self.assertRaises(SmokeError):
            make_report(good,{"peak_used_mib":7000,"total_mib":12282,"sample_count":0})

if __name__=="__main__":unittest.main()
