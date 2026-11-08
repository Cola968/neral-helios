"""CPU RED: finite validation checkpoints and safe best-restore on QLoRA.

No CUDA/weights. Production default must remain old train behavior; only
explicit future experimental candidates may enable checkpoint selection.
"""
import inspect
import tempfile
import unittest
from pathlib import Path

from scripts.neral_training_checkpoint_policy import (
    options_for_training, validate_best_checkpoint,
)


class CheckpointTrainingTests(unittest.TestCase):
    def test_existing_training_defaults_unchanged(self):
        from scripts.neral_v6_r16_gpu import _gpu_train
        signature=inspect.signature(_gpu_train)
        self.assertEqual(signature.parameters["checkpoint_interval"].default,None)
        ordinary=options_for_training(steps=120,interval=None)
        self.assertEqual(ordinary,{"eval_strategy":"no","save_strategy":"no"})
        self.assertEqual(signature.parameters["lr"].default,1e-5)
        self.assertEqual(signature.parameters["steps"].default,120)

    def test_explicit_20step_selection_loads_best_eval_adapter_not_last(self):
        cfg=options_for_training(steps=120,interval=20)
        self.assertEqual(cfg["eval_strategy"],"steps")
        self.assertEqual(cfg["save_strategy"],"steps")
        self.assertEqual(cfg["eval_steps"],20)
        self.assertEqual(cfg["save_steps"],20)
        self.assertTrue(cfg["load_best_model_at_end"])
        self.assertEqual(cfg["metric_for_best_model"],"eval_loss")
        self.assertFalse(cfg["greater_is_better"])
        self.assertEqual(cfg["save_total_limit"],2)

    def test_unaligned_excessive_or_invalid_checkpoint_rejected(self):
        for bad in (0,1,2,3,5,13,19,120,121,60.0,True,"20",None):
            if bad is None: continue
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    options_for_training(steps=120,interval=bad)
        self.assertEqual(options_for_training(steps=120,interval=20)["eval_steps"],20)

    def test_saved_best_path_is_strictly_private_regular_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);folder=root/"trainer-temp";folder.mkdir()
            best=folder/"checkpoint-40";best.mkdir()
            self.assertEqual(validate_best_checkpoint(best,0.923,folder,steps=120,interval=20),40)
            for bad in (
                folder/"checkpoint-41",
                folder/"checkpoint-999",
                root/"checkpoint-40",
                folder/"custom",
            ):
                bad.mkdir(exist_ok=True)
                with self.assertRaises(ValueError):
                    validate_best_checkpoint(bad,.6,folder,steps=120,interval=20)
            with self.assertRaises(ValueError):
                validate_best_checkpoint(best,float("nan"),folder,steps=120,interval=20)
            with self.assertRaises(ValueError):
                validate_best_checkpoint(best,0.5,folder,steps=120,interval=25)

    def test_checkpointed_training_path_has_best_restore_before_adapter_save(self):
        from scripts.neral_v6_r16_gpu import _gpu_train
        from scripts.neral_v61_replay_train import main
        source=inspect.getsource(_gpu_train)
        self.assertIn("options_for_training(steps=steps,interval=checkpoint_interval,",source)
        self.assertIn("max_steps_limit=max_steps_limit",source)
        self.assertEqual(inspect.signature(_gpu_train).parameters["max_steps_limit"].default,120)
        self.assertIn("TRAIN_NERAL_V8_SELFVERIFY_R16_240_EXPERIMENTAL",source)
        self.assertIn("NERAL_V8_SELFVERIFY_R16_240_UNRELEASED",source)
        self.assertIn("validate_best_checkpoint(",source)
        self.assertIn("trainer.state.best_model_checkpoint",source)
        self.assertLess(source.index("validate_best_checkpoint("),source.index("policy.save_pretrained("))
        self.assertIn("checkpoint_interval",inspect.getsource(main))


if __name__=="__main__":
    unittest.main()
