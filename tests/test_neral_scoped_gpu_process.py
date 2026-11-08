"""Never terminate unrelated Python processes to free NERAL benchmark VRAM."""
import unittest
from pathlib import Path
from unittest.mock import Mock
from scripts.neral_scoped_gpu_process import (
    gpu_python_pids, classify_neral_candidate, stop_verified,
)

EXPECTED=r"C:\Users\test\NERAL-QLoRA-Research\training-env-py312\Scripts\python.exe"

def fake_process(path=EXPECTED,args=None,age=900,cpu=0,now=10000):
    p=Mock()
    p.exe.return_value=path
    p.cmdline.return_value=args if args is not None else [
        EXPECTED, "-m", "neral_local_chat_api"
    ]
    p.create_time.return_value=now-age
    p.cpu_percent.return_value=cpu
    p.status.return_value="sleeping"
    p.is_running.return_value=True
    return p

class ScopedProcessTests(unittest.TestCase):
    def test_only_gpu_python_pids_read_from_nvidia_snapshot(self):
        raw=(
            "101, C:\\Tools\\python.exe, N/A\n"
            "102, C:\\Windows\\dwm.exe, N/A\n"
            "103, C:\\NERAL\\pythonw.exe, 5100 MiB\n"
            "bad, C:\\Users\\python.exe, N/A\n"
        )
        self.assertEqual(gpu_python_pids(raw),[101,103])
        self.assertEqual(gpu_python_pids("not supported"),[])
        self.assertEqual(gpu_python_pids("101, C:\\Tools\\python.exe, N/A\n101, C:\\Tools\\python.exe, N/A"),[101])

    def test_pinned_older_idle_neral_python_is_in_scope(self):
        proc=fake_process()
        info=classify_neral_candidate(proc,EXPECTED,now=10000)
        self.assertTrue(info["verified"])
        self.assertTrue(info["pinned_executable"])
        self.assertTrue(info["project_command"])
        self.assertFalse(info["runner_process"])
        self.assertTrue(info["alive"])
        self.assertFalse(info["training_process"])
        self.assertNotIn("test",str(info))
        self.assertNotIn("Scripts",str(info))

    def test_other_python_cannot_be_touched(self):
        proc=fake_process(path=r"C:\Users\test\unrelated\python.exe")
        evidence=classify_neral_candidate(proc,EXPECTED,now=10000)
        self.assertFalse(evidence["verified"])
        self.assertFalse(evidence["pinned_executable"])
        self.assertTrue(evidence["project_command"])
        self.assertNotIn("unrelated",str(evidence))
        proc=fake_process(args=[EXPECTED,"-c","print('hi')"])
        self.assertFalse(classify_neral_candidate(proc,EXPECTED,now=10000)["verified"])

    def test_windows_venv_launcher_resolves_to_base_python_binary(self):
        base=r"C:\Users\test\AppData\Local\Programs\Python\Python312\python.exe"
        proc=fake_process(path=base,args=[EXPECTED,"-m","neral_local_chat_api"])
        result=classify_neral_candidate(proc,EXPECTED,now=10000)
        self.assertFalse(result["pinned_executable"])
        self.assertTrue(result["pinned_launch_argv0"])
        self.assertTrue(result["compatible_python_binary"])
        self.assertTrue(result["verified"])
        malicious=fake_process(path=r"C:\Windows\System32\cmd.exe",
                               args=[EXPECTED,"-m","neral_local_chat_api"])
        self.assertFalse(classify_neral_candidate(
            malicious,EXPECTED,now=10000)["verified"])
        unrelated=fake_process(path=base,args=[base,"-m","neral_local_chat_api"])
        self.assertFalse(classify_neral_candidate(
            unrelated,EXPECTED,now=10000)["verified"])

    def test_recent_busy_or_training_python_not_eligible(self):
        self.assertFalse(classify_neral_candidate(fake_process(age=10),EXPECTED,now=10000)["verified"])
        self.assertFalse(classify_neral_candidate(fake_process(cpu=20),EXPECTED,now=10000)["verified"])
        train=fake_process(args=[EXPECTED,"-m","scripts.neral_v8_selfverify_train","--mode","train-local-gpu"])
        self.assertFalse(classify_neral_candidate(train,EXPECTED,now=10000)["verified"])
        runner=fake_process(args=[EXPECTED,"D:\\actions-runner\\_work\\neral-helios\\benchmark.py"])
        self.assertFalse(classify_neral_candidate(runner,EXPECTED,now=10000)["verified"])

    def test_only_verified_process_can_be_terminated_and_no_force_kill(self):
        safe=fake_process()
        result=stop_verified(safe,EXPECTED,now=10000)
        self.assertTrue(result)
        safe.terminate.assert_called_once()
        safe.wait.assert_called_once()
        safe.kill.assert_not_called()
        unsafe=fake_process(path=r"C:\AnotherPython\python.exe")
        self.assertFalse(stop_verified(unsafe,EXPECTED,now=10000))
        unsafe.terminate.assert_not_called()
        unsafe.kill.assert_not_called()

    def test_recheck_criteria_before_termination(self):
        flaky=fake_process()
        flaky.cpu_percent.side_effect=[0,11]
        self.assertFalse(stop_verified(flaky,EXPECTED,now=10000))
        flaky.terminate.assert_not_called()

if __name__=="__main__":
    unittest.main()
