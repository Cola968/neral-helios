"""V2.1 real-file-to-Docker contracts; never execute model code on the host."""
import io
import json
import subprocess
import threading
import unittest
import zipfile
from unittest.mock import patch

from flask import Flask

from neral_tir_experimental import _run_ephemeral_python


class DatasetSandboxContracts(unittest.TestCase):
    def test_readonly_staged_csv_available_only_during_docker_run(self):
        observed=[]
        def subprocess_fake(cmd, **kwargs):
            if cmd[:3]==["docker","image","inspect"]:
                return subprocess.CompletedProcess(cmd,0,b"ok",b"")
            if cmd[:2]==["docker","run"]:
                root=next(p.split("src=",1)[1].split(",dst=",1)[0]
                          for p in cmd if p.startswith("type=bind,src="))
                from pathlib import Path
                file=Path(root)/"data"/"data1.csv"
                self.assertEqual(file.read_bytes(),b"name,value\nA,2\nB,4\n")
                self.assertIn("readonly"," ".join(cmd))
                observed.append(cmd)
                return subprocess.CompletedProcess(cmd,0,b"6\n",b"")
            raise AssertionError("Unknown docker invocation")
        with patch("neral_tir_experimental.subprocess.run",side_effect=subprocess_fake):
            result=_run_ephemeral_python(
                "import csv\nprint(sum(int(r['value']) for r in csv.DictReader(open('data/data1.csv'))))",
                staged_files={"data/data1.csv":b"name,value\nA,2\nB,4\n"})
        self.assertEqual(result["status"],"PASSED")
        self.assertEqual(result["output"],"6")
        self.assertEqual(len(observed),1)

    def test_reject_untrusted_staging_names_and_oversize(self):
        for name in ("../secrets.txt","data/../evil.csv","data/injected.py",
                     "/etc/passwd","data/a\\b.csv","data/sub/f.csv"):
            with self.subTest(name=name):
                result=_run_ephemeral_python("print(1)",staged_files={name:b"x"})
                self.assertEqual(result["status"],"FAILED")
        result=_run_ephemeral_python(
            "print(1)",staged_files={"data/data1.csv":b"x"*(4*1024*1024+1)})
        self.assertEqual(result["status"],"FAILED")

    def test_spreadsheet_xml_to_csv_without_excel_execution(self):
        from neral_v21_attachment_data import prepare_attachment_dataset
        with io.BytesIO() as buffer:
            with zipfile.ZipFile(buffer,"w") as z:
                z.writestr("xl/worksheets/sheet1.xml",
                    '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
                    '<sheetData><row r="1"><c r="A1" t="inlineStr"><is><t>value</t></is></c></row>'
                    '<row r="2"><c r="A2"><v>3</v></c></row>'
                    '<row r="3"><c r="A3"><v>5</v></c></row></sheetData></worksheet>')
            prepared=prepare_attachment_dataset([("auswertung.xlsx",".xlsx",buffer.getvalue())])
        self.assertEqual(prepared["staged_files"]["data/data1.csv"],b"value\r\n3\r\n5\r\n")
        self.assertIn("data/data1.csv",prepared["manifest"])
        self.assertNotIn("C:\\",prepared["manifest"])


class MultipartContracts(unittest.TestCase):
    def test_csv_analysis_uses_helios_and_local_docker_instead_of_legacy(self):
        from neral_preview_attachments import register_preview_attachment_api
        from neral_three_tier_runtime import ThreeTierRuntime

        class Model:
            def __init__(self,model,answers=()):
                self.model=model
                self.adapter_sha256=("a"*64 if "9b" in model else None)
                self.answers=list(answers);self.loaded=False;self.calls=0
            def respond_chat(self,turns,*,max_new_tokens,temperature=None,max_time_seconds=None):
                self.loaded=True;self.calls+=1
                return self.answers.pop(0)
            def unload(self):self.loaded=False
        luna=Model("qwen3.5:4b",["Should not run"])
        helios=Model("qwen3.5:9b+neral-v2-full",[
            "\x60\x60\x60python\nimport csv\nprint(sum(int(r['value']) for r in csv.DictReader(open('data/data1.csv'))))\n\x60\x60\x60",
            "Die Summe ist 6.",
        ])
        staged=[]
        def execute_data(code, files):
            staged.append(files)
            self.assertIn("data/data1.csv",code)
            return {"status":"PASSED","output":"6"}
        runtime=ThreeTierRuntime(luna,helios,execute_python=lambda code:
                                 {"status":"FAILED"},execute_dataset_python=execute_data)
        app=Flask("v21-files")
        class Legacy:
            model="qwen3:8b"
            def respond(self, *a, **kw):raise AssertionError("Legacy cannot process file analysis")
        register_preview_attachment_api(app,Legacy(),threading.Lock(),
                                       three_tier_runtime=runtime)
        client=app.test_client()
        result=client.post("/api/neral/preview/attachment-chat",data={
            "mode":"generate","tier":"Luna","force_sandbox":"true",
            "prompt":"Summiere die Spalte value aus der CSV-Datei.",
            "files":(io.BytesIO(b"name,value\nA,2\nB,4\n"),"werte.csv"),
        },content_type="multipart/form-data")
        self.assertEqual(result.status_code,200,result.get_json())
        self.assertEqual(result.json["tier"],"Helios")
        self.assertEqual(result.json["tool_used"],"helios_code_sandbox")
        self.assertTrue(result.json["sandbox_executed"])
        self.assertEqual(staged[0]["data/data1.csv"],b"name,value\nA,2\nB,4\n")
        self.assertFalse(result.json["file_bytes_persisted"])
        self.assertFalse(result.json["verified"])
        self.assertEqual(luna.calls,0)


if __name__=="__main__":
    unittest.main()
