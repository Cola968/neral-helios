from __future__ import annotations
import ast,json,subprocess,sys
ALLOWED_MODULES={"math","re","collections","itertools","functools","heapq","bisect","operator","string"}
BANNED_CALLS={"open","exec","eval","compile","input","globals","locals","vars","dir","getattr","setattr","delattr","breakpoint","help"}
def validate(code):
 tree=ast.parse(code)
 for n in ast.walk(tree):
  if isinstance(n,(ast.Global,ast.Nonlocal)):raise ValueError("global/nonlocal forbidden")
  if isinstance(n,ast.Name) and n.id.startswith("__"):raise ValueError("dunder name forbidden")
  if isinstance(n,ast.Attribute) and n.attr.startswith("__"):raise ValueError("dunder attribute forbidden")
  if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in BANNED_CALLS:raise ValueError("dangerous builtin forbidden")
  if isinstance(n,ast.Import) and any(x.name.split(".")[0] not in ALLOWED_MODULES for x in n.names):raise ValueError("import forbidden")
  if isinstance(n,ast.ImportFrom) and ((n.module or "").split(".")[0] not in ALLOWED_MODULES or n.level):raise ValueError("import-from forbidden")
 return True
RUNNER=r"""
import json,sys,resource
p=json.loads(sys.stdin.read())
resource.setrlimit(resource.RLIMIT_CPU,(2,2))
resource.setrlimit(resource.RLIMIT_AS,(384*1024*1024,384*1024*1024))
resource.setrlimit(resource.RLIMIT_FSIZE,(0,0))
allowed=set(p["modules"]);real_import=__import__
def safe_import(name,globals=None,locals=None,fromlist=(),level=0):
 if level or name.split(".")[0] not in allowed:raise ImportError("module blocked")
 return real_import(name,globals,locals,fromlist,level)
safe={"__import__":safe_import,"range":range,"len":len,"sum":sum,"min":min,"max":max,"abs":abs,"all":all,"any":any,"enumerate":enumerate,"zip":zip,"map":map,"filter":filter,"sorted":sorted,"reversed":reversed,"list":list,"tuple":tuple,"set":set,"dict":dict,"str":str,"int":int,"float":float,"bool":bool,"pow":pow,"round":round,"ord":ord,"chr":chr,"isinstance":isinstance,"Exception":Exception,"ValueError":ValueError,"TypeError":TypeError,"IndexError":IndexError,"KeyError":KeyError}
g={"__builtins__":safe}
try:
 exec(p["source"],g,g);print(json.dumps({"pass":True}))
except BaseException as e:print(json.dumps({"pass":False,"error":type(e).__name__+":"+str(e)[:200]}))
"""
def run(code,test_imports,test_list,timeout=4):
 try:validate(code)
 except Exception as e:return {"pass":False,"error":"validation:"+str(e)}
 source=code+"\n"+"\n".join(test_imports)+"\n"+"\n".join(test_list)+"\n"
 try:cp=subprocess.run([sys.executable,"-I","-c",RUNNER],input=json.dumps({"source":source,"modules":sorted(ALLOWED_MODULES)}),text=True,capture_output=True,timeout=timeout,env={})
 except subprocess.TimeoutExpired:return {"pass":False,"error":"timeout"}
 if cp.returncode!=0:return {"pass":False,"error":"runner_exit:"+str(cp.returncode)}
 try:return json.loads(cp.stdout.strip().splitlines()[-1])
 except Exception:return {"pass":False,"error":"invalid_runner_output"}
