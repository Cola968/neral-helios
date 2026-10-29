"""Windows desktop launcher. Starts only this local server, no auto-installation."""
import os,socket,subprocess,sys,time,urllib.request,json,webbrowser
from pathlib import Path
ROOT=Path(__file__).resolve().parent
URL='http://127.0.0.1:9680'
def status():
    try:
        # The real API requires the local HttpOnly session cookie issued by /.
        # Keep it inside this short-lived opener; never read gateway-token files
        # or weaken authentication just to poll readiness from the batch starter.
        opener=urllib.request.build_opener(urllib.request.HTTPCookieProcessor())
        with opener.open(URL+'/',timeout=1):
            pass
        with opener.open(URL+'/api/info',timeout=1) as response:
            data=json.load(response)
        if data.get('name')!='Neral / Helios' or data.get('parameters') not in {20587008,40051200,50012976,349825152}:return False
        if os.environ.get('NERAL_ASSISTANT_ENABLED')=='1':
            assistant=data.get('assistant',{})
            if data.get('native_arctus350_integrated') is not True or not isinstance(assistant,dict) or assistant.get('enabled') is not True or assistant.get('base_model')!=os.environ.get('NERAL_ASSISTANT_MODEL','qwen3.5:9b'):return False
        if os.environ.get('HELIOS_ENABLE_ARCTUS350')=='1' and not data.get('arctus350_enabled'):return False
        desired=os.environ.get('HELIOS_CHECKPOINT')
        if desired and data.get('active_checkpoint')!=desired:return False
        return True
    except Exception:return False
def port_occupied():
    try:
        with socket.create_connection(('127.0.0.1',9680),timeout=1):return True
    except OSError:return False
def main():
    if status():webbrowser.open(URL);return
    if port_occupied():raise RuntimeError('Port 9680 ist belegt oder die laufende NERAL-App entspricht nicht der gewählten Konfiguration. Alte App beenden und erneut starten.')
    logs=ROOT/'logs';logs.mkdir(exist_ok=True)
    with (logs/'server.log').open('ab') as log:
        flags=subprocess.CREATE_NO_WINDOW if sys.platform=='win32' else 0
        process=subprocess.Popen([sys.executable,'-u','app.py'],cwd=ROOT,stdout=log,stderr=log,creationflags=flags)
    for _ in range(90):
        if process.poll() is not None:raise RuntimeError('Serverstart fehlgeschlagen. Details: logs/server.log')
        if status():webbrowser.open(URL);print('Neral läuft lokal. Protokoll: logs/server.log');return
        time.sleep(1)
    process.terminate();raise RuntimeError('Serverstart dauert zu lange. Details: logs/server.log')
if __name__=='__main__':main()

