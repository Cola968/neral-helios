"""Windows desktop launcher. Starts only this local server, no auto-installation."""
import subprocess,sys,time,urllib.request,json,webbrowser
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
        return data.get('name')=='Neral / Helios' and data.get('parameters') in {20587008,40051200,50012976,349825152}
    except Exception:return False
def main():
    if status():webbrowser.open(URL);return
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
