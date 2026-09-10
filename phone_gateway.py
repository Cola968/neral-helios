"""Opt-in authenticated TLS gateway to the CPU-only loopback app."""
import argparse, hashlib, hmac, http.client, json, secrets, ssl, time
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit


def create_server(config, backend_port=9684):
    host=config['host'];port=config['port'];origin=f'https://{host}:{port}'
    sessions={};attempts={}
    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup();self.connection.settimeout(15)
        def log_message(self,*args):pass
        def answer(self,status,body,kind='text/plain; charset=utf-8',cookie=None):
            data=body.encode() if isinstance(body,str) else body
            self.send_response(status);self.send_header('Content-Type',kind)
            self.send_header('Content-Length',str(len(data)));self.send_header('Cache-Control','no-store')
            self.send_header('X-Content-Type-Options','nosniff');self.send_header('X-Frame-Options','DENY')
            self.send_header('Referrer-Policy','no-referrer')
            if cookie:self.send_header('Set-Cookie',cookie)
            self.end_headers();self.wfile.write(data)
        def handle_request(self):
            if self.headers.get('Host')!=f'{host}:{port}':return self.answer(403,'Host rejected')
            if self.headers.get('Origin') not in (None,origin) or self.headers.get('Sec-Fetch-Site')=='cross-site':return self.answer(403,'Origin rejected')
            if self.headers.get('Transfer-Encoding'):return self.answer(400,'Unsupported transfer encoding')
            try:length=int(self.headers.get('Content-Length','0'))
            except ValueError:return self.answer(400,'Invalid length')
            if not 0<=length<=262144:return self.answer(413,'Request too large')
            if not self.path.startswith('/') or self.path.startswith('//') or urlsplit(self.path).netloc:return self.answer(400,'Invalid path')
            body=self.rfile.read(length)
            now=time.time()
            if self.command=='POST' and self.path=='/pair':
                key=self.client_address[0];recent=[t for t in attempts.get(key,[]) if now-t<60]
                if len(attempts)>256:attempts.clear()
                attempts[key]=recent+[now]
                if len(recent)>=5:return self.answer(429,'Bitte eine Minute warten.')
                try:code=parse_qs(body.decode(),max_num_fields=4).get('code',[''])[0]
                except (ValueError,UnicodeError):return self.answer(400,'Invalid form')
                if not hmac.compare_digest(hashlib.sha256(code.encode()).hexdigest(),config['pair_hash']):return self.answer(403,'Kopplungscode stimmt nicht.')
                for token,expiry in list(sessions.items()):
                    if expiry<now:sessions.pop(token,None)
                if len(sessions)>=16:return self.answer(429,'Zu viele gekoppelte Sitzungen.')
                token=secrets.token_urlsafe(32);sessions[token]=now+8*3600
                return self.answer(200,'<!doctype html><html lang="de"><meta name="viewport" content="width=device-width"><title>Neral verbunden</title><h1>Verbunden</h1><a href="/">Neral öffnen</a></html>','text/html; charset=utf-8',f'neral_session={token}; Path=/; Secure; HttpOnly; SameSite=Strict; Max-Age=28800')
            cookie=SimpleCookie()
            try:cookie.load(self.headers.get('Cookie',''));token=cookie['neral_session'].value if 'neral_session' in cookie else ''
            except Exception:token=''
            if self.command=='POST' and self.path=='/logout':
                sessions.pop(token,None)
                return self.answer(200,'Abgemeldet.',cookie='neral_session=; Path=/; Secure; HttpOnly; SameSite=Strict; Max-Age=0')
            if sessions.get(token,0)<now:
                if self.command=='GET' and self.path=='/':
                    return self.answer(200,'<!doctype html><html lang="de"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Neral verbinden</title><h1>Mit deinem PC verbinden</h1><p>Den Kopplungscode findest du in HANDY-ZUGANG.txt auf deinem PC.</p><form action="/pair" method="post"><label>Kopplungscode <input name="code" type="password" autocomplete="off" required></label><button>Verbinden</button></form></html>','text/html; charset=utf-8')
                return self.answer(401,'Bitte zuerst auf der Startseite koppeln.')
            # Never forward client authentication or arbitrary proxy headers.
            connection=http.client.HTTPConnection('127.0.0.1',backend_port,timeout=180)
            headers={'Host':f'127.0.0.1:{backend_port}','Origin':f'http://127.0.0.1:{backend_port}'}
            token_path=Path(config.get('backend_token_file',str(Path(__file__).parent/'local_state/gateway-token')))
            try:headers['X-Helios-Gateway']=token_path.read_text(encoding='ascii').strip()
            except OSError:return self.answer(502,'PC-App bitte zuerst starten.')
            if self.headers.get('Content-Type'):headers['Content-Type']=self.headers['Content-Type']
            try:
                connection.request(self.command,self.path,body=body,headers=headers);response=connection.getresponse()
                if 'application/x-ndjson' in response.getheader('Content-Type',''):
                    self.send_response(response.status);self.send_header('Content-Type','application/x-ndjson');self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff');self.send_header('Connection','close');self.end_headers();self.close_connection=True
                    try:
                        while True:
                            line=response.readline(16385)
                            if not line:break
                            if len(line)>16384:raise ValueError('Oversize stream frame')
                            self.wfile.write(line);self.wfile.flush()
                    except (OSError,ValueError):pass
                    return
                data=response.read()
                self.send_response(response.status)
                for key,value in response.getheaders():
                    if key.lower() in {'content-type','content-security-policy','x-content-type-options','x-frame-options','referrer-policy','permissions-policy','service-worker-allowed'}:self.send_header(key,value)
                self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
            except (OSError,http.client.HTTPException):self.answer(502,'Neral auf dem PC ist nicht erreichbar.')
            finally:connection.close()
        do_GET=handle_request
        do_POST=handle_request
    server=ThreadingHTTPServer((host,port),Handler)
    server.daemon_threads=True
    return server


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',default='local_state/phone/config.json');parser.add_argument('--backend-port',type=int,default=9684)
    args=parser.parse_args();config=json.loads(Path(args.config).read_text())
    server=create_server(config,args.backend_port);context=ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER);context.minimum_version=ssl.TLSVersion.TLSv1_2
    context.load_cert_chain(config['certificate'],config['key']);server.socket=context.wrap_socket(server.socket,server_side=True)
    print(f"Neral Android: https://{config['host']}:{config['port']}",flush=True)
    server.serve_forever()
