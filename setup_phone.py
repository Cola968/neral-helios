"""Create local-only Android TLS credentials; never commit local_state."""
import argparse, datetime, hashlib, ipaddress, json, secrets, socket
from pathlib import Path
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

def setup(host,root):
    address=ipaddress.ip_address(host)
    if not address.is_private or address.is_loopback:raise ValueError('Use the PC private LAN address')
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=True)
    if (root/'config.json').exists():raise ValueError('Phone credentials already exist; reuse the existing setup')
    now=datetime.datetime.now(datetime.timezone.utc)
    ca_key=rsa.generate_private_key(public_exponent=65537,key_size=2048)
    name=x509.Name([x509.NameAttribute(NameOID.COMMON_NAME,'Neral Local Device CA')])
    ca=(x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(ca_key.public_key()).serial_number(x509.random_serial_number()).not_valid_before(now-datetime.timedelta(minutes=5)).not_valid_after(now+datetime.timedelta(days=365)).add_extension(x509.BasicConstraints(ca=True,path_length=0),critical=True).add_extension(x509.KeyUsage(digital_signature=True,key_encipherment=False,key_cert_sign=True,crl_sign=True,content_commitment=False,data_encipherment=False,key_agreement=False,encipher_only=False,decipher_only=False),critical=True).sign(ca_key,hashes.SHA256()))
    key=rsa.generate_private_key(public_exponent=65537,key_size=2048)
    cert=(x509.CertificateBuilder().subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME,'Neral on your PC')])).issuer_name(name).public_key(key.public_key()).serial_number(x509.random_serial_number()).not_valid_before(now-datetime.timedelta(minutes=5)).not_valid_after(now+datetime.timedelta(days=180)).add_extension(x509.SubjectAlternativeName([x509.IPAddress(address)]),critical=False).add_extension(x509.BasicConstraints(ca=False,path_length=None),critical=True).add_extension(x509.ExtendedKeyUsage([x509.oid.ExtendedKeyUsageOID.SERVER_AUTH]),critical=False).sign(ca_key,hashes.SHA256()))
    (root/'Neral-Android-CA.crt').write_bytes(ca.public_bytes(serialization.Encoding.PEM))
    (root/'server.pem').write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    (root/'server.key').write_bytes(key.private_bytes(serialization.Encoding.PEM,serialization.PrivateFormat.PKCS8,serialization.NoEncryption()))
    # CA signing key deliberately not retained.
    code=secrets.token_hex(8)
    config={'host':host,'port':9443,'certificate':str(root/'server.pem'),'key':str(root/'server.key'),'pair_hash':hashlib.sha256(code.encode()).hexdigest()}
    (root/'config.json').write_text(json.dumps(config,indent=2))
    (root/'HANDY-ZUGANG.txt').write_text(f'Neral auf Android\n\n1. PC und Handy ins gleiche WLAN.\n2. Neral-Android-CA.crt auf das eigene Handy kopieren und in Android unter Sicherheit > Verschluesselung/Anmeldedaten > Zertifikat installieren > CA-Zertifikat installieren (Bezeichnung je nach Hersteller).\n3. https://{host}:9443/ in Chrome oeffnen.\n4. Kopplungscode: {code}\n5. Nach dem Verbinden: Chrome-Menue > App installieren.\n\nPC muss eingeschaltet bleiben. Nur dieses Zertifikat installieren; keine Browser-Sicherheitspruefung deaktivieren. Zertifikat spaeter in Android wieder entfernen, wenn Neral nicht mehr verwendet wird. Windows-Firewall muss Port 9443 fuer das private Heimnetz zulassen. Keine Router-Portfreigabe.\n',encoding='utf-8')
    return config

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--host');parser.add_argument('--out',default='local_state/phone');args=parser.parse_args()
    host=args.host
    if not host:
        with socket.socket(socket.AF_INET,socket.SOCK_DGRAM) as s:
            s.connect(('192.0.2.1',9));host=s.getsockname()[0]
    config=setup(host,args.out)
    print('Phone setup created. Open '+str(Path(args.out)/'HANDY-ZUGANG.txt'))
