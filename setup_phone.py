"""Create local-only Android TLS credentials; never commit local_state."""
import argparse, datetime, hashlib, ipaddress, json, os, secrets, socket
from pathlib import Path
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

def _write_private(path, data):
    """Create credentials exclusively, without following or overwriting files."""
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(data if isinstance(data, bytes) else data.encode("utf-8"))
    if os.name != "nt":
        os.chmod(path, 0o600)


def setup(host,root):
    address=ipaddress.ip_address(host)
    if (address.version != 4 or not address.is_private or address.is_loopback
            or address.is_unspecified or address.is_link_local):
        raise ValueError('Use a concrete private IPv4 LAN address')
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=True,mode=0o700)
    if os.name != "nt":root.chmod(0o700)
    if (root/'config.json').exists():raise ValueError('Phone credentials already exist; reuse the existing setup')
    now=datetime.datetime.now(datetime.timezone.utc)
    ca_key=rsa.generate_private_key(public_exponent=65537,key_size=2048)
    name=x509.Name([x509.NameAttribute(NameOID.COMMON_NAME,'Neral Local Device CA')])
    ca=(x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(ca_key.public_key()).serial_number(x509.random_serial_number()).not_valid_before(now-datetime.timedelta(minutes=5)).not_valid_after(now+datetime.timedelta(days=365)).add_extension(x509.BasicConstraints(ca=True,path_length=0),critical=True).add_extension(x509.KeyUsage(digital_signature=True,key_encipherment=False,key_cert_sign=True,crl_sign=True,content_commitment=False,data_encipherment=False,key_agreement=False,encipher_only=False,decipher_only=False),critical=True).sign(ca_key,hashes.SHA256()))
    key=rsa.generate_private_key(public_exponent=65537,key_size=2048)
    cert=(x509.CertificateBuilder().subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME,'Neral on your PC')])).issuer_name(name).public_key(key.public_key()).serial_number(x509.random_serial_number()).not_valid_before(now-datetime.timedelta(minutes=5)).not_valid_after(now+datetime.timedelta(days=180)).add_extension(x509.SubjectAlternativeName([x509.IPAddress(address)]),critical=False).add_extension(x509.BasicConstraints(ca=False,path_length=None),critical=True).add_extension(x509.ExtendedKeyUsage([x509.oid.ExtendedKeyUsageOID.SERVER_AUTH]),critical=False).sign(ca_key,hashes.SHA256()))
    _write_private(root/'Neral-Android-CA.crt',ca.public_bytes(serialization.Encoding.PEM))
    _write_private(root/'server.pem',cert.public_bytes(serialization.Encoding.PEM))
    _write_private(root/'server.key',key.private_bytes(serialization.Encoding.PEM,serialization.PrivateFormat.PKCS8,serialization.NoEncryption()))
    # CA signing key deliberately not retained.
    code=secrets.token_hex(8)
    config={'host':host,'port':9443,'certificate':str(root/'server.pem'),'key':str(root/'server.key'),'pair_hash':hashlib.sha256(code.encode()).hexdigest()}
    _write_private(root/'config.json',json.dumps(config,indent=2))
    _write_private(root/'HANDY-ZUGANG.txt',f'Neral auf Android\n\n1. PC und Handy ins gleiche WLAN.\n2. Neral-Android-CA.crt auf das eigene Handy kopieren und in Android unter Sicherheit > Verschluesselung/Anmeldedaten > Zertifikat installieren > CA-Zertifikat installieren (Bezeichnung je nach Hersteller).\n3. https://{host}:9443/ in Chrome oeffnen.\n4. Kopplungscode: {code}\n5. Nach dem Verbinden: Chrome-Menue > App installieren.\n\nPC muss eingeschaltet bleiben. Nur dieses Zertifikat installieren; keine Browser-Sicherheitspruefung deaktivieren. Zertifikat spaeter in Android wieder entfernen, wenn Neral nicht mehr verwendet wird. Windows-Firewall muss Port 9443 fuer das private Heimnetz zulassen. Keine Router-Portfreigabe.\n')
    return config

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--host');parser.add_argument('--out',default='local_state/phone');args=parser.parse_args()
    host=args.host
    if not host:
        with socket.socket(socket.AF_INET,socket.SOCK_DGRAM) as s:
            s.connect(('192.0.2.1',9));host=s.getsockname()[0]
    config=setup(host,args.out)
    print('Phone setup created. Open '+str(Path(args.out)/'HANDY-ZUGANG.txt'))
