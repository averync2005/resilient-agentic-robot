"""
TLS certificates for the HTTPS link between the cloud agent (laptop) and the edge agent (Jetson)

Why a project CA:
- The Jetson sits on a direct Ethernet link with no domain name, so no public CA can issue it a certificate
- Instead this script creates a small, private certificate authority (CA) just for this project
- The CA signs the Jetson's server certificate, and the cloud agent trusts only that CA
- Anyone without the CA's private key can't impersonate the Jetson

Files written (to certs/, which is git-ignored):
- ca.crt     The CA certificate - the cloud agent verifies the edge agent against it (CLOUD_CA_CERT)
- ca.key     The CA private key - never leaves the laptop
- edge.crt   The Jetson's server certificate, valid for its IP plus localhost (EDGE_TLS_CERT)
- edge.key   The server certificate's private key (EDGE_TLS_KEY)

Re-running:
- An existing CA is reused, so a laptop that already trusts it keeps working
- The server certificate is always re-issued (ex. after the Jetson's IP changes)

Running it (on the laptop, from the project root):
- Registered as the `make-certs` CLI command in `pyproject.toml`
- Run it with:
    uv run make-certs
    uv run make-certs --ip 192.168.10.2 --hostname jetson-edge
- Then copy edge.crt and edge.key to the Jetson's certs/ folder
"""

# ---------------------------------
# Imports
# ---------------------------------
import argparse
import datetime
import ipaddress
from pathlib import Path
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID




# ---------------------------------
# Validity periods
# ---------------------------------

CA_VALID_DAYS = 3650  # The CA outlives the project, so it never needs replacing
SERVER_VALID_DAYS = 825  # Longest server-certificate lifetime most TLS clients still accept




# ---------------------------------
# Creating the CA
# ---------------------------------

def createCa() -> tuple[x509.Certificate, ec.EllipticCurvePrivateKey]:
    """
    Create a new self-signed project CA

    - Includes the key-usage and key-identifier extensions Python 3.13+ requires when verifying a chain

    Returns:
        The CA certificate and its private key
    """

    caKey = ec.generate_private_key(ec.SECP256R1())
    caName = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Capstone Project CA")])
    now = datetime.datetime.now(datetime.timezone.utc)

    caKeyUsage = x509.KeyUsage (
        digital_signature=False,
        content_commitment=False,
        key_encipherment=False,
        data_encipherment=False,
        key_agreement=False,
        key_cert_sign=True,
        crl_sign=True,
        encipher_only=False,
        decipher_only=False
    )

    caCert = (
        x509.CertificateBuilder()
        .subject_name(caName)
        .issuer_name(caName)
        .public_key(caKey.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + datetime.timedelta(days=CA_VALID_DAYS))
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .add_extension(caKeyUsage, critical=True)
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(caKey.public_key()), critical=False)
        .sign(caKey, hashes.SHA256())
    )

    return caCert, caKey




# ---------------------------------
# Creating the server certificate
# ---------------------------------

def createServerCert(caCert: x509.Certificate, caKey: ec.EllipticCurvePrivateKey, ip: str, hostname: str) -> tuple[x509.Certificate, ec.EllipticCurvePrivateKey]:
    """
    Create the Jetson's server certificate, signed by the project CA

    - Valid for the given IP and hostname, plus localhost/127.0.0.1 so the server can be tested on one machine

    Args:
        caCert: The project CA certificate
        caKey: The project CA private key
        ip: The Jetson's IP on the Ethernet link (ex. 192.168.10.2)
        hostname: A hostname for the Jetson (ex. jetson-edge)

    Returns:
        The server certificate and its private key
    """

    serverKey = ec.generate_private_key(ec.SECP256R1())
    now = datetime.datetime.now(datetime.timezone.utc)

    subjectAltNames = x509.SubjectAlternativeName ([
        x509.IPAddress(ipaddress.ip_address(ip)),
        x509.IPAddress(ipaddress.ip_address("127.0.0.1")),
        x509.DNSName(hostname),
        x509.DNSName("localhost")
    ])

    serverKeyUsage = x509.KeyUsage (
        digital_signature=True,
        content_commitment=False,
        key_encipherment=False,
        data_encipherment=False,
        key_agreement=False,
        key_cert_sign=False,
        crl_sign=False,
        encipher_only=False,
        decipher_only=False
    )

    serverCert = (
        x509.CertificateBuilder()
        .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, hostname)]))
        .issuer_name(caCert.subject)
        .public_key(serverKey.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + datetime.timedelta(days=SERVER_VALID_DAYS))
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(serverKeyUsage, critical=True)
        .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
        .add_extension(subjectAltNames, critical=False)
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(serverKey.public_key()), critical=False)
        .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(caKey.public_key()), critical=False)
        .sign(caKey, hashes.SHA256())
    )

    return serverCert, serverKey




# ---------------------------------
# Reading/writing PEM files
# ---------------------------------

def writeCert(path: Path, cert: x509.Certificate) -> None:
    """
    Write a certificate to a PEM file
    """

    path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))


def writeKey(path: Path, key: ec.EllipticCurvePrivateKey) -> None:
    """
    Write an unencrypted private key to a PEM file (unencrypted so the server can start unattended)
    """

    path.write_bytes (
        key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
    )


def loadCa(certsDir: Path) -> tuple[x509.Certificate, ec.EllipticCurvePrivateKey]:
    """
    Load an existing project CA from certsDir

    Args:
        certsDir: Folder holding ca.crt and ca.key

    Returns:
        The CA certificate and its private key
    """

    caCert = x509.load_pem_x509_certificate((certsDir / "ca.crt").read_bytes())
    caKey = serialization.load_pem_private_key((certsDir / "ca.key").read_bytes(), password=None)

    return caCert, caKey




# ---------------------------------
# Generating every file
# ---------------------------------

def makeCerts(certsDir: Path, ip: str, hostname: str) -> bool:
    """
    Write the project CA (if it doesn't exist yet) and a fresh server certificate to certsDir

    Args:
        certsDir: Output folder, created if missing
        ip: The Jetson's IP on the Ethernet link
        hostname: A hostname for the Jetson

    Returns:
        True if a new CA was created, False if an existing one was reused
    """

    certsDir.mkdir(parents=True, exist_ok=True)
    caExists = (certsDir / "ca.crt").is_file() and (certsDir / "ca.key").is_file()

    if (caExists):
        caCert, caKey = loadCa(certsDir)
    else:
        caCert, caKey = createCa()
        writeCert(certsDir / "ca.crt", caCert)
        writeKey(certsDir / "ca.key", caKey)

    serverCert, serverKey = createServerCert(caCert, caKey, ip, hostname)
    writeCert(certsDir / "edge.crt", serverCert)
    writeKey(certsDir / "edge.key", serverKey)

    return (not caExists)


def main() -> None:
    """
    Parse command-line options, generate the certificates, and print the next steps
    """

    parser = argparse.ArgumentParser(description="Generate the project CA and the Jetson's server certificate")
    parser.add_argument("--ip", default="192.168.10.2", help="The Jetson's IP on the Ethernet link (default: 192.168.10.2)")
    parser.add_argument("--hostname", default="jetson-edge", help="A hostname for the Jetson (default: jetson-edge)")
    parser.add_argument("--out", default="certs", help="Output folder (default: certs)")
    args = parser.parse_args()

    certsDir = Path(args.out)
    createdCa = makeCerts(certsDir, args.ip, args.hostname)

    print(f"{'Created a new' if (createdCa) else 'Reused the existing'} project CA in {certsDir}/")
    print(f"Issued edge.crt for {args.ip} / {args.hostname} (plus localhost)")
    print()
    print("Next, copy the server certificate and key to the Jetson:")
    print(f"    scp {certsDir}/edge.crt {certsDir}/edge.key <user>@{args.ip}:<repo>/certs/")


if (__name__ == "__main__"):
    main()
