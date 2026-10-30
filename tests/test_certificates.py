"""
Tests for certificates.py's project CA and server certificate generation

- Every test writes into pytest's tmp_path, never the real certs/ folder
- Chain checks use cryptography's own verifier, plus Python's ssl module in strict mode (what the cloud agent uses)
"""

# ---------------------------------
# Imports
# ---------------------------------
import ipaddress
import ssl
import sys
from cryptography import x509
from cryptography.x509.verification import PolicyBuilder, Store
import capstone.certificates as certificates




# ---------------------------------
# Test helpers
# ---------------------------------

def loadCert(path) -> x509.Certificate:
    """
    Read a PEM certificate from disk
    """

    return x509.load_pem_x509_certificate(path.read_bytes())


def verifyServerCert(certsDir, subject) -> None:
    """
    Verify edge.crt against ca.crt for the given subject, raising if the chain or name doesn't check out
    """

    store = Store([loadCert(certsDir / "ca.crt")])
    verifier = PolicyBuilder().store(store).build_server_verifier(subject)

    verifier.verify(loadCert(certsDir / "edge.crt"), [])




# ---------------------------------
# makeCerts tests
# ---------------------------------

def test_make_certs_writes_all_four_files(tmp_path):
    """
    A first run should write the CA and server certificate, each with its key
    """

    createdCa = certificates.makeCerts(tmp_path / "certs", "192.168.10.2", "jetson-edge")

    assert (createdCa is True)
    assert (sorted(path.name for path in (tmp_path / "certs").iterdir()) == ["ca.crt", "ca.key", "edge.crt", "edge.key"])


def test_server_cert_verifies_for_the_jetson_ip_and_localhost(tmp_path):
    """
    The server certificate should chain to the CA for the Jetson's IP, 127.0.0.1, the hostname, and localhost
    """

    certificates.makeCerts(tmp_path, "192.168.10.2", "jetson-edge")

    verifyServerCert(tmp_path, x509.IPAddress(ipaddress.ip_address("192.168.10.2")))
    verifyServerCert(tmp_path, x509.IPAddress(ipaddress.ip_address("127.0.0.1")))
    verifyServerCert(tmp_path, x509.DNSName("jetson-edge"))
    verifyServerCert(tmp_path, x509.DNSName("localhost"))


def test_server_cert_rejects_a_different_ip(tmp_path):
    """
    The certificate shouldn't be valid for an address it wasn't issued for
    """

    certificates.makeCerts(tmp_path, "192.168.10.2", "jetson-edge")

    try:
        verifyServerCert(tmp_path, x509.IPAddress(ipaddress.ip_address("192.168.10.99")))
    except Exception:
        return

    raise AssertionError("certificate verified for an IP it wasn't issued for")


def test_rerun_reuses_the_existing_ca(tmp_path):
    """
    Re-running should keep the CA (so the laptop's trust stays valid) but re-issue the server certificate
    """

    certificates.makeCerts(tmp_path, "192.168.10.2", "jetson-edge")
    firstCa = (tmp_path / "ca.crt").read_bytes()
    firstServer = (tmp_path / "edge.crt").read_bytes()

    createdCa = certificates.makeCerts(tmp_path, "192.168.10.3", "jetson-edge")

    assert (createdCa is False)
    assert ((tmp_path / "ca.crt").read_bytes() == firstCa)
    assert ((tmp_path / "edge.crt").read_bytes() != firstServer)
    verifyServerCert(tmp_path, x509.IPAddress(ipaddress.ip_address("192.168.10.3")))


def test_ca_loads_into_a_strict_python_ssl_context(tmp_path):
    """
    The CA should load into the same kind of ssl context the cloud agent builds, with strict X.509 checks on
    """

    certificates.makeCerts(tmp_path, "192.168.10.2", "jetson-edge")

    context = ssl.create_default_context(cafile=str(tmp_path / "ca.crt"))

    if (sys.version_info >= (3, 13)):
        assert (context.verify_flags & ssl.VERIFY_X509_STRICT)

    assert (len(context.get_ca_certs()) == 1)




# ---------------------------------
# main tests
# ---------------------------------

def test_main_writes_to_the_requested_folder_and_prints_next_steps(tmp_path, monkeypatch, capsys):
    """
    The CLI should honor --out/--ip/--hostname and tell the operator to copy the server files to the Jetson
    """

    outDir = tmp_path / "out"
    monkeypatch.setattr("sys.argv", ["make-certs", "--out", str(outDir), "--ip", "10.0.0.7", "--hostname", "edge-test"])

    certificates.main()

    output = capsys.readouterr().out

    assert ((outDir / "edge.crt").is_file())
    assert ("Created a new project CA" in output)
    assert ("10.0.0.7" in output and "scp" in output)
    verifyServerCert(outDir, x509.DNSName("edge-test"))
