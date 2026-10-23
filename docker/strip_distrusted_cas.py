#!/usr/bin/env python3
"""Distrust the compromised CA roots from the eMudhra / emSign G3 advisory.

Removes the same root set that certifi >= 2026.10.19 and Chainguard
ca-certificates >= 20260909-r3 drop (the emSign family and the legacy
"AAA Certificate Services" root) from every TLS trust store the image ships:

  * the system CA bundle at /etc/ssl/certs/ca-certificates.crt (Wolfi /
    Chainguard / Alpine / Debian all converge on this path). On Debian-family
    layouts the roots are also commented out of /etc/ca-certificates.conf and
    deleted from /usr/share/ca-certificates so a later `update-ca-certificates`
    trigger cannot resurrect them; the bundle is then regenerated.
  * every vendored cacert.pem / cacert.txt under the venv — botocore ships its
    own static Mozilla snapshot that still carries AAA Certificate Services,
    and it will come back whenever the dependency is bumped.

Wolfi note: `ca-certificates V:20260909-r3` is NOT trustworthy as a signal.
The package we ship claims that version while its prebuilt bundle still
anchored emSign ECC Root CA - G3 (DER sha256 86a1ecba...) — the package was
bumped without the bundle being regenerated. Content is the only trustworthy
check, which is what this script enforces.

Run during the image build BEFORE any TLS downloads (and again after the venv
COPY so vendored bundles are covered). Idempotent, and SELF-VERIFYING: it
exits non-zero if any distrusted root survives anywhere, so a base-image or
dependency change that re-introduces one fails the build loudly.

Ported from grid-ai-onboarding c697a5c (log-viewer/docker/strip_distrusted_cas.py)
to handle Wolfi paths plus venv-vendored bundles.
"""

import glob
import os
import re
import ssl
import subprocess
import sys
import tempfile

# CNs to distrust: the advisory's critical set (emSign ECC Root CA - G3 and the
# AAA Certificate Services cross-sign of the same eight keys) plus the rest of
# the eMudhra emSign family, matching what patched certifi / ca-certificates
# already drop. G1 / C1 / C3 are not parents of the compromised intermediates,
# but they are not needed by this app either and their presence in the bundle
# is exactly how the stale-bundle problem hides.
DISTRUSTED_CN = re.compile(
    r"^(AAA Certificate Services|emSign .*)$", re.IGNORECASE
)

PEM_BLOCK = re.compile(
    b"-----BEGIN CERTIFICATE-----.*?-----END CERTIFICATE-----", re.DOTALL
)

# Debian / Wolfi layout.
CONF_PATH = "/etc/ca-certificates.conf"
BUNDLE_PATH = "/etc/ssl/certs/ca-certificates.crt"
ALPINE_BUNDLE_PATH = "/etc/ssl/cert.pem"
SOURCE_GLOBS = (
    "/usr/share/ca-certificates/**/*.crt",
    "/usr/local/share/ca-certificates/**/*.crt",
)
# Vendored bundles: botocore is the known dirty one (AAA). Scan the whole venv
# so the next dependency that ships its own snapshot is caught too.
VENDORED_GLOBS = (
    "/app/.venv/**/cacert.pem",
    "/app/.venv/**/cacert.txt",
    "/app/.venv/**/cacerts.pem",
    "/usr/lib/google-cloud-sdk/**/cacerts.txt",
)


def cert_common_name(pem_block: bytes) -> str:
    """Extract the Subject CN from one PEM block (stdlib only)."""
    with tempfile.NamedTemporaryFile(suffix=".pem", delete=False) as tmp:
        tmp.write(pem_block + b"\n")
        tmp_path = tmp.name
    try:
        decoded = ssl._ssl._test_decode_cert(tmp_path)  # noqa: SLF001
        subject = dict(item[0] for item in decoded.get("subject", ()))
        return subject.get("commonName", "")
    finally:
        os.unlink(tmp_path)


def is_distrusted(pem_block: bytes) -> bool:
    try:
        return bool(DISTRUSTED_CN.match(cert_common_name(pem_block)))
    except Exception:
        return False  # undecodable: never drop what we cannot identify


def filter_pem_file(path: str) -> int:
    """Rewrite a multi-PEM file without the distrusted certs. Returns count."""
    try:
        with open(path, "rb") as fh:
            data = fh.read()
    except (PermissionError, FileNotFoundError):
        return 0
    blocks = PEM_BLOCK.findall(data)
    if not blocks:
        return 0
    kept = [block for block in blocks if not is_distrusted(block)]
    removed = len(blocks) - len(kept)
    if removed:
        # Preserve whatever non-PEM preamble/footer the file had (certifi has
        # a comment header; the system bundle is pure PEM).
        out = data
        for block in blocks:
            if is_distrusted(block):
                out = out.replace(block + b"\n", b"", 1)
                out = out.replace(block, b"", 1)
        out = re.sub(rb"\n{3,}", b"\n\n", out)
        try:
            with open(path, "wb") as fh:
                fh.write(out if out.endswith(b"\n") else out + b"\n")
        except PermissionError:
            print(f"note: cannot rewrite {path}; verifying only")
            return 0
    return removed


def strip_system_store() -> None:
    # 1. Disable the roots in ca-certificates.conf (Debian / Wolfi switch).
    if os.path.isfile(CONF_PATH):
        try:
            with open(CONF_PATH) as fh:
                lines = fh.readlines()
            with open(CONF_PATH, "w") as fh:
                for line in lines:
                    stripped = line.strip()
                    if (
                        stripped
                        and not stripped.startswith("#")
                        and ("emSign" in stripped or "AAA_Certificate" in stripped)
                    ):
                        fh.write("# " + line)
                    else:
                        fh.write(line)
        except PermissionError:
            print(f"note: {CONF_PATH} not writable here; verifying only")

    # 2. Delete the source certs so a later `update-ca-certificates` trigger
    #    cannot resurrect them. (Wolfi ships these under
    #    /usr/share/ca-certificates/mozilla/; the 20260909-r3 package already
    #    dropped G3/AAA here — this is belt and braces.)
    for pattern in SOURCE_GLOBS:
        for path in glob.glob(pattern, recursive=True):
            base = os.path.basename(path)
            if "emSign" in base or "AAA_Certificate" in base:
                try:
                    os.unlink(path)
                except PermissionError:
                    print(f"note: cannot delete {path}; verifying only")

    # 3. Regenerate the bundle if the tooling works. On a healthy install this
    #    yields a clean bundle on its own; the direct filter below is still the
    #    source of truth, because on this image the "patched" package shipped a
    #    stale bundle and the version number lied. Missing tooling is fine —
    #    the filter is what must actually run.
    if os.path.isfile(BUNDLE_PATH) or os.path.isfile(ALPINE_BUNDLE_PATH):
        for args in (["update-ca-certificates", "--fresh"], ["update-ca-certificates"]):
            try:
                subprocess.run(args, check=False, capture_output=True)
            except (FileNotFoundError, PermissionError, OSError):
                print(f"note: {' '.join(args)} unavailable; relying on direct filter")


def all_trust_files() -> list:
    files = [BUNDLE_PATH, ALPINE_BUNDLE_PATH]
    for pattern in SOURCE_GLOBS:
        files.extend(glob.glob(pattern, recursive=True))
    for pattern in VENDORED_GLOBS:
        files.extend(glob.glob(pattern, recursive=True))
    return sorted({f for f in files if os.path.isfile(f)})


def verify() -> list:
    leftovers = []
    for path in all_trust_files():
        try:
            with open(path, "rb") as fh:
                data = fh.read()
        except (PermissionError, FileNotFoundError):
            continue
        for block in PEM_BLOCK.findall(data):
            try:
                cn = cert_common_name(block)
            except Exception:
                continue
            if DISTRUSTED_CN.match(cn):
                leftovers.append(f"{path}: {cn}")
    return leftovers


def main() -> int:
    strip_system_store()

    removed = 0
    for path in all_trust_files():
        removed += filter_pem_file(path)

    leftovers = verify()
    if leftovers:
        print("ERROR: distrusted CA roots still present:")
        for item in leftovers:
            print("  -", item)
        return 1

    scanned = all_trust_files()
    print(
        f"OK: distrusted CA roots removed "
        f"(filtered {removed} cert(s) across {len(scanned)} trust file(s))"
    )
    for path in scanned:
        print(f"  - {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
