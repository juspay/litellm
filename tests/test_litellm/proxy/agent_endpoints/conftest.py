from __future__ import annotations

import socket
from collections.abc import Iterator
from ipaddress import ip_address

import pytest

from litellm.litellm_core_utils import url_utils


@pytest.fixture(autouse=True)
def public_dns(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Resolve every non-literal test hostname to a public address.

    Agent cards and A2A routing now SSRF-validate their upstream URLs with
    ``validate_url``, which resolves DNS. Test hostnames like ``localhost`` and
    ``upstream`` do not exist (or resolve to loopback), so make them resolve to
    a public address. IP literals keep their real (in)validation so tests that
    pin a private address still exercise the blocklist.
    """

    real_getaddrinfo = url_utils.socket.getaddrinfo

    def _fake_getaddrinfo(host, port, *args, **kwargs):
        try:
            ip_address(host)
        except ValueError:
            return [
                (
                    socket.AF_INET,
                    socket.SOCK_STREAM,
                    socket.IPPROTO_TCP,
                    "",
                    ("93.184.216.34", port),
                )
            ]
        return real_getaddrinfo(host, port, *args, **kwargs)

    monkeypatch.setattr(url_utils.socket, "getaddrinfo", _fake_getaddrinfo)
    yield
