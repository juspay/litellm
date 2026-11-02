"""Segment-boundary path matching for the gateway / backend component allowlists.

A prefix entry matches a route path only on whole path segments:
- an entry ending in ``/`` is a directory prefix (``/get/`` matches
  ``/get/config/callbacks`` but not ``/gets``);
- any other entry matches the exact path or that path plus a further segment
  (``/login`` matches ``/login`` and ``/login/renew`` but not ``/loginfoo``).

This keeps a data-plane or passthrough route whose path merely shares leading
characters with an allowlist entry from being trimmed into the wrong
component's route table.
"""


def path_matches_prefix(path: str, prefix: str) -> bool:
    if prefix.endswith("/"):
        return path.startswith(prefix)
    return path == prefix or path.startswith(prefix + "/")


def path_matches_any(path: str, prefixes: tuple[str, ...]) -> bool:
    return any(path_matches_prefix(path, prefix) for prefix in prefixes)
