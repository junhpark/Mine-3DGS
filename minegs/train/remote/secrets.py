"""What must never be written down (Phase 6 §3).

The values of these variables do not enter a record, a status file, a log line, a command, a
provenance block or an exception message. Their *names* may: ``credential_source:
"RUNPOD_API_KEY"`` says where a credential came from without carrying it.
"""

from __future__ import annotations

import os
import re

#: The RunPod API key, as the SDK and CLI name it.
API_KEY_ENV = "RUNPOD_API_KEY"

_SECRET_NAMES = re.compile(
    r"(^RUNPOD_API_KEY$|^RCLONE_CONFIG_PASS$|^RCLONE_.*(SECRET|KEY|PASS|TOKEN).*$"
    r"|^AWS_SECRET_ACCESS_KEY$|^AWS_SESSION_TOKEN$|.*_REGISTRY_(PASSWORD|TOKEN)$"
    r"|^DOCKER_(PASSWORD|AUTH|TOKEN)$|^GH_TOKEN$|^GITHUB_TOKEN$|.*_PRIVATE_KEY$)"
)
#: Shorter than this a "secret" is not one, and redacting it would mangle ordinary text.
_MIN_SECRET_LEN = 6
#: An Authorization header in any scheme (Bearer, Basic, Token, AWS4-HMAC-SHA256 ...): the value
#: is the rest of the line, or the quoted string in a JSON/dict rendering of the headers.
_AUTH_QUOTED = re.compile(r"""(?i)(authorization["']?\s*[:=]\s*)(["'])(?:\\.|(?!\2).)*\2""")
_AUTH_LINE = re.compile(r"(?i)(authorization\s*[:=]\s*)[^\r\n]*")
_BEARER = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+")


def secret_env_names(environ: dict[str, str] | None = None) -> list[str]:
    env = os.environ if environ is None else environ
    return sorted(n for n in env if _SECRET_NAMES.match(n))


def secret_values(environ: dict[str, str] | None = None) -> list[str]:
    env = os.environ if environ is None else environ
    vals = {env[n] for n in secret_env_names(env) if len(env.get(n) or "") >= _MIN_SECRET_LEN}
    return sorted(vals, key=len, reverse=True)


def redact(text: object, environ: dict[str, str] | None = None) -> str:
    """``text`` with every secret value, and any Authorization header, replaced by ``***``."""
    s = str(text)
    for v in secret_values(environ):
        s = s.replace(v, "***")
    s = _AUTH_QUOTED.sub(r"\1\2***\2", s)
    s = _AUTH_LINE.sub(r"\1***", s)
    return _BEARER.sub("Bearer ***", s)


__all__ = ["API_KEY_ENV", "redact", "secret_env_names", "secret_values"]
