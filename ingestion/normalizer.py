"""Phase-1 text normalisation for inbound comments.

Cleans tracking parameters out of URLs, masks obvious PII (email addresses,
phone numbers), collapses messy whitespace, and trims decorative quoting —
all before fingerprinting/storage so downstream analytics see stable text.
"""

from __future__ import annotations

import re
import urllib.parse

# ---------------------------------------------------------------------------
# URL cleaning — strip tracking params, keep meaningful ones
# ---------------------------------------------------------------------------

_URL_RE = re.compile(r"""https?://[^\s<>"')\]]+""", re.IGNORECASE)

_TRACKING_PARAMS = frozenset({
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "utm_id", "gclid", "gbraid", "wbraid", "fbclid", "msclkid",
    "mc_cid", "mc_eid", "igshid", "_hsenc", "_hsmi", "vero_id", "yclid",
})


def _clean_url(match: re.Match[str]) -> str:
    """Rewrite one matched URL with tracking parameters removed."""
    url = match.group(0)
    try:
        parts = urllib.parse.urlsplit(url)
    except ValueError:
        return url  # malformed URL — leave untouched rather than kill ingestion
    if not parts.query:
        return url
    all_pairs = urllib.parse.parse_qsl(parts.query, keep_blank_values=True)
    kept = [(k, v) for k, v in all_pairs if k.lower() not in _TRACKING_PARAMS]
    if len(kept) == len(all_pairs):
        return url  # nothing to strip
    if not kept:
        return urllib.parse.urlunsplit(parts._replace(query=""))
    return urllib.parse.urlunsplit(parts._replace(query=urllib.parse.urlencode(kept)))


# ---------------------------------------------------------------------------
# PII masking
# ---------------------------------------------------------------------------

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

# International layout: +<country> <groups>, e.g. "+1 555 123 4567"
_PHONE_INTL_RE = re.compile(r"\+\d{1,3}[\s\-().]*\d{2,4}(?:[\s\-().]*\d{2,4}){1,4}")

# Common local layouts: 555-867-5309 / 555 867 5309 / (555) 867-5309
_PHONE_LOCAL_RE = re.compile(r"(?:\(\d{2,4}\)[\s\-]?)?\d{3}[\s\-]\d{3}[\s\-]\d{4}")

_EMAIL_TOKEN = "[EMAIL]"
_PHONE_TOKEN = "[PHONE]"

_QUOTE_CHARS = "\"'“”‘’"


def sanitize_comment_text(text: str) -> str:
    """Return a cleaned copy of *text*, safe for storage and fingerprinting."""
    cleaned = _URL_RE.sub(_clean_url, text)
    cleaned = _EMAIL_RE.sub(_EMAIL_TOKEN, cleaned)
    cleaned = _PHONE_INTL_RE.sub(_PHONE_TOKEN, cleaned)
    cleaned = _PHONE_LOCAL_RE.sub(_PHONE_TOKEN, cleaned)

    cleaned = re.sub(r"[ \t]+", " ", cleaned)     # collapse runs of spaces/tabs
    cleaned = re.sub(r" ?\n ?", "\n", cleaned)    # no padding around newlines
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)  # cap blank-line runs at one gap

    cleaned = cleaned.strip().strip(_QUOTE_CHARS).strip()
    return cleaned
