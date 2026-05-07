"""
utils.py - Utility helpers for YouTube Video Analyzer
"""

import re
from urllib.parse import urlparse, parse_qs


def extract_video_id(url_or_id: str) -> str | None:
    """
    Extract YouTube Video ID from various URL formats or raw ID.

    Supported formats:
      - https://www.youtube.com/watch?v=VIDEO_ID
      - https://youtu.be/VIDEO_ID
      - https://www.youtube.com/embed/VIDEO_ID
      - https://youtube.com/shorts/VIDEO_ID
      - Raw 11-character video ID
    """
    url_or_id = url_or_id.strip()

    # Raw video ID (11 alphanumeric + dash/underscore chars)
    if re.match(r'^[A-Za-z0-9_\-]{11}$', url_or_id):
        return url_or_id

    try:
        parsed = urlparse(url_or_id)

        # youtu.be/VIDEO_ID
        if parsed.netloc in ("youtu.be", "www.youtu.be"):
            vid = parsed.path.lstrip("/").split("/")[0]
            return vid if vid else None

        # youtube.com/watch?v=VIDEO_ID
        if "youtube.com" in parsed.netloc:
            # /watch?v=
            qs = parse_qs(parsed.query)
            if "v" in qs:
                return qs["v"][0]
            # /embed/VIDEO_ID  or  /shorts/VIDEO_ID
            path_parts = [p for p in parsed.path.split("/") if p]
            if len(path_parts) >= 2 and path_parts[0] in ("embed", "shorts", "v"):
                return path_parts[1]

    except Exception:
        pass

    return None


def format_number(n: int) -> str:
    """
    Format large numbers into human-readable strings.
    e.g. 1_234_567 → '1.2M'
    """
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.1f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def truncate_text(text: str, max_len: int = 300) -> str:
    """Truncate text to max_len chars, appending ellipsis if needed."""
    if len(text) <= max_len:
        return text
    return text[:max_len].rsplit(" ", 1)[0] + "…"
