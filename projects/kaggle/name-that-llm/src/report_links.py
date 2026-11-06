"""Rebase simple local Markdown links when moving a project report."""
import os
from pathlib import Path
import re
from urllib.parse import urlsplit


def rebase_local_links(text, source_dir, destination_dir, moved_targets=None):
    """Preserve link destinations, including files moved together with a report.

    These project reports use inline links without titles or escaped parentheses.
    External URLs and same-document fragments are preserved verbatim.
    """
    source_dir, destination_dir = Path(source_dir).resolve(), Path(destination_dir).resolve()
    moved_targets = {Path(k).resolve(): Path(v).resolve() for k, v in (moved_targets or {}).items()}

    def replace(match):
        target = match.group(1)
        parts = urlsplit(target)
        if not target or target.startswith(('#', '/', '\\')) or parts.scheme:
            return match.group(0)
        old = (source_dir / parts.path).resolve()
        new = moved_targets.get(old, old)
        result = Path(os.path.relpath(new, destination_dir)).as_posix()
        if parts.query:
            result += '?' + parts.query
        if parts.fragment:
            result += '#' + parts.fragment
        return '](' + result + ')'

    return re.sub(r'\]\(([^)\n]+)\)', replace, text)
