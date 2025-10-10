import hashlib
from pathlib import Path
from typing import Optional


def get_file_hash(file_path: Path) -> str:
    """Calculate SHA256 hash of file"""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def safe_filename(filename: str) -> str:
    """Convert filename to safe version"""
    import re

    # Remove or replace unsafe characters
    safe_name = re.sub(r"[^a-zA-Z0-9_.-]", "_", filename)
    # Limit length
    if len(safe_name) > 100:
        name, ext = safe_name.rsplit(".", 1)
        safe_name = name[:95] + "." + ext
    return safe_name
