"""Refresh the publication inventory without rewriting original provenance records."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = "PUBLICATION_MANIFEST.json"


def repository_files(root=ROOT):
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=root, check=True, capture_output=True,
    )
    return sorted(set(p.decode("utf-8") for p in result.stdout.split(b"\0") if p), key=str.casefold)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def inventory(root=ROOT):
    rows = []
    for name in repository_files(root):
        path = root / name
        # Ignore unstaged deletions; git add -u will record their removal later.
        if name == MANIFEST or not path.exists():
            continue
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Only regular repository files are supported: {name}")
        rows.append({"path": name, "bytes": path.stat().st_size, "sha256": digest(path)})
    return rows


def refresh(root=ROOT, check=False):
    path = root / MANIFEST
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = inventory(root)
    total = sum(row["bytes"] for row in rows)
    matches = data.get("file_inventory") == rows and data.get("uploaded_content_bytes") == total
    if check:
        if not matches:
            raise ValueError("Inventory is stale. Run: python tools/update_manifest.py")
    else:
        data["file_inventory"] = rows
        data["uploaded_content_bytes"] = total
        # Preserve the original inventory's CRLF style consistently on every OS.
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\r\n")
    return len(rows), total


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    try:
        count, size = refresh(check=args.check)
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        parser.exit(1, str(exc) + "\n")
    print(f"Manifest {'verified' if args.check else 'updated'}: {count} files, {size:,} bytes")
