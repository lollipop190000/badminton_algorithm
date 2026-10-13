from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import urllib.request
from urllib.parse import quote
from pathlib import Path


REPOSITORY = "wywyWang/CoachAI-Projects"
BRANCH = "main"
SOURCE_PREFIX = "CoachAI-Challenge-IJCAI2023/ShuttleSet22/"
API_ROOT = f"https://api.github.com/repos/{REPOSITORY}"


def request_json(url: str) -> dict:
    request = urllib.request.Request(
        url,
        headers={"Accept": "application/vnd.github+json", "User-Agent": "ShuttleSet22-research-pipeline"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def request_bytes(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "ShuttleSet22-research-pipeline"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def download_dataset(destination: Path, force: bool = False) -> list[dict[str, object]]:
    branch = request_json(f"{API_ROOT}/branches/{BRANCH}")
    commit_sha = branch["commit"]["sha"]
    tree_sha = branch["commit"]["commit"]["tree"]["sha"]
    tree = request_json(f"{API_ROOT}/git/trees/{tree_sha}?recursive=1")
    blobs = sorted(
        (
            item
            for item in tree["tree"]
            if item["type"] == "blob" and item["path"].startswith(SOURCE_PREFIX)
        ),
        key=lambda item: item["path"],
    )
    if not blobs:
        raise RuntimeError("The official GitHub tree contained no ShuttleSet22 files.")

    destination.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, object]] = []
    for number, item in enumerate(blobs, start=1):
        relative = item["path"][len(SOURCE_PREFIX) :]
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        status = "existing"
        if force or not target.exists() or target.stat().st_size != int(item["size"]):
            encoded_path = quote(item["path"], safe="/")
            raw_url = f"https://raw.githubusercontent.com/{REPOSITORY}/{BRANCH}/{encoded_path}"
            payload = request_bytes(raw_url)
            if len(payload) != int(item["size"]):
                raise RuntimeError(
                    f"Size mismatch for {relative}: expected {item['size']}, received {len(payload)}"
                )
            target.write_bytes(payload)
            status = "downloaded"
        payload = target.read_bytes()
        manifest.append(
            {
                "relative_path": relative.replace("\\", "/"),
                "size_bytes": len(payload),
                "git_blob_sha": item["sha"],
                "sha256": sha256_bytes(payload),
                "status": status,
                "commit_sha": commit_sha,
            }
        )
        print(f"[{number:03d}/{len(blobs):03d}] {status:10s} {relative}")

    manifest_path = destination / "download_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)

    expected = {row["relative_path"] for row in manifest}
    present = {
        path.relative_to(destination).as_posix()
        for path in destination.rglob("*")
        if path.is_file() and path.name != manifest_path.name
    }
    missing = sorted(expected - present)
    if missing:
        raise RuntimeError(f"Missing downloaded files: {missing}")
    print(f"Downloaded/verified {len(manifest)} official files at commit {commit_sha}.")
    print(f"Manifest: {manifest_path}")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the official ShuttleSet22 tree via GitHub API.")
    parser.add_argument("--destination", type=Path, default=Path("raw/ShuttleSet22"))
    parser.add_argument("--force", action="store_true", help="Redownload even when size matches.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        download_dataset(args.destination.resolve(), force=args.force)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
