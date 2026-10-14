"""Dependency-free checks for the organized research repository."""
import argparse
import ast
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
from urllib.parse import unquote, urlsplit

from update_manifest import ROOT, digest, refresh, repository_files

TOPIC = re.compile(r"\d{2}_[a-z0-9_]+$")
HEADINGS = ("## 1. 원인 / 하는 이유 / 개요", "## 2. 과정", "## 3. 결과")
LIMIT = 50 * 1024 * 1024
FORBIDDEN_PARTS = {".runtime", "__pycache__", "node_modules", "raw", "_archive"}


def path_errors(name, size):
    path = PurePosixPath(name)
    errors = []
    if any(p in FORBIDDEN_PARTS or p.startswith(".venv") for p in path.parts):
        errors.append(f"Generated/raw/private directory is not publishable: {name}")
    if path.name == ".env" or path.name.startswith(".env.") or path.suffix in {".pyc", ".pyo", ".pem", ".key"}:
        errors.append(f"Private/generated file is not publishable: {name}")
    if size > LIMIT:
        errors.append(f"File exceeds 50 MiB: {name} ({size:,} bytes)")
    return errors


def report_errors(path, root):
    text = path.read_text(encoding="utf-8")
    positions = [text.find(h) for h in HEADINGS]
    errors = []
    if -1 in positions or positions != sorted(positions):
        errors.append(f"Required REPORT headings missing/out of order: {path.relative_to(root)}")
    for target in re.findall(r"\]\(([^\s)]+)\)", text):
        target = target.strip("<>")
        parsed = urlsplit(target)
        if parsed.scheme or target.startswith("#"):
            continue
        resolved = (path.parent / unquote(parsed.path)).resolve()
        if not resolved.is_relative_to(root.resolve()) or not resolved.exists():
            errors.append(f"Broken/outside REPORT link: {path.relative_to(root)} -> {target}")
    return errors


def run_errors(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    errors = []
    required = {"schema_version", "run_id", "status", "code_commit", "command", "seed", "environment", "inputs", "config", "split", "metrics", "outputs", "limitations"}
    if not isinstance(data, dict) or not required.issubset(data):
        return [f"Incomplete RUN.json: {path}"]
    if data["schema_version"] != 1 or data["run_id"] != path.parent.name:
        errors.append(f"RUN schema/id mismatch: {path}")
    if data["status"] not in {"completed", "failed", "not_run"}:
        errors.append(f"Invalid RUN status: {path}")
    if "REPLACE" in json.dumps(data) or not data["command"] or not data["split"]:
        errors.append(f"RUN still has placeholders or missing reproduction fields: {path}")
    if not re.fullmatch(r"[0-9a-f]{40}", str(data["code_commit"])):
        errors.append(f"RUN code_commit must identify executed code: {path}")
    inputs = data["inputs"]
    config = data["config"]
    if not isinstance(inputs, list) or not inputs or not isinstance(config, dict):
        errors.append(f"RUN needs input and configuration provenance: {path}")
    else:
        for item in [*inputs, config]:
            if not isinstance(item, dict) or not re.fullmatch(r"[0-9a-f]{64}", str(item.get("sha256", ""))):
                errors.append(f"RUN needs actual input/config SHA256: {path}")
    outputs = data["outputs"]
    if not isinstance(outputs, list) or (data["status"] == "completed" and not outputs):
        errors.append(f"Completed RUN needs output records: {path}")
    elif isinstance(outputs, list):
        for item in outputs:
            if not isinstance(item, dict) or not isinstance(item.get("path"), str):
                errors.append(f"RUN output needs path and SHA256: {path}")
                continue
            output = (path.parent / item["path"]).resolve()
            if not output.is_relative_to(path.parent.resolve()) or not output.is_file() or output == path.resolve():
                errors.append(f"RUN output missing/outside run: {item['path']}")
            elif digest(output) != item.get("sha256"):
                errors.append(f"RUN output SHA256 mismatch: {item['path']}")
    return errors


def changed_report_errors(names):
    changed = set(names)
    errors = []
    for name in sorted(changed):
        parts = PurePosixPath(name).parts
        if len(parts) < 3 or not TOPIC.fullmatch(parts[0]):
            continue
        research = parts[1] == "results" or (parts[1] == "code" and not name.endswith(".md"))
        if research and f"{parts[0]}/REPORT.md" not in changed:
            message = f"Research changed without REPORT update: {parts[0]}/REPORT.md"
            if message not in errors:
                errors.append(message)
    return errors


def topic_number_errors(names):
    used = {}
    errors = []
    for name in names:
        if not TOPIC.fullmatch(name):
            errors.append(f"Invalid topic folder name (use NN_topic): {name}")
        elif name[:2] in used:
            errors.append(f"Topic number collision: {used[name[:2]]}, {name}")
        else:
            used[name[:2]] = name
    return errors


def validate(root=ROOT, base=None):
    files = repository_files(root)
    errors = []
    owners = {}
    errors.extend(topic_number_errors(p.name for p in root.iterdir() if p.is_dir() and re.match(r"^\d", p.name)))
    topics = sorted(p for p in root.iterdir() if p.is_dir() and TOPIC.fullmatch(p.name))
    manifest = json.loads((root / "PUBLICATION_MANIFEST.json").read_text(encoding="utf-8"))
    declared = set(manifest["modules"])
    actual = {p.name for p in topics}
    if declared != actual:
        errors.append("Topic index differs from manifest.modules; update both the manifest metadata and README")
    readme = (root / "README.md").read_text(encoding="utf-8")
    for topic in topics:
        for name in ("code", "results"):
            if not (topic / name).is_dir() or not any(f.startswith(f"{topic.name}/{name}/") for f in files):
                errors.append(f"Topic needs tracked {name}/ content: {topic.name}")
        report = topic / "REPORT.md"
        if not report.is_file():
            errors.append(f"Missing REPORT: {topic.name}")
        else:
            errors.extend(report_errors(report, root))
        if f"{topic.name}/REPORT.md" not in readme:
            errors.append(f"Missing root index link: {topic.name}")
    for name in files:
        path = root / name
        if not path.exists():
            continue
        if path.is_symlink():
            errors.append(f"Symlink is not allowed: {name}")
            continue
        errors.extend(path_errors(name, path.stat().st_size))
        if path.suffix == ".py":
            try:
                ast.parse(path.read_text(encoding="utf-8-sig"), filename=name)
            except (SyntaxError, UnicodeError) as exc:
                errors.append(f"Python syntax: {name}: {exc}")
        parts = PurePosixPath(name).parts
        if len(parts) > 3 and TOPIC.fullmatch(parts[0]) and parts[1] == "code" and parts[2] in {"src", "scripts", "configs", "tests", "app", "bfmd_zago", "notebooks"}:
            runtime_name = "/".join(parts[2:])
            if runtime_name in owners:
                errors.append(f"Duplicate runtime ownership: {owners[runtime_name]}, {name}")
            owners[runtime_name] = name
        if len(parts) >= 5 and TOPIC.fullmatch(parts[0]) and parts[1:3] == ("results", "runs"):
            metadata = root.joinpath(*parts[:4], "RUN.json")
            if not metadata.exists():
                errors.append(f"New run needs RUN.json: {'/'.join(parts[:4])}")
            elif path == metadata:
                errors.extend(run_errors(metadata))
    if base:
        result = subprocess.run(["git", "diff", "--name-only", "--no-renames", "-z", f"{base}...HEAD"], cwd=root, check=True, capture_output=True)
        errors.extend(changed_report_errors(p.decode("utf-8") for p in result.stdout.split(b"\0") if p))
    try:
        refresh(root, check=True)
    except ValueError as exc:
        errors.append(str(exc))
    return list(dict.fromkeys(errors)), len(files), len(topics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", help="PR base commit; require REPORT for changed code/results")
    args = parser.parse_args()
    try:
        errors, count, topics = validate(base=args.base)
    except (OSError, ValueError, TypeError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"Validation could not complete: {exc}\n")
    if errors:
        parser.exit(1, "\n".join(errors) + "\n")
    print(f"Repository validation passed: {count} files, {topics} topics; syntax, layout, links, ownership, size and inventory verified")
