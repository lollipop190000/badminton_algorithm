"""Read back collaboration controls. GH_TOKEN needs repository administration:read."""
import json
import os
from pathlib import Path
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://api.github.com/repos/lollipop190000/badminton_algorithm"


def get(path):
    request = urllib.request.Request(BASE + path, headers={
        "Authorization": "Bearer " + os.environ["GH_TOKEN"],
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    })
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def mismatches(actual, expected, prefix=""):
    failures = []
    for key, value in expected.items():
        name = f"{prefix}.{key}" if prefix else key
        got = actual.get(key)
        if isinstance(value, dict):
            if not isinstance(got, dict):
                failures.append(name)
            else:
                failures.extend(mismatches(got, value, name))
        elif got != value:
            failures.append(name)
    return failures


def main():
    if not os.environ.get("GH_TOKEN"):
        sys.exit("Set GH_TOKEN in your environment; never commit a token.")
    protection = get("/branches/main/protection")
    expected = json.loads((ROOT / ".github/main-protection.json").read_text())
    # GitHub serializes boolean protection flags as {enabled: bool}.
    for key in ("enforce_admins", "required_linear_history", "allow_force_pushes", "allow_deletions", "block_creations", "required_conversation_resolution", "lock_branch", "allow_fork_syncing"):
        protection[key] = protection.get(key, {}).get("enabled", False)
    protection.setdefault("restrictions", None)
    # Ignore server-generated app/link fields; compare the two expected check identities.
    checks = protection["required_status_checks"]["checks"]
    protection["required_status_checks"]["checks"] = [{"context": x["context"], "app_id": x["app_id"]} for x in checks]
    errors = mismatches(protection, expected)
    repo = get("")
    repo_policy = json.loads((ROOT / ".github/repository-settings.json").read_text())
    errors.extend(mismatches(repo, repo_policy))
    errors.extend(mismatches(get("/actions/permissions/workflow"), {"default_workflow_permissions": "read", "can_approve_pull_request_reviews": False}))
    if errors:
        sys.exit("GitHub settings differ from policy: " + ", ".join(errors))
    print("GitHub policy verified: independent approval, CODEOWNERS, both Actions checks, up-to-date main, conversations, administrators, linear history, no force/deletion, squash-only, branch cleanup and read-only Actions")


if __name__ == "__main__":
    main()
