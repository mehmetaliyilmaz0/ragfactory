#!/usr/bin/env python3
"""
Release helper — bumps version, commits, tags, and pushes in one shot.

Usage:
    python release.py 0.2.0
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

PYPROJECT = Path(__file__).parent / "pyproject.toml"


def run(cmd: str) -> None:
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"\n✗ Command failed: {cmd}")
        sys.exit(1)


def current_version() -> str:
    text = PYPROJECT.read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    if not match:
        print("✗ Could not find version in pyproject.toml")
        sys.exit(1)
    return match.group(1)


def bump_version(new: str) -> None:
    text = PYPROJECT.read_text(encoding="utf-8")
    updated = re.sub(
        r'^(version\s*=\s*)"[^"]+"',
        f'\\1"{new}"',
        text,
        count=1,
        flags=re.MULTILINE,
    )
    PYPROJECT.write_text(updated, encoding="utf-8")


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python release.py <version>  (e.g. python release.py 0.2.0)")
        sys.exit(1)

    new_version = sys.argv[1].lstrip("v")
    old_version = current_version()

    print(f"\n  {old_version}  →  {new_version}\n")

    bump_version(new_version)
    print(f"✓ pyproject.toml updated to {new_version}")

    run(f'git add pyproject.toml')
    run(f'git commit -m "chore: bump version to {new_version}"')
    run(f'git tag v{new_version}')
    run(f'git push origin main')
    run(f'git push origin v{new_version}')

    print(f"\n✓ Released v{new_version} — GitHub Actions will publish to PyPI.")


if __name__ == "__main__":
    main()
