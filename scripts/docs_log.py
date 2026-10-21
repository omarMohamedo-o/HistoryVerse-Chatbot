"""Documentation and changelog utility script.

Creates standardized markdown fragments for changelogs and bug reports.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def slugify(value: str) -> str:
    """Convert text to safe filename slug."""
    value = re.sub(r"[^\w\s-]", "", value).strip().lower()
    return re.sub(r"[-\s]+", "-", value)


def create_changelog(title: str) -> str:
    """Create a new changelog fragment."""
    os.makedirs("docs/changelog", exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    slug = slugify(title)
    filename = f"docs/changelog/{timestamp}_{slug}.md"

    content = f"""# Changelog: {title}

- **Date**: {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")}
- **Author**: @omarMohamedo-o

## Summary
{title}

## Impacted Modules
- `src/historyverse/`

## Deployment & Verification
- Tests passed: unit & integration
- Schema validations verified
"""
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)
    logger.info("Created changelog fragment: %s", filename)
    return filename


def create_bug(bug_id: str, title: str) -> str:
    """Create a new bug tracking fragment."""
    os.makedirs("docs/bugs", exist_ok=True)
    slug = slugify(title)
    filename = f"docs/bugs/{bug_id}_{slug}.md"

    content = f"""# Bug Report: {bug_id} - {title}

- **Date**: {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")}
- **Identifier**: {bug_id}
- **Author**: @omarMohamedo-o

## Description
{title}

## Failing Test Case
```python
# test_reproduce_{bug_id.lower()}
```

## Resolution Strategy
1. Add failing test to reproduction suite
2. Apply targeted fix without relaxing assertions
3. Ensure test passes and no regressions introduced
"""
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)
    logger.info("Created bug fragment: %s", filename)
    return filename


def main() -> None:
    """CLI entry point for docs_log utility."""
    parser = argparse.ArgumentParser(description="HistoryVerse Docs & Changelog Utility")
    subparsers = parser.add_subparsers(dest="command", required=True)

    new_parser = subparsers.add_parser("new", help="Create a new fragment")
    new_subparsers = new_parser.add_subparsers(dest="type", required=True)

    cl_parser = new_subparsers.add_parser("changelog", help="Create a changelog entry")
    cl_parser.add_argument("title", type=str, help="Title of the change")

    bug_parser = new_subparsers.add_parser("bug", help="Create a bug entry")
    bug_parser.add_argument("bug_id", type=str, help="Bug identifier (e.g. BUG-001)")
    bug_parser.add_argument("title", type=str, help="Title or description of the bug")

    args = parser.parse_args()

    if args.command == "new":
        if args.type == "changelog":
            create_changelog(args.title)
        elif args.type == "bug":
            create_bug(args.bug_id, args.title)


if __name__ == "__main__":
    main()
