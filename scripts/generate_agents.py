#!/usr/bin/env python3
"""Generate agent configurations from skills definitions.

This script reads skill definitions and generates agent configuration files
for the marketplace, supporting both Claude and Cursor plugin formats.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any

SKILLS_DIR = Path("skills")
CLAUDE_MARKETPLACE = Path(".claude-plugin/marketplace.json")
CURSOR_MARKETPLACE = Path(".cursor-plugin/marketplace.json")


def load_skill(skill_path: Path) -> dict[str, Any]:
    """Load a skill definition from a JSON file."""
    with open(skill_path, "r", encoding="utf-8") as f:
        return json.load(f)


def skill_to_claude_agent(skill: dict[str, Any]) -> dict[str, Any]:
    """Convert a skill definition to a Claude plugin agent entry."""
    return {
        "id": skill["id"],
        "name": skill["name"],
        "description": skill.get("description", ""),
        "version": skill.get("version", "1.0.0"),
        "author": skill.get("author", "community"),
        "tags": skill.get("tags", []),
        "capabilities": skill.get("capabilities", []),
        "prompt": skill.get("system_prompt", ""),
        "examples": skill.get("examples", []),
        "created_at": skill.get("created_at", ""),
        "updated_at": skill.get("updated_at", ""),
    }


def skill_to_cursor_agent(skill: dict[str, Any]) -> dict[str, Any]:
    """Convert a skill definition to a Cursor plugin agent entry."""
    return {
        "id": skill["id"],
        "name": skill["name"],
        "description": skill.get("description", ""),
        "version": skill.get("version", "1.0.0"),
        "tags": skill.get("tags", []),
        "rules": skill.get("cursor_rules", skill.get("system_prompt", "")),
    }


def load_existing_marketplace(path: Path) -> dict[str, Any]:
    """Load existing marketplace JSON, returning empty structure if not found."""
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"agents": [], "version": "1.0.0", "generated": True}


def save_marketplace(path: Path, data: dict[str, Any]) -> None:
    """Save marketplace data to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"Saved marketplace to {path}")


def generate_agents() -> int:
    """Main function to generate agent configurations from skills.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    if not SKILLS_DIR.exists():
        print(f"Skills directory '{SKILLS_DIR}' not found. Creating empty marketplaces.")
        for path in [CLAUDE_MARKETPLACE, CURSOR_MARKETPLACE]:
            save_marketplace(path, {"agents": [], "version": "1.0.0", "generated": True})
        return 0

    skill_files = list(SKILLS_DIR.glob("**/*.json"))
    if not skill_files:
        print("No skill files found.")
        return 0

    print(f"Found {len(skill_files)} skill file(s).")

    claude_marketplace = load_existing_marketplace(CLAUDE_MARKETPLACE)
    cursor_marketplace = load_existing_marketplace(CURSOR_MARKETPLACE)

    claude_agents: list[dict[str, Any]] = []
    cursor_agents: list[dict[str, Any]] = []

    errors = 0
    for skill_path in sorted(skill_files):
        try:
            skill = load_skill(skill_path)
            claude_agents.append(skill_to_claude_agent(skill))
            cursor_agents.append(skill_to_cursor_agent(skill))
            print(f"  Processed: {skill_path.name} -> {skill.get('id', 'unknown')}")
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  ERROR processing {skill_path}: {e}", file=sys.stderr)
            errors += 1

    claude_marketplace["agents"] = claude_agents
    cursor_marketplace["agents"] = cursor_agents

    save_marketplace(CLAUDE_MARKETPLACE, claude_marketplace)
    save_marketplace(CURSOR_MARKETPLACE, cursor_marketplace)

    print(f"\nGenerated {len(claude_agents)} agent(s). Errors: {errors}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(generate_agents())
