#!/usr/bin/env python3
"""Script to push skills to the Hugging Face Hub as datasets.

This script reads skill YAML files from the skills directory and pushes
them to the Hugging Face Hub, updating the leaderboard datasets.
"""

import os
import json
import argparse
from pathlib import Path
from typing import Optional

try:
    from datasets import Dataset
    from huggingface_hub import HfApi, login
except ImportError:
    raise ImportError(
        "Required packages not found. Install with: pip install datasets huggingface_hub"
    )

import yaml


DEFAULT_SKILLS_DIR = Path("skills")
DEFAULT_EVALS_DATASET = "huggingface/skills-evals"
DEFAULT_HACKERS_DATASET = "huggingface/skills-hackers"


def load_skills(skills_dir: Path) -> list[dict]:
    """Load all skill definitions from YAML files in the given directory.

    Args:
        skills_dir: Path to the directory containing skill YAML files.

    Returns:
        List of skill dictionaries.
    """
    skills = []
    for skill_file in sorted(skills_dir.glob("*.yaml")):
        with open(skill_file, "r", encoding="utf-8") as f:
            skill = yaml.safe_load(f)
            skill["_source_file"] = skill_file.name
            skills.append(skill)
    return skills


def skills_to_evals_records(skills: list[dict]) -> list[dict]:
    """Convert skills to evaluation leaderboard records.

    Args:
        skills: List of skill dictionaries.

    Returns:
        List of records suitable for the evals leaderboard dataset.
    """
    records = []
    for skill in skills:
        record = {
            "skill_id": skill.get("id", ""),
            "name": skill.get("name", ""),
            "description": skill.get("description", ""),
            "category": skill.get("category", "uncategorized"),
            "tags": json.dumps(skill.get("tags", [])),
            "prompt": skill.get("prompt", ""),
            "source_file": skill.get("_source_file", ""),
        }
        records.append(record)
    return records


def skills_to_hackers_records(skills: list[dict]) -> list[dict]:
    """Convert skills to hackers leaderboard records.

    Args:
        skills: List of skill dictionaries.

    Returns:
        List of records suitable for the hackers leaderboard dataset.
    """
    records = []
    for skill in skills:
        record = {
            "skill_id": skill.get("id", ""),
            "name": skill.get("name", ""),
            "author": skill.get("author", "unknown"),
            "version": skill.get("version", "1.0.0"),
            "category": skill.get("category", "uncategorized"),
            "created_at": skill.get("created_at", ""),
            "source_file": skill.get("_source_file", ""),
        }
        records.append(record)
    return records


def push_dataset(
    records: list[dict],
    repo_id: str,
    token: Optional[str] = None,
    commit_message: str = "Update skills dataset",
) -> None:
    """Push records to a Hugging Face Hub dataset.

    Args:
        records: List of record dictionaries to push.
        repo_id: The Hub dataset repository ID (e.g., 'org/dataset-name').
        token: Hugging Face API token. Defaults to HF_TOKEN env variable.
        commit_message: Commit message for the push.
    """
    if not records:
        print(f"No records to push to {repo_id}, skipping.")
        return

    hf_token = token or os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)

    dataset = Dataset.from_list(records)
    dataset.push_to_hub(
        repo_id,
        commit_message=commit_message,
        token=hf_token,
    )
    print(f"Successfully pushed {len(records)} records to {repo_id}")


def main():
    parser = argparse.ArgumentParser(
        description="Push skills to Hugging Face Hub leaderboard datasets."
    )
    parser.add_argument(
        "--skills-dir",
        type=Path,
        default=DEFAULT_SKILLS_DIR,
        help="Directory containing skill YAML files.",
    )
    parser.add_argument(
        "--evals-dataset",
        default=DEFAULT_EVALS_DATASET,
        help="Hub dataset repo ID for the evals leaderboard.",
    )
    parser.add_argument(
        "--hackers-dataset",
        default=DEFAULT_HACKERS_DATASET,
        help="Hub dataset repo ID for the hackers leaderboard.",
    )
    parser.add_argument(
        "--target",
        choices=["evals", "hackers", "both"],
        default="both",
        help="Which leaderboard dataset to update.",
    )
    parser.add_argument("--token", help="Hugging Face API token (or set HF_TOKEN env).")
    args = parser.parse_args()

    if not args.skills_dir.exists():
        print(f"Skills directory not found: {args.skills_dir}")
        return

    skills = load_skills(args.skills_dir)
    print(f"Loaded {len(skills)} skills from {args.skills_dir}")

    if args.target in ("evals", "both"):
        evals_records = skills_to_evals_records(skills)
        push_dataset(
            evals_records,
            args.evals_dataset,
            token=args.token,
            commit_message=f"Update evals leaderboard with {len(evals_records)} skills",
        )

    if args.target in ("hackers", "both"):
        hackers_records = skills_to_hackers_records(skills)
        push_dataset(
            hackers_records,
            args.hackers_dataset,
            token=args.token,
            commit_message=f"Update hackers leaderboard with {len(hackers_records)} skills",
        )


if __name__ == "__main__":
    main()
