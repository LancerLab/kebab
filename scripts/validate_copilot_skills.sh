#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SKILLS_DIR="$ROOT_DIR/.github/skills"

if [[ ! -d "$SKILLS_DIR" ]]; then
  echo "ERROR: skills directory not found: $SKILLS_DIR"
  exit 1
fi

fail_count=0
skill_count=0

for skill_path in "$SKILLS_DIR"/*; do
  [[ -d "$skill_path" ]] || continue

  folder_name="$(basename "$skill_path")"

  if [[ "$folder_name" == "README.md" ]]; then
    continue
  fi

  skill_file="$skill_path/SKILL.md"
  if [[ ! -f "$skill_file" ]]; then
    echo "FAIL [$folder_name] Missing SKILL.md"
    ((fail_count+=1))
    continue
  fi

  ((skill_count+=1))

  if ! grep -q '^---$' "$skill_file"; then
    echo "FAIL [$folder_name] Missing YAML frontmatter markers"
    ((fail_count+=1))
    continue
  fi

  name_line="$(sed -n '/^---$/,/^---$/p' "$skill_file" | grep -E '^name:' | head -n1 || true)"
  desc_line="$(sed -n '/^---$/,/^---$/p' "$skill_file" | grep -E '^description:' | head -n1 || true)"

  if [[ -z "$name_line" ]]; then
    echo "FAIL [$folder_name] Missing required field: name"
    ((fail_count+=1))
  fi

  if [[ -z "$desc_line" ]]; then
    echo "FAIL [$folder_name] Missing required field: description"
    ((fail_count+=1))
  fi

  skill_name="$(echo "$name_line" | sed 's/^name:[[:space:]]*//')"
  if [[ -n "$skill_name" && "$skill_name" != "$folder_name" ]]; then
    echo "FAIL [$folder_name] name does not match folder: $skill_name"
    ((fail_count+=1))
  fi

  if [[ -n "$skill_name" && ! "$skill_name" =~ ^[a-z0-9-]+$ ]]; then
    echo "FAIL [$folder_name] name must be lowercase letters/numbers/hyphens"
    ((fail_count+=1))
  fi
done

echo "Checked $skill_count skill(s)."

if [[ "$fail_count" -gt 0 ]]; then
  echo "Validation failed with $fail_count error(s)."
  exit 1
fi

echo "Validation passed."
