"""Collect entry points from upstream pants BUILD files into one installable shim dist."""

import ast
import sys
from collections import defaultdict
from pathlib import Path

src = Path(sys.argv[1])
out = Path(sys.argv[2])
entry_points: dict[str, dict[str, str]] = defaultdict(dict)

for build in src.rglob("BUILD"):
    try:
        tree = ast.parse(build.read_text())
    except SyntaxError as e:
        print(f"skip {build}: {e}", file=sys.stderr)
        continue
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "python_distribution":
            for kw in node.keywords:
                if kw.arg == "entry_points":
                    try:
                        value = ast.literal_eval(kw.value)
                    except ValueError:
                        print(f"skip non-literal entry_points in {build}", file=sys.stderr)
                        continue
                    for group, entries in value.items():
                        entry_points[group].update(entries)

out.mkdir(parents=True, exist_ok=True)
lines = [
    "[build-system]", 'requires = ["setuptools>=69"]', 'build-backend = "setuptools.build_meta"', "",
    "[project]", 'name = "backendai-dev-entrypoints"', 'version = "0.0.0"', "",
    "[tool.setuptools]", "packages = []", "",
]
for group, entries in sorted(entry_points.items()):
    lines.append(f'[project.entry-points."{group}"]')
    for name, target in sorted(entries.items()):
        lines.append(f'"{name}" = "{target}"')
    lines.append("")
(out / "pyproject.toml").write_text("\n".join(lines))
print({g: len(e) for g, e in sorted(entry_points.items())})
