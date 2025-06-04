import os

structure = [
    "data/external",
    "data/processed",
    "data/raw",
    "networks",
    "notebooks",
]

files = ["README.md", ".gitignore", ".lefthook.yml", "pyproject.toml"]

for folder in structure:
    os.makedirs(folder, exist_ok=True)

for file in files:
    open(file, "a").close()
