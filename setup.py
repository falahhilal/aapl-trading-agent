import os

# All files to create with their correct paths
files = [
    "config.py",
    "main.py",
    "requirements.txt",
    ".gitignore",
    "data/__init__.py",
    "data/collector.py",
    "data/preprocessor.py",
    "features/__init__.py",
    "features/technical.py",
    "features/sentiment.py",
    "agent/__init__.py",
    "agent/classifier.py",
    "agent/heuristic.py",
    "agent/backtester.py",
    "evaluation/__init__.py",
    "evaluation/metrics.py",
]

folders = [
    "raw_data",
    "outputs",
    "models",
]

print("Creating folders...")
for folder in folders:
    os.makedirs(folder, exist_ok=True)
    print(f"  created folder: {folder}")

print("\nCreating files...")
for filepath in files:
    # Create parent folder if needed
    parent = os.path.dirname(filepath)
    if parent:
        os.makedirs(parent, exist_ok=True)
    # Create empty file
    with open(filepath, "w") as f:
        pass
    print(f"  created: {filepath}")

print("\nDone. All files and folders created successfully.")