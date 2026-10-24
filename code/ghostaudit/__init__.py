"""Ghost Data Still Exists: evidence-driven deletion audit."""

from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = CODE_ROOT.parent
DATASET_DIR = REPO_ROOT / "dataset"
