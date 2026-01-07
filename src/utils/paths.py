from pathlib import Path

def project_root() -> Path:
    # src/utils/paths.py -> src/utils -> src -> project root
    return Path(__file__).resolve().parents[2]

def data_dir() -> Path:
    return project_root() / "data"

def mnist_dir() -> Path:
    return data_dir() / "mnist"
    
def artifacts_dir() -> Path:
    return project_root() / "artifacts"