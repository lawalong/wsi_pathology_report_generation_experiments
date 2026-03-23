"""
Shared configuration for the pipeline.
All scripts import paths and settings from here.
"""

import os
import platform
from pathlib import Path
from datetime import datetime

# ============ Auto-detect WSL vs Windows ============
def get_data_root():
    """Get data root path, auto-detecting WSL vs Windows."""
    if platform.system() == "Linux" and os.path.exists("/mnt/z"):
        # WSL - use Linux mount path
        return Path("/mnt/z/Kevin/USYD/dataset")
    else:
        # Native Windows
        return Path(r"Z:\Kevin\USYD\dataset")

def get_reference_root():
    """Get reference folder path."""
    # Reference is in code/reference/TCGA-BRCA
    if platform.system() == "Linux":
        return Path("/mnt/c/Users/lawal/Dropbox/Study/USYD/Maphil/code/reference/TCGA-BRCA")
    else:
        return Path(r"C:\Users\lawal\Dropbox\Study\USYD\Maphil\code\reference\TCGA-BRCA")

# ============ Paths ============
# Data source (SVS files) - external drive
SVS_DIR = get_data_root()

# Reference folder (reports)
REF_DIR = get_reference_root()

# Output directory (in code folder)
if platform.system() == "Linux":
    OUTPUT_DIR = Path("/mnt/c/Users/lawal/Dropbox/Study/USYD/Maphil/code/data")
else:
    OUTPUT_DIR = Path(r"C:\Users\lawal\Dropbox\Study\USYD\Maphil\code\data")

# Sub-directories
FEATURES_DIR = OUTPUT_DIR / "features"
WSI_DIR = OUTPUT_DIR / "wsi"
MASKS_DIR = OUTPUT_DIR / "masks"
TEXT_DIR = OUTPUT_DIR / "text"

# Index files
DATASET_FILE = OUTPUT_DIR / "dataset.jsonl"
SPLITS_FILE = OUTPUT_DIR / "splits.json"
TARGETS_FILE = OUTPUT_DIR / "targets_diagnosis.jsonl"

# Progress tracking
PROGRESS_FILE = OUTPUT_DIR / "progress.jsonl"

# ============ Feature Extraction Settings ============
DEFAULT_NUM_PATCHES = 50
DEFAULT_PATCH_LEVEL = 1
DEFAULT_PATCH_SIZE = 256
BATCH_SIZE = 32

# ============ Split Ratios ============
TRAIN_RATIO = 0.60
VAL_RATIO = 0.20
TEST_RATIO = 0.20
RANDOM_SEED = 42

# ============ Progress Tracking ============
class ProgressTracker:
    """Track processing progress to support incremental runs."""
    
    def __init__(self, progress_file: Path = PROGRESS_FILE):
        self.progress_file = progress_file
        self.completed = self._load_progress()
    
    def _load_progress(self) -> dict:
        """Load completed cases from progress file."""
        completed = {}
        if self.progress_file.exists():
            with open(self.progress_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        entry = __import__("json").loads(line)
                        case_id = entry.get("case_id")
                        if case_id:
                            completed[case_id] = entry
                    except:
                        pass
        return completed
    
    def is_completed(self, case_id: str, step: str = "all") -> bool:
        """Check if a case has completed a specific step."""
        if case_id not in self.completed:
            return False
        entry = self.completed[case_id]
        if step == "all":
            return entry.get("status") == "completed"
        return entry.get(f"{step}_completed", False)
    
    def mark_completed(self, case_id: str, step: str = "all", **extra):
        """Mark a case as completed for a step."""
        import json
        
        # Update in-memory
        if case_id not in self.completed:
            self.completed[case_id] = {"case_id": case_id}
        
        if step == "all":
            self.completed[case_id]["status"] = "completed"
            self.completed[case_id]["completed_at"] = datetime.now().isoformat()
        else:
            self.completed[case_id][f"{step}_completed"] = True
            self.completed[case_id][f"{step}_at"] = datetime.now().isoformat()
        
        # Add extra info
        self.completed[case_id].update(extra)
        
        # Append to file
        with open(self.progress_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(self.completed[case_id], ensure_ascii=False) + "\n")
    
    def mark_failed(self, case_id: str, error: str):
        """Mark a case as failed."""
        import json
        
        entry = {
            "case_id": case_id,
            "status": "failed",
            "error": error,
            "failed_at": datetime.now().isoformat()
        }
        
        self.completed[case_id] = entry
        
        with open(self.progress_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    
    def get_pending(self, all_case_ids: list) -> list:
        """Get list of case IDs that haven't been completed."""
        return [cid for cid in all_case_ids if not self.is_completed(cid)]
    
    def summary(self) -> dict:
        """Get progress summary."""
        completed = sum(1 for e in self.completed.values() if e.get("status") == "completed")
        failed = sum(1 for e in self.completed.values() if e.get("status") == "failed")
        return {
            "total_tracked": len(self.completed),
            "completed": completed,
            "failed": failed
        }


def print_config():
    """Print current configuration."""
    print("=" * 60)
    print("CONFIGURATION")
    print("=" * 60)
    print(f"Platform:     {platform.system()}")
    print(f"SVS_DIR:      {SVS_DIR}")
    print(f"REF_DIR:      {REF_DIR}")
    print(f"OUTPUT_DIR:   {OUTPUT_DIR}")
    print(f"PROGRESS:     {PROGRESS_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    print_config()
    print(f"\nSVS_DIR exists: {SVS_DIR.exists()}")
    print(f"REF_DIR exists: {REF_DIR.exists()}")
    
    if SVS_DIR.exists():
        folders = list(SVS_DIR.iterdir())[:5]
        print(f"\nFirst 5 folders in SVS_DIR: {[f.name for f in folders]}")
