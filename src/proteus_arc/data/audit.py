"""
Dataset audit module for Proteus Arc.
"""
import os
import pandas as pd
from pathlib import Path

def audit_dataset(data_dir: str):
    """
    Audits the dataset directory to ensure required BIDS layout and files exist.
    """
    base_path = Path(data_dir)
    report = {
        "dataset_exists": False,
        "participants_file_exists": False,
        "derivatives_dir_exists": False,
        "total_participants": 0,
        "class_counts": {},
        "missing_files_error": None
    }
    
    if not base_path.exists():
        report["missing_files_error"] = f"Dataset directory '{data_dir}' not found."
        return report
        
    report["dataset_exists"] = True
    
    participants_file = base_path / "participants.tsv"
    if participants_file.exists():
        report["participants_file_exists"] = True
        try:
            df = pd.read_csv(participants_file, sep='\t')
            report["total_participants"] = len(df)
            if "Group" in df.columns:
                report["class_counts"] = df["Group"].value_counts().to_dict()
        except Exception as e:
            report["missing_files_error"] = f"Error reading participants.tsv: {e}"
    else:
        report["missing_files_error"] = "participants.tsv not found in dataset root."
        
    derivatives_dir = base_path / "derivatives"
    if derivatives_dir.exists() and derivatives_dir.is_dir():
        report["derivatives_dir_exists"] = True
        
    return report

if __name__ == "__main__":
    import json
    # Run audit on default ds004504
    res = audit_dataset("ds004504")
    print(json.dumps(res, indent=4))
