import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import yaml

class DatasetMetadata:
    def __init__(self, data_dir: str, config_path: str = "configs/data.yaml"):
        self.data_dir = Path(data_dir)
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.participants = self._load_participants()
        
    def _load_config(self) -> Dict:
        if not self.config_path.exists():
            return {}
        with open(self.config_path, "r") as f:
            return yaml.safe_load(f).get("dataset", {})

    def _load_participants(self) -> pd.DataFrame:
        participants_file = self.data_dir / "participants.tsv"
        if not participants_file.exists():
            return pd.DataFrame()
        return pd.read_csv(participants_file, sep='\t')
        
    def validate_metadata(self) -> Dict[str, any]:
        report = {
            "valid": True,
            "errors": [],
            "participants_count": 0,
            "class_distribution": {}
        }
        
        if self.participants.empty:
            report["valid"] = False
            report["errors"].append("participants.tsv is missing or empty.")
            return report
            
        report["participants_count"] = len(self.participants)
        
        if "Group" not in self.participants.columns:
            report["valid"] = False
            report["errors"].append("Column 'Group' missing from participants.tsv")
            return report
            
        if "participant_id" not in self.participants.columns:
            report["valid"] = False
            report["errors"].append("Column 'participant_id' missing from participants.tsv")
            return report
            
        report["class_distribution"] = self.participants["Group"].value_counts().to_dict()
        
        expected_classes = self.config.get("classes", [])
        if expected_classes:
            found_classes = set(report["class_distribution"].keys())
            missing_classes = set(expected_classes) - found_classes
            if missing_classes:
                report["errors"].append(f"Missing expected classes: {missing_classes}")
                
        return report

    def get_participant_info(self, participant_id: str) -> Optional[Dict]:
        if self.participants.empty or "participant_id" not in self.participants.columns:
            return None
        row = self.participants[self.participants["participant_id"] == participant_id]
        if row.empty:
            return None
        return row.iloc[0].to_dict()
