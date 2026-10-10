import pandas as pd
from pathlib import Path
from typing import List, Dict

class DatasetManifest:
    """
    Manages the manifest of valid EEG recordings linked to participants.
    """
    def __init__(self):
        self.records = []
        
    def add_recording(self, participant_id: str, label: str, file_path: str, sfreq: float, channels: List[str], duration_sec: float):
        self.records.append({
            "participant_id": participant_id,
            "label": label,
            "file_path": str(file_path),
            "sfreq": sfreq,
            "n_channels": len(channels),
            "channels": ",".join(channels),
            "duration_sec": duration_sec
        })
        
    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.records)
        
    def save(self, path: str):
        df = self.to_dataframe()
        df.to_csv(path, index=False)
        
    @classmethod
    def load(cls, path: str) -> 'DatasetManifest':
        manifest = cls()
        if Path(path).exists():
            df = pd.read_csv(path)
            manifest.records = df.to_dict('records')
        return manifest

    def get_summary(self) -> Dict:
        df = self.to_dataframe()
        if df.empty:
            return {"total_recordings": 0}
        return {
            "total_recordings": len(df),
            "participants": df["participant_id"].nunique(),
            "class_distribution": df["label"].value_counts().to_dict(),
            "avg_duration_sec": df["duration_sec"].mean(),
            "sampling_rates": df["sfreq"].unique().tolist()
        }
