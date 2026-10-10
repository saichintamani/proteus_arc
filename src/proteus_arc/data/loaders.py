import os
from pathlib import Path
import mne
import pandas as pd
from src.proteus_arc.data.metadata import DatasetMetadata
from src.proteus_arc.data.manifest import DatasetManifest

class EEGLoader:
    """
    Scans the dataset directory and creates a manifest of valid recordings.
    """
    def __init__(self, data_dir: str, config_path: str = "configs/data.yaml"):
        self.data_dir = Path(data_dir)
        self.metadata = DatasetMetadata(data_dir, config_path)
        self.manifest = DatasetManifest()
        self.config = self.metadata.config

    def discover_recordings(self, check_mne: bool = False) -> DatasetManifest:
        """
        Discovers recordings in the derivatives folder and matches them to participants.
        If check_mne is True, actually opens the file to verify channels and sfreq.
        """
        derivatives_dir = self.data_dir / "derivatives"
        if not derivatives_dir.exists():
            print(f"Warning: Derivatives directory {derivatives_dir} not found.")
            return self.manifest

        participant_ids = self.metadata.participants.get("participant_id", pd.Series()).tolist()
        
        for participant_id in participant_ids:
            # Expected file pattern for ds004504 derivatives
            eeg_file = derivatives_dir / participant_id / "eeg" / f"{participant_id}_task-eyesclosed_eeg.set"
            
            if not eeg_file.exists():
                print(f"Missing recording for {participant_id} at {eeg_file}")
                continue
                
            info = self.metadata.get_participant_info(participant_id)
            if not info:
                continue
                
            group = info.get("Group")
            label_map = self.config.get("mapping", {})
            label = label_map.get(group, group)
            
            sfreq = 0.0
            channels = []
            duration = 0.0
            
            if check_mne:
                try:
                    # Preload=False to just read metadata
                    raw = mne.io.read_raw_eeglab(str(eeg_file), preload=False, verbose='ERROR')
                    sfreq = raw.info['sfreq']
                    channels = raw.ch_names
                    duration = raw.times[-1] if len(raw.times) > 0 else 0.0
                except Exception as e:
                    print(f"Error reading {eeg_file}: {e}")
                    continue
                    
            self.manifest.add_recording(
                participant_id=participant_id,
                label=label,
                file_path=str(eeg_file),
                sfreq=sfreq,
                channels=channels,
                duration_sec=duration
            )
            
        return self.manifest
