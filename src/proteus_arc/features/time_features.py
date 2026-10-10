import numpy as np
from scipy.stats import skew, kurtosis
from typing import Dict, Any

class TimeDomainFeatures:
    @staticmethod
    def extract(signal: np.ndarray, channel_name: str) -> Dict[str, float]:
        """
        Extract time-domain features for a 1D signal.
        """
        if signal.ndim != 1:
            raise ValueError("Signal must be 1D.")
            
        features = {}
        prefix = str(channel_name)
        
        features[f"{prefix}_mean"] = float(np.mean(signal))
        features[f"{prefix}_std"] = float(np.std(signal))
        features[f"{prefix}_variance"] = float(np.var(signal))
        features[f"{prefix}_rms"] = float(np.sqrt(np.mean(signal ** 2)))
        features[f"{prefix}_minimum"] = float(np.min(signal))
        features[f"{prefix}_maximum"] = float(np.max(signal))
        features[f"{prefix}_range"] = float(np.max(signal) - np.min(signal))
        features[f"{prefix}_skewness"] = float(skew(signal, bias=False))
        features[f"{prefix}_kurtosis"] = float(kurtosis(signal, bias=False))
        
        # Hjorth parameters
        activity = np.var(signal)
        first_derivative = np.diff(signal)
        second_derivative = np.diff(first_derivative)
        
        first_var = np.var(first_derivative) if len(first_derivative) > 0 else 0.0
        second_var = np.var(second_derivative) if len(second_derivative) > 0 else 0.0
        
        mobility = np.sqrt(first_var / activity) if activity > 0 else 0.0
        complexity = (
            np.sqrt(second_var / first_var) / mobility
            if first_var > 0 and mobility > 0 else 0.0
        )
        
        features[f"{prefix}_hjorth_activity"] = float(activity)
        features[f"{prefix}_hjorth_mobility"] = float(mobility)
        features[f"{prefix}_hjorth_complexity"] = float(complexity)
        
        # Handle NaN values explicitly (convert to 0 or leave for imputer)
        for k, v in features.items():
            if not np.isfinite(v):
                features[k] = 0.0
                
        return features
