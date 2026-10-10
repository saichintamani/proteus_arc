import numpy as np

class QualityControl:
    """
    Validates signal quality and produces quality checks.
    """
    @staticmethod
    def check_nan_inf(data: np.ndarray) -> bool:
        return np.isfinite(data).all()
        
    @staticmethod
    def identify_flat_channels(data: np.ndarray, threshold: float = 1e-10) -> list:
        """
        Identifies channels that have extremely low variance (flat signals).
        """
        variances = np.var(data, axis=1)
        flat = np.where(variances < threshold)[0].tolist()
        return flat
