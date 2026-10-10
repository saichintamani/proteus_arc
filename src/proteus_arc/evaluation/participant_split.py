import numpy as np
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from typing import Tuple

class ParticipantSplitter:
    """
    Ensures participant-independent splits to prevent epoch leakage.
    """
    def __init__(self, method: str = "group_shuffle", test_size: float = 0.2, n_splits: int = 5, random_state: int = 42):
        self.method = method
        self.test_size = test_size
        self.n_splits = n_splits
        self.random_state = random_state
        
    def split(self, X: np.ndarray, y: np.ndarray, participant_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Splits data into train and test sets strictly isolating participants.
        Returns:
            X_train, X_test, y_train, y_test, groups_train, groups_test
        """
        if not (len(X) == len(y) == len(participant_ids)):
            raise ValueError("Lengths of X, y, and participant_ids must match.")
            
        if self.method == "group_shuffle":
            splitter = GroupShuffleSplit(n_splits=1, test_size=self.test_size, random_state=self.random_state)
            train_idx, test_idx = next(splitter.split(X, y, groups=participant_ids))
        elif self.method == "stratified_group":
            splitter = StratifiedGroupKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
            # Take the first fold
            train_idx, test_idx = next(splitter.split(X, y, groups=participant_ids))
        else:
            raise ValueError(f"Unknown split method: {self.method}")
            
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        groups_train, groups_test = participant_ids[train_idx], participant_ids[test_idx]
        
        # Verify isolation
        train_p = set(groups_train)
        test_p = set(groups_test)
        if not train_p.isdisjoint(test_p):
            overlap = train_p.intersection(test_p)
            raise RuntimeError(f"Participant leakage detected! Participants in both sets: {overlap}")
            
        if len(train_p) == 0 or len(test_p) == 0:
            raise RuntimeError("Split resulted in empty train or test set. Not enough participants.")
            
        return X_train, X_test, y_train, y_test, groups_train, groups_test
