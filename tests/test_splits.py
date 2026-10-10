import unittest
import numpy as np
from src.proteus_arc.evaluation.participant_split import ParticipantSplitter

class TestParticipantSplit(unittest.TestCase):
    def setUp(self):
        # 10 participants, 5 epochs each = 50 total epochs
        self.n_participants = 10
        self.epochs_per_participant = 5
        self.X = np.random.randn(self.n_participants * self.epochs_per_participant, 2, 100)
        self.y = np.array(["AD", "CN"] * (self.n_participants // 2)).repeat(self.epochs_per_participant)
        self.groups = np.array([f"sub-{i}" for i in range(self.n_participants)]).repeat(self.epochs_per_participant)
        
    def test_split_isolation(self):
        splitter = ParticipantSplitter(method="group_shuffle", test_size=0.2, random_state=42)
        X_tr, X_te, y_tr, y_te, g_tr, g_te = splitter.split(self.X, self.y, self.groups)
        
        train_set = set(g_tr)
        test_set = set(g_te)
        
        # The sets must be disjoint
        self.assertTrue(train_set.isdisjoint(test_set))
        
        # Total participants should sum to 10
        self.assertEqual(len(train_set) + len(test_set), self.n_participants)
        
    def test_insufficient_data(self):
        # Only 1 participant
        X = np.random.randn(5, 2, 100)
        y = np.array(["AD"] * 5)
        groups = np.array(["sub-1"] * 5)
        
        splitter = ParticipantSplitter(method="group_shuffle", test_size=0.2)
        with self.assertRaises((RuntimeError, ValueError)):
            splitter.split(X, y, groups)
            
if __name__ == "__main__":
    unittest.main()
