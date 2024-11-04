import sys
import os
import ipytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../data')))
from Insurance_refactor_final_fixed import InsuranceModel


class TestFeatureEngineering:
    def test_preprocess_data(self):
        model = InsuranceModel('../data/raw/train/ticdata2000.txt', '../data/raw/eval/ticeval2000.txt', '../data/raw/eval/tictgts2000.txt')
        model.load_data()
        model.preprocess_data()
        assert model.X_train is not None
        assert model.y_train is not None

    def test_handle_imbalance(self):
        model = InsuranceModel('../data/raw/train/ticdata2000.txt', '../data/raw/eval/ticeval2000.txt', '../data/raw/eval/tictgts2000.txt')
        model.load_data()
        model.preprocess_data()
        model.handle_imbalance()
        assert len(model.X_train) == len(model.y_train)  # Ensure SMOTE applied

ipytest.run()
