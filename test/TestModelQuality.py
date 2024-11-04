import sys
import os
import ipytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../data')))
from Insurance_refactor_final_fixed import InsuranceModel

class TestModelQuality:
    def test_model_performance(self):
        model = InsuranceModel('../data/raw/train/ticdata2000.txt', '../data/raw/eval/ticeval2000.txt', '../data/raw/eval/tictgts2000.txt')
        model.load_data()
        model.preprocess_data()
        model.handle_imbalance()
        model.train_model()
        
        predictions = model.model.predict(model.X_test)
        accuracy = model.model.score(model.X_test, model.y_test)
        assert accuracy > 0.7  # Example threshold, adjust as needed

ipytest.run()
