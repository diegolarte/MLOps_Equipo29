import sys
import os
import ipytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../data')))
from Insurance_refactor_final_fixed import InsuranceModel

class TestModelSettings:
    def test_save_model(self):
        model = InsuranceModel('../data/raw/train/ticdata2000.txt', '../data/raw/eval/ticeval2000.txt', '../data/raw/eval/tictgts2000.txt')
        model.load_data()
        model.preprocess_data()
        model.handle_imbalance()
        model.train_model()
        model.save_model()
        
        model_path = '../models/insurance_model.pkl'
        assert os.path.exists(model_path)
        
ipytest.run()
