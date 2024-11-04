import sys
import os
import ipytest
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../data')))
from Insurance_refactor import InsuranceModel


class TestDataLoading:
    def test_load_data_success(self):
        model = InsuranceModel('../data/raw/train/ticdata2000.txt', '../data/raw/eval/ticeval2000.txt', '../data/raw/eval/tictgts2000.txt')
        model.load_data()
        assert model.train_data is not None
        assert model.eval_data is not None
        assert model.target_data is not None

    def test_load_data_missing_file(self):
        model = InsuranceModel('invalid/path/train.txt', 'invalid/path/eval.txt', 'invalid/path/target.txt')
        with pytest.raises(FileNotFoundError):
            model.load_data()

ipytest.run()
