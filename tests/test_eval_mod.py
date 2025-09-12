import unittest

from pathlib import Path

import pandas as pd

from auto_feat.eval_module.evaluator import create_evaluation_agent_wrap
from auto_feat import AutoFeaturizer
from auto_feat.eval_module.evaluator import create_evaluation_agent_wrap

class TestEvalModule(unittest.TestCase):

    def test_run(self):
        mpea_dir = Path.home() / "Box/LLM hackathon 2025 Featurizers/datasets/MPEA/"
        paper = str(mpea_dir.absolute()) + '/manuscript.txt'
        data = pd.read_csv(str(mpea_dir.absolute()) + '/data.csv')

        state = AutoFeaturizer(papers=[paper],
                                data=data,
                                target='OUTPUT PROPERTY: Exp. Young modulus (GPa)')
        
        dummy = create_evaluation_agent_wrap()
        success = dummy(state)
        print(success.eval_report)
        self.assertTrue(len(state.eval_report) == 1)
        

if __name__ == "__main__":
    unittest.main()      

