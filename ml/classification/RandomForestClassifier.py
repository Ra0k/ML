import math
import random
import pandas as pd

from ml.classification.DecisionTreeClassifier import DecisionTree
from ml.validation.split_data import train_test_split
from ml.validation.accuracy import accuracy

class RandomForestClassifier(object):
    def __init__(self, tree_number=15, result_label='label', max_depth=10):
        self.trees = [ DecisionTree(max_depth=max_depth, result_label=result_label) for _ in range(tree_number)]
        self.max_depth = max_depth
        
    def train(self, data, attributes, alpha=0.2, beta=0.5):
        attributes_number = len(attributes)
        row_number = data.shape[0]
        
        for tree in self.trees:
            sampled_data = data.sample(n=int(alpha*row_number))
            sampled_attr = random.sample(attributes.tolist(), int(beta*attributes_number))
            tree.train(data=sampled_data, attributes=sampled_attr)
            
    def predict(self, data):
        vote_true, vote_false = 0, 0
        for tree in self.trees:
            if tree.predict(data):
                vote_true += 1
            else:
                vote_false += 1
                
        if vote_true > vote_false:
            return True
        elif vote_true < vote_false:
            return False
        else:
            return random.choice([True, False])
