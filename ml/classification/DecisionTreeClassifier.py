from ml.data_structures.Tree import TreeNode
from ml.utils.Sample import random_sampling

import random

import pandas as pd
from pandas.api.types import is_bool_dtype
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype

class Leaf(TreeNode):
    def __init__(self, value, parent):
        TreeNode.__init__(self, value=value, children=[], parent=parent)

class DecisionNode(TreeNode):
    def __init__(self, attribute, value=None, right=None, left=None, gini_value=1, parent=None):
        TreeNode.__init__(self, value=attribute, children=[left, right], parent=parent)
        self.attribute_value = value
        self.attribute = attribute
        self.gini_value = gini_value
    
    def get_gini_value(self):
        return self.gini_value
    
    def get_attribute_value(self):
        return self.attribute_value
    
    def get_attribute(self):
        return self.value
    
    def get_side(self):
        if self.is_root():
            return None
        else:
            if self == self.parent.get_children()[0]:
                return 'left'
            else:
                return 'right'

class DecisionTree(object):
    def __init__(self, max_depth=2, result_label='label', numerical_sampler=random_sampling(5)):
        self.attributes = None
        self.max_depth = max_depth
        self.tree = None
        self.result_label = result_label
        self.arg_property = dict()
        self.numerical_sampler = numerical_sampler
        
    def compute_gini_index(self, data, attribute, value, result_label='label'):
        left_data  = self._filter(data, attribute, value, 'left')
        right_data = self._filter(data, attribute, value, 'right')
        
        left_N = left_data.shape[0]
        left_n = left_data[left_data[result_label] == True].shape[0]
        
        try:
            left_gini = 1 - (left_n/left_N) ** 2 - ( (left_N-left_n)/left_N) ** 2
        except:
            left_gini = 1

        right_N = right_data.shape[0]
        right_n = right_data[right_data[result_label] == True].shape[0]
        
        try:
            right_gini = 1 - (right_n/right_N) ** 2 - ( (right_N-right_n)/right_N) ** 2
        except:
            right_gini = 1

        try:
            return left_N/(left_N+right_N) * left_gini + right_N/(left_N+right_N)*right_gini
        except:
            return 1
    
    def get_best_split(self, data, attributes):
        min_arg = None
        min_E   = 1
        
        for arg in attributes:
            if self.arg_property[arg]['type'] == 'categorical':
                for value in self.arg_property[arg]['classes']:
                    E = self.compute_gini_index(data, arg, value, self.result_label)
                    if E < min_E:
                        min_E = E
                        min_arg = arg
                        min_val = value
            elif self.arg_property[arg]['type'] == 'numerical':
                for value in self.numerical_sampler(data[arg]).values:
                    E = self.compute_gini_index(data, arg, value, self.result_label)
                    if E < min_E:
                        min_E = E
                        min_arg = arg
                        min_val = value
                         
        return min_arg, min_val, min_E 
    
    def _filter(self, data, attribute, val, side):
        if self.arg_property[attribute]['type'] == 'categorical':
            if side == 'left':
                return data[data[attribute] != val]
            else:
                return data[data[attribute] == val]
        elif self.arg_property[attribute]['type'] == 'numerical':
            if side == 'left':
                return data[data[attribute] < val]
            else:
                return data[data[attribute] >= val]
        else:
            return None
        
    def _check(self, node, value):
        if self.arg_property[node.get_attribute()]['type'] == 'categorical':
            if node.get_attribute_value() == value:
                return 'right'
            else:
                return 'left'
        elif self.arg_property[node.get_attribute()]['type'] == 'numerical':
            if value < node.get_attribute_value():
                return 'left'
            else:
                return 'right'
        else:
            return None
    
    def train(self, data, attributes):
        self.attributes = attributes
        self._compute_attributes_properties(data)
        
        self.tree = self._build_tree(data, attributes=set(self.attributes), parent=None, level=0)
        
    def _build_tree(self, data, attributes, parent=None, level=0):
        if len(attributes) != 0 and level < self.max_depth:
            min_arg, min_val, min_gini = self.get_best_split(data, list(attributes))
            node = DecisionNode(attribute=min_arg, value=min_val, gini_value=min_gini, parent=parent)
            
            data_left = self._filter(data, min_arg, min_val, 'left')
            data_right  = self._filter(data, min_arg, min_val, 'right')
            
            attributes = attributes - { min_arg }

            if data_left.shape[0] != 0 and data_right.shape[0] != 0:
                node.left  = self._build_tree(data_left, attributes=attributes, parent=node, level=level+1)
                node.right = self._build_tree(data_right,  attributes=attributes, parent=node, level=level+1)

                return node

        N = data.shape[0]
        n = data[data[self.result_label] == True].shape[0]

        node = Leaf({'True' : n/N, 'False' : (N-n)/N}, parent=parent)
        return node
    
    def _compute_attributes_properties(self, data):
        self.arg_property = dict()
        
        for attribute in self.attributes:
            if is_string_dtype(data[attribute]) or is_bool_dtype(data[attribute]):
                self.arg_property[attribute] = { 'type' : 'categorical' , 'classes' : data[attribute].unique() }
            else:
                self.arg_property[attribute] = { 'type' : 'numerical' }
                
    def predict(self, data):
        node = self.tree
        
        while not node.is_leaf():
            side = self._check(node, data[node.get_attribute()])
            if side == 'left':
                node = node.left
            elif side == 'right':
                node = node.right
            else:
                return None
            
        value = node.get_value()
        if value['True'] > value['False']:
            return True
        elif value['True'] < value['False']:
            return False
        else:
            return random.choice([True, False])
