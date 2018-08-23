# Copyright, David Szabo 2018
# MIT Licence

class Node(object):
    def __init__(self, value, neighbours=[]):
        self.value = value
        self.neighbours = neighbours
        
    def get_value(self):
        return self.value
    
    def get_neighbours(self):
        return self.neighbours
