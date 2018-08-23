from .Graph import Node

class TreeNode(Node):
    def __init__(self, value, children=[], parent=None):
        Node.__init__(self, value=value, neighbours=None)
        
        self.children = children
        self.parent = parent
        
    def get_children(self):
        return self.children
        
    def get_parent(self):
        return self.parent
    
    def is_root(self):
        return self.parent == None
    
    def is_leaf(self):
        return self.children == []
    
    def get_depth(self):
        if self.is_root():
            return 0
        else:
            return 1 + self.parent.get_depth()
