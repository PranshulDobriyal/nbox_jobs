from nbox import Operator

class Add(Operator):
    def __init__(self):
        super().__init__()
        self.sum = 0
    
    def forward(self, a, b):
        self.sum = a + b
        print("The sum is ", self.sum)