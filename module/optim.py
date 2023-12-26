class SGD:
    def __init__(self, parameters, lr=0.01) -> None:
        self.parameters = parameters
        self.lr = lr
    
    def zero_grad(self):
        for p in self.parameters:
            p.grad = 0

    def step(self):
        for p in self.parameters:
            p.data += (-1*self.lr) * p.grad