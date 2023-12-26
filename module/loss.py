class CrossEntropyLoss():
    def __init__(self):
        pass

    def __call__(self, input, target):
        tol = 1e-6
        sum = 0
        for y_pred, y_true in zip(input, target):
            for i in range(len(y_pred)):
                sum += (y_pred[i] + tol).log() * y_true[i]
        loss = sum / len(input)
        # loss = -np.mean(np.sum(np.log(input + tol) * target,axis=1))
        return -loss
    
class MSE:
    def __init__(self) -> None:
        pass

    def __call__(self, input, target):
        loss = sum((y_in - y_target)**2 for y_in, y_target in zip(input, target))
        return loss
    
class BinaryCrossEntropyLoss():
    def __init__(self):
        pass
    
    # TODO: Simplify it
    def __call__(self, input, target):
        result = 0
        for y_pred, y_true in zip(input, target):
            if y_true == -1:
                a = (1-y_pred)
                result += a.log()
            elif y_true == 1:
                result += y_pred.log()
        result = -(result / len(input))
        return result