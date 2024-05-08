# Source code for confusion matrix.
#
# Author: Matt Krol

import torch

torch.manual_seed(0)
torch.cuda.manual_seed(0)


class ConfusionMatrix(object):
    def __init__(self, device, classes=100):
        self.matrix = torch.zeros((classes, classes), dtype=torch.int64,
                                  device=device)


    def add(self, pred, target):
        assert torch.numel(pred) == torch.numel(target)

        for i in range(torch.numel(pred)):
            self.matrix[target[i].item(), pred[i].item()] += 1


    def accuracy(self):
        return torch.sum(torch.diag(self.matrix)).item()/torch.sum(self.matrix).item()


    def tpr(self):
        return torch.div(torch.diag(self.matrix),
                         torch.sum(self.matrix, 1))


    def precision(self):
        return torch.div(torch.diag(self.matrix),
                         torch.sum(self.matrix, 0))


    def get(self):
        return self.matrix.cpu().numpy()
