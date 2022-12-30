import numpy as np
import torch
from sklearn import metrics as skmetrics

class MultiClassMetrics():
    def __init__(self):
        self.tgt = []
        self.prd = []
        self.nnloss = []
    def reset(self):
        self.__init__()

    def add_entry(self, prd, tgt, loss=0):
        self.prd.extend(prd.cpu().detach().numpy())
        self.tgt.extend(tgt.cpu())
        if loss: self.nnloss.append(loss.cpu().detach().numpy())

    def get_loss(self):
        return sum(self.nnloss) / len(self.nnloss)

    def get_accuracy(self):
        return skmetrics.accuracy_score(self.tgt, self.prd)

    def get_class_report(self):
        return skmetrics.classification_report(self.tgt, self.prd, output_dict= True)

    def get_stat_summary(self):
        pass


if __name__ == "__main__":

    obj = MultiClassMetrics()
    obj.tgt = [1,1,1,2,2,2,3,3,3,4,4,4,5,5,5]
    obj.prd = [1,1,2,2,2,3,3,3,4,4,4,5,5,5,1]

    print(obj.get_class_report())