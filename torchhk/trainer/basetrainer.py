import torch
import torch.nn as nn

from .trainer import Trainer

######################### Important Attributes ###########################
# self.model : model
# self.device : device where model is
# self.optimizer : optimizer
# self.scheduler : scheduler (* Automatically Updated)
# self.max_epoch : total number of epochs
# self.max_iter : total number of iterations
# self.epoch : current epoch (* Automatically Updated)
# self.iter : current iter (* Automatically Updated)
# self.record_keys : items to record == items returned by do_iter
#########################################################################

class BaseTrainer(Trainer):
    def __init__(self, model, **kwargs):
        super(BaseTrainer, self).__init__("BaseTrainer", model, **kwargs)
        # Set Records (* Must be same as the items returned by do_iter)
        self.record_keys = ["Loss", "Acc"]
    
    # Override Do Iter
    def _do_iter(self, images, labels):
        X = images.to(self.device)
        Y = labels.to(self.device)

        pre = self.model(X)
        cost = nn.CrossEntropyLoss()(pre, Y)

        self.optimizer.zero_grad()
        cost.backward()
        self.optimizer.step()

        _, pre = torch.max(pre.data, 1)
        total = pre.size(0)
        correct = (pre == Y).sum()
        cost = cost.item()
        
        return cost, 100*float(correct)/total