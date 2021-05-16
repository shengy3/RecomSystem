import torch
import torch.nn.functional as F
import numpy as np


items = np.load("./metadata/rating.npy")

gender_T = F.one_hot(torch.tensor(self.gender[index]), num_classes = np.max(self.gender))
print(items.shape)
print(items[0])
#for i in range(items.shape[0]):
#    items[i] = items[i] % 1000
items_T = F.one_hot(torch.tensor(items))

print(items_T)
