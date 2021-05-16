import numpy as np
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F

class RSDataset(Dataset):
    def __init__(self):
        self.gender = np.load("./metadata/gender.npy")
        self.item = np.load("./metadata/item.npy")
        self.user = np.load("./metadata/user.npy")
        self.category = np.load("./metadata/category.npy")
        self.rating = np.load("./metadata/rating.npy")

        print("Dataset size: ", self.gender.shape[0])


    def __getitem__(self, index):
        return (self.gender[index], self.item[index], self.user[index], self.category[index], self.rating[index])

    def __len__(self):
        #return 100 # len(self.gender.shape[0])
        return self.gender.shape[0]
