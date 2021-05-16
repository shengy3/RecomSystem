import torch
from torch import nn

class RS(nn.Module):
    def __init__(self, reduced_dim = True):
        super(RS, self).__init__()

        if reduced_dim:
            item_size = 1000
            user_size = 1000
        else:
            item_size = 9560
            user_size = 1157633
        if reduced_dim:
            print("You use the model with reduced dimensions")

        self.item_embedding = nn.Sequential(
        nn.Embedding(item_size, 200),
        nn.Linear(200, 200),
        nn.ReLU(),
        nn.Linear(200, 100),
        nn.ReLU(),
        nn.Linear(100, 50),
        nn.ReLU()
        )

        self.user_embedding =  nn.Sequential(
        nn.Embedding(user_size, 200),
        nn.Linear(200, 200),
        nn.ReLU(),
        nn.Linear(200, 100),
        nn.ReLU(),
        nn.Linear(100, 50),
        nn.ReLU()
        )

        self.gender_embedding =  nn.Sequential(
        nn.Embedding(3, 10),
        nn.Linear(10, 200),
        nn.ReLU(),
        nn.Linear(200, 100),
        nn.ReLU(),
        nn.Linear(100, 50),
        nn.ReLU()
        )

        self.category_embedding =  nn.Sequential(
        nn.Embedding(10, 10),
        nn.Linear(10, 200),
        nn.ReLU(),
        nn.Linear(200, 100),
        nn.ReLU(),
        nn.Linear(100, 50),
        nn.ReLU()
        )

        self.output_fc = nn.Sequential(
        nn.Linear(200, 200),
        nn.ReLU(),
        nn.Linear(200, 100),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(100, 6),
        )


    def forward(self, gender, item, user, category):
        x_item = self.item_embedding(item)
        x_user = self.user_embedding(user)
        x_gender = self.gender_embedding(gender)
        x_category = self.category_embedding(category)
        #print("---------------------size---------------------")

        #print("x_item", x_item.size(), "x_user", x_user.size(),"gender", x_gender.size(), "category", x_category.size())

        x = torch.cat((x_item, x_user, x_user, x_category), dim = -1)
        #print("Total:", x.size())
        x = self.output_fc(x)
        #print("Output:", x.size())
        #print("---------------------size---------------------")

        return x
