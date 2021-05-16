import torch
from torch import nn
from torch.utils.data import DataLoader
from CustomerData import RSDataset
from torchvision import datasets
import matplotlib.pyplot as plt
from RecomSysModel import RS


debug = False
batch_size = 64
epochs = 5

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

train_dataloader = DataLoader(RSDataset(), batch_size = batch_size)
#test_dataloader = DataLoader(test_data, batch_size = batch_size)




model = RS().to(device)
#model = NeuralNetwork().to(device)
print(model)


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 1e-3)

def stop_layer_grad(layer_name):
    for param in model.named_parameters():
        if param[0] in layer_name:
            param[1].requires_grad = False

def stop_module_grad(module):
    for param in module.parameters():
        param.requires_grad = False


def train(dataloader, model, loss_fn, optimizer):
  size = len(dataloader.dataset)
  for batch, (gender, item, user, category, rating)  in enumerate(dataloader):

    x_g, x_i, x_u, x_c, y_r = gender.to(device), item.to(device), user.to(device), category.to(device), rating.to(device)

    if debug:
        print("x_g", x_g.size(), "x_i", x_i.size(),"x_u", x_u.size(), "x_c", x_c.size(), "y_u", y_r.size())
    # Compute prediction error
    pred = model(x_g, x_i, x_u, x_c)
    loss = loss_fn(pred, y_r)

    # backpropagate
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if batch % 100 == 0:
      loss, current = loss.item(), batch * len(x_g)
      print(f"loss {loss:>7f}, [{current:>5d}/{size:>5d}]")
      if debug:
          return


def test(dataloader, model):
  size = len(dataloader.dataset)
  model.eval()
  test_loss, correct = 0, 0

  print("Start inferring")

  with torch.no_grad():
    for batch, (gender, item, user, category, rating)  in enumerate(dataloader):

      x_g, x_i, x_u, x_c, y_r = gender.to(device), item.to(device), user.to(device), category.to(device), rating.to(device)

      pred = model(x_g, x_i, x_u, x_c)

      test_loss += loss_fn(pred, y_r).item()
      correct += (pred.argmax(1) == y_r).type(torch.int).sum().item()
      if batch % 1000 == 0:
        current =  batch * len(x_g)
        print(f"[{current:>5d}/{size:>5d}]")

  test_loss /= size
  correct /= size
  print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


for t in range(epochs):
  print(f"Epoch {t+1} \n -----------------------------")
  train(train_dataloader, model, loss_fn, optimizer)
  test(train_dataloader, model)
