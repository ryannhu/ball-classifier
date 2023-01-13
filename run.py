
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision
from torchvision import datasets
from torchvision.utils import make_grid
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np
# from torchvision import transforms
from PIL import Image
import random


if __name__ == '__main__':
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    import torchvision.transforms as transforms
    import torchvision
    from torchvision import datasets
    from torchvision.utils import make_grid
    from torchvision.utils import save_image
    import matplotlib.pyplot as plt
    import numpy as np
    # from torchvision import transforms
    from PIL import Image
    import random


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMAGE_SIZE = 64
TEST_DIR = "./test"
TRAIN_DIR = "./train"

# transform = transforms.CenterCrop(250)
transform = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=transform)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=1)


test_dataset = datasets.ImageFolder(root=TEST_DIR, transform=transform)
test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=1)

def show_transformed_images (dataset):
    loader = torch.utils.data.DataLoader(dataset, batch_size = 6, shuffle=True)
    batch = next(iter(loader))
    images, labels = batch

    print(images)

    grid = torchvision.utils.make_grid( images, nrow=3)
    plt.figure(figsize=(11,11))
    plt.imshow(np.transpose(grid, (1,2,0)))
    print('labels: ', labels)
    plt.show()

# show_transformed_images(dataset)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(IMAGE_SIZE*IMAGE_SIZE * 3, 3*64),
            nn.ReLU(),
            nn.Linear(3*64, 256),
            nn.ReLU(),
            nn.Linear(256, 15),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork()
model.to(device)
classes = {
    0: "American Football",
    1: "Baseball",
    2: "Basektball",
    3: "Billard Ball",
    4: "Bowling Ball",
    5: "Cricket Ball",
    6: "Football",
    7: "Golf Ball",
    8: "Hockey Ball",
    9: "Hockey Buck",
    10: "Rugby Ball",
    11: "Shuttlecock",
    12: "Table Tennis Ball",
    13: "Tennis Ball",
    14: "Volleyball"
    
}

learning_rate = 1e-3
batch_size = 64
epochs = 5

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backproagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}")

def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# model = torch.load('ball.pth')
# model.eval()  
epochs = 15

# print('Training')
# for t in range(epochs):
#     print(f"Epoch {t+1}\n-------------------------------")
#     train_loop(train_dataloader, model, loss_fn, optimizer)
#     test_loop(test_dataloader, model, loss_fn)
# print("Done!")
# print(torch.cuda.is_available())

# state = {
#     'epoch': epochs,
#     'state_dict': model.state_dict(),
#     'optimizer': optimizer.state_dict(),
# }
# torch.save(state, 'ball.pth')
# torch.save(model.module.state_dict(), 'ball.pth')
#Function to Convert to ONNX 
def convert(): 

    # set the model to inference mode 
    model.eval() 

    # Let's create a dummy input tensor  
    dummy_input = torch.randn(1, 3, 32, 32, requires_grad=True)  

    # Export the model   
    torch.onnx.export(model,         # model being run 
         dummy_input,       # model input (or a tuple for multiple inputs) 
         "Network.onnx",       # where to save the model  
         export_params=True,  # store the trained parameter weights inside the model file 
         opset_version=11,    # the ONNX version to export the model to 
         do_constant_folding=True,  # whether to execute constant folding for optimization 
         input_names = ['input'],   # the model's input names 
         output_names = ['output'], # the model's output names 
         dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes 
                                'output' : {0 : 'batch_size'}}) 
    print(" ") 
    print('Model has been converted to ONNX')


model = torch.load('ball.pth')


model.eval()


img = Image.open('POOP.jpg')


out = model(img)

print(out)



