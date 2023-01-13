from torch.autograd import Variable

import torch.onnx
import torchvision.transforms as transforms
from torch import nn
import torchvision
import torch
from PIL import Image
import numpy as np

IMAGE_SIZE = 64

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



dummy_input = Variable(torch.randn(1, 3, 64, 64))
print("starting")
model = NeuralNetwork()
model.load_state_dict(torch.load('model.pth'))
model.eval()

im = Image.open("./train/hockey_puck/hockey_puck_730.jpg")
transform = transforms.Compose([transforms.Resize((64 * 3, 64 * 1)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

convert = transform(im)
print(convert)

result = model(convert)
print(result[0])
classes = {
    0: "American Football",
    1: "Baseball",
    2: "Basketball",
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
print(classes[torch.argmax(result[0]).item()])
classes[torch.argmax(result[0]).item()]

# torch.onnx.export(model, dummy_input, "moment-in-time.onnx")