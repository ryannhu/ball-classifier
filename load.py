import torch
from torchvision.transforms import transforms
from PIL import Image
from cnn_main import CNNet
from pathlib import Path

model = CNNet(5)
checkpoint = torch.load(Path('./model.pth'))
model.load_state_dict(checkpoint)

trans = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Resize(32),
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
    ])

image = Image.open(Path('./POOP.jpg'))

input = trans(image)

input = input.view(1, 3, 32,32)

output = model(input)

prediction = int(torch.max(output.data, 1)[1].numpy())
print(prediction)

if (prediction == 0):
    print ('daisy')
if (prediction == 1):
    print ('dandelion')
if (prediction == 2):
    print ('rose')
if (prediction == 3):
    print ('sunflower')
if (prediction == 4):
    print ('tulip')