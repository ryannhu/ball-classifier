import torch
from torchvision import datasets, transforms

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Load the dataset
dataset = datasets.ImageFolder(root='./train', transform=transform)

# Create the data loader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)

# Get a batch of training data
images, labels = next(iter(dataloader))

# Check the size of the batch
print(images.size())  # (batch_size, 3, height, width)
print(labels.size())  # (batch_size)
