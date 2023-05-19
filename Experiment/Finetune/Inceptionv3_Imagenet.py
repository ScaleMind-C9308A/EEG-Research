import torch
import torchvision
import torchvision.transforms as transforms

# Load the pre-trained Inception V3 model
model = torchvision.models.inception_v3(pretrained=True)

# Freeze the parameters of the pre-trained model
for param in model.parameters():
    param.requires_grad = False

# Replace the last layer of the model with a new layer with the desired number of classes
num_classes = 40
model.fc = torch.nn.Linear(2048, num_classes)

# Define the loss function and the optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# Load the ImageNet dataset
train_dataset = torchvision.datasets.ImageNet(root='/home/exx/GithubClonedRepo/EEG-Research/Dataset/imagenet', train=True, transform=transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]))

test_dataset = torchvision.datasets.ImageNet(root='/home/exx/GithubClonedRepo/EEG-Research/Dataset/imagenet', train=False, transform=transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]))

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# Train the model
for epoch in range(50):
    for i, (images, labels) in enumerate(train_loader):
        # Forward pass
        outputs = model(images)

        # Calculate the loss
        loss = criterion(outputs, labels)

        # Backpropagate the loss
        optimizer.zero_grad()
        loss.backward()

        # Update the parameters
        optimizer.step()

        # Print the loss
        if i % 100 == 0:
            print('Epoch {} Loss: {}'.format(epoch + 1, loss.item()))

# Evaluate the model
test_loss = 0.0
correct = 0
for images, labels in test_loader:
    outputs = model(images)
    loss = criterion(outputs, labels)
    test_loss += loss.item()
    correct += (outputs.argmax(1) == labels).sum()

test_accuracy = 100.0 * correct / len(test_dataset)
print('Test Loss: {} Test Accuracy: {}'.format(test_loss, test_accuracy))