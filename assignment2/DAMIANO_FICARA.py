'''
Assignment 2
Student: Damiano Ficara
Description: Image classification using CNN on CIFAR-10 dataset
'''
# Required packages
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.optim as optim
from math import floor
import copy
    
'''
Q6 - Basic CNN model implementation
'''
def out_dimensions(conv_layer, h_in, w_in):
    """Calculate output size after applying convolution"""
    h_out = floor((h_in + 2 * conv_layer.padding[0] - 
                conv_layer.dilation[0] * (conv_layer.kernel_size[0] - 1) - 1) / 
                conv_layer.stride[0] + 1)
    w_out = floor((w_in + 2 * conv_layer.padding[1] - 
                conv_layer.dilation[1] * (conv_layer.kernel_size[1] - 1) - 1) / 
                conv_layer.stride[1] + 1)
    return h_out, w_out

class CNNBasic(nn.Module):
    """Basic CNN for image classification"""
    def __init__(self):
        super(CNNBasic, self).__init__()
        
        # First block: 2 conv layers + pooling
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), padding=0, stride=1)
        h_out, w_out = out_dimensions(self.conv1, 32, 32)
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=0, stride=1)
        h_out, w_out = out_dimensions(self.conv2, h_out, w_out)
        
        self.pool1 = nn.MaxPool2d(2, 2)
        h_out, w_out = int(h_out/2), int(w_out/2)
        
        # Second block: 2 conv layers + pooling
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=0, stride=1)
        h_out, w_out = out_dimensions(self.conv3, h_out, w_out)
        
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=0, stride=1)
        h_out, w_out = out_dimensions(self.conv4, h_out, w_out)
        
        self.pool2 = nn.MaxPool2d(2, 2)
        h_out, w_out = int(h_out/2), int(w_out/2)
        
        # Fully connected layers for classification
        self.fc1 = nn.Linear(64 * h_out * w_out, 256)
        self.fc2 = nn.Linear(256,64)
        self.fc3 = nn.Linear(64, 10)
        
        # Save dimensions for reshape in forward pass
        self.dimensions_final = (64, h_out, w_out)
    
    def forward(self, x):
        # Process first block
        x = self.conv1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool1(x)

        # Process second block
        x = self.conv3(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.pool2(x)

        # Flatten and pass through FC layers
        n_channels, h, w = self.dimensions_final
        x = x.view(-1, n_channels * h * w)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
'''
Q9 - Advanced model implementations
'''
class CNNPro(nn.Module):
    """
    CNN for image classification with 2 conv blocks and 3 fully connected layers
    Input: 32x32 RGB images
    Output: 10 classes
    """
    def __init__(self):
        super(CNNPro, self).__init__()
        # First conv block: 3->64 channels
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), padding=1, stride=1)
        self.bn1 = nn.BatchNorm2d(64)
        h_out, w_out = out_dimensions(self.conv1, 32, 32)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(64)
        h_out, w_out = out_dimensions(self.conv2, h_out, w_out)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.1)  
        h_out, w_out = int(h_out / 2), int(w_out / 2)

        # Second conv block: 64->128 channels
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1, stride=1)
        self.bn3 = nn.BatchNorm2d(128)
        h_out, w_out = out_dimensions(self.conv3, h_out, w_out)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1, stride=1)
        self.bn4 = nn.BatchNorm2d(128)
        h_out, w_out = out_dimensions(self.conv4, h_out, w_out)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout(0.1)
        h_out, w_out = int(h_out / 2), int(w_out / 2)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * h_out * w_out, 512)
        self.bn5 = nn.BatchNorm1d(512)
        self.dropout3 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 128)
        self.bn6 = nn.BatchNorm1d(128)
        self.dropout4 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(128, 10)

        self.dimensions_final = (128, h_out, w_out)

    def forward(self, x):
        # First conv block
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.gelu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.gelu(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        # Second conv block
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.gelu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.gelu(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        # FC layers
        n_channels, h, w = self.dimensions_final
        x = x.view(-1, n_channels * h * w)
        x = self.fc1(x)
        x = self.bn5(x)
        x = F.gelu(x)
        x = self.dropout3(x)
        x = self.fc2(x)
        x = self.bn6(x)
        x = F.gelu(x)
        x = self.dropout4(x)
        x = self.fc3(x)
        return x
    
if __name__ == "__main__":
    # Setup for reproducibility
    manual_seed = 42
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed(manual_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.benchmark = False

    '''
    Q2 - Data preparation
    '''
    # Setup data transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0, 0, 0), (1, 1, 1))])

    batch_size = 32

    # Load CIFAR-10 datasets
    dataset_train = datasets.CIFAR10(root='assignment2/data', train=True,download=True, transform=transform)
    dataset_test = datasets.CIFAR10(root='assignment2/data', train=False,download=True, transform=transform)

    # Create data loaders
    trainloader = DataLoader(dataset_train, batch_size=batch_size,shuffle=True, num_workers=2)
    testloader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=2)

    # Define class names
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Function to display images
    def imshow(img):
        npimg = img.numpy()
        plt.figure(figsize=(15, 15))
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.axis('off')

    # Get sample images
    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    # Display one example per class
    plt.figure(figsize=(8, 4))
    class_samples = {}
    for img, label in zip(images, labels):
        label_name = classes[label]
        if label_name not in class_samples and len(class_samples) < 10:
            class_samples[label_name] = img

    # Create visualization grid
    for idx, (class_name, img) in enumerate(class_samples.items(), 1):
        plt.subplot(2, 5, idx)
        img = img.numpy().transpose((1, 2, 0))
        plt.imshow(img)
        plt.title(f'{class_name}')
        plt.axis('off')
    plt.suptitle('One Sample per class', fontsize=14, y=1.0)
    plt.tight_layout()
    plt.savefig('img1.png')
    plt.show()

    # Count class distribution
    train_dist = [0] * len(classes)
    test_dist = [0] * len(classes)

    for data, labels in trainloader:
        for label in labels:
            train_dist[label.item()] += 1

    for data, labels in testloader:
        for label in labels:
            test_dist[label.item()] += 1

    # Plot distribution
    plt.figure(figsize=(12, 6))
    x = np.arange(len(classes))
    width = 0.35

    plt.bar(x - width / 2, train_dist, width, label='Training Set', color='#2ecc71', alpha=0.8, edgecolor='black')
    plt.bar(x + width / 2, test_dist, width, label='Test Set', color='#3498db', alpha=0.8, edgecolor='black')

    plt.xlabel('Classes', fontsize=12)
    plt.ylabel('Images Number', fontsize=12)
    plt.title('Distribution of Images in Training and Test Sets', fontsize=14)
    plt.xticks(x, classes, rotation=45)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    # Add count labels on bars
    for i in range(len(x)):
        plt.text(x[i] - width / 2, train_dist[i], str(train_dist[i]), ha='center', va='bottom')
        plt.text(x[i] + width / 2, test_dist[i], str(test_dist[i]), ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('img2.png')
    plt.show()

    '''
    Q3 - Verify data shapes
    '''
    print(f"Image shape: {images[0].shape}")
    print(f"Label: {labels[0]}")

    '''
    Q5 - Split test data
    '''
    # Create validation set
    dataset_validation, dataset_test = torch.utils.data.random_split(dataset_test, [0.5, 0.5])
    validloader = DataLoader(dataset_validation, batch_size=batch_size)
    testloader = DataLoader(dataset_test, batch_size=batch_size)

    # Print dataset sizes
    total_train_batches = len(trainloader)
    print(f"Number of batches per epoch (training): {total_train_batches}")

    total_valid_batches = len(validloader)
    print(f"Number of batches per epoch (validation): {total_valid_batches}")

    '''
    Q7 - Basic model training
    '''
    # Setup model and training parameters
    model = CNNBasic() 
    learning_rate = 0.034
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    # Set device (GPU/CPU)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' 
        if torch.backends.mps.is_available() else 'cpu')
    model = model.to(DEVICE)
    print("Working on", DEVICE)

    # Training settings
    n_epochs = 4
    n_step_train = 100  
    n_steps_valid = 50 
    train_loss_list = []
    validation_loss_list = []
    train_accuracy_list = []
    validation_accuracy_list = []

    # Training loop
    for epoch in range(n_epochs):
        loss_train = 0
        correct_train = 0
        total_train = 0
        
        # Training phase
        for step, (data, target) in enumerate(trainloader, 1):
            model.train()
            data, target = data.to(DEVICE), target.to(DEVICE)
            
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss_train += loss.item()
            
            loss.backward()
            optimizer.step()
            
            # Calculate accuracy
            _, predicted = torch.max(output.data, 1)
            total_train += target.size(0)
            correct_train += (predicted == target).sum().item()
            
            # Print progress
            if step % n_step_train == 0:
                avg_train_loss = loss_train / step
                train_accuracy = 100 * correct_train / total_train
                print(f"Epoch {epoch + 1}, Step {step}: Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
                if step == 1500:
                    train_loss_list.append(avg_train_loss)
                    train_accuracy_list.append(train_accuracy)
        
        # Validation phase
        model.eval()
        loss_valid = 0
        correct_valid = 0
        total_valid = 0
        with torch.no_grad():
            for step, (data, target) in enumerate(validloader, 1):
                data, target = data.to(DEVICE), target.to(DEVICE)
                output = model(data)
                loss = loss_fn(output, target)
                loss_valid += loss.item()
                
                # Calculate validation accuracy
                _, predicted = torch.max(output.data, 1)
                total_valid += target.size(0)
                correct_valid += (predicted == target).sum().item()
                
                if step % n_steps_valid == 0:
                    avg_valid_loss = loss_valid / step
                    valid_accuracy = 100 * correct_valid / total_valid
                    print(f"Epoch {epoch + 1}, Step {step}: Validation Loss: {avg_valid_loss:.4f}, Validation Accuracy: {valid_accuracy:.2f}%")
                    if step == 150:
                        validation_loss_list.append(avg_valid_loss)
                        validation_accuracy_list.append(valid_accuracy)

    # Final test evaluation
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for data, target in testloader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            n_samples += target.size(0)
            n_correct += (predicted == target).sum().item()

        acc = 100.0 * n_correct / n_samples
    print("Accuracy on the test set:", acc, "%")

    '''
    Q8 - Plot training results
    '''
    plt.figure()
    plt.plot(range(n_epochs), train_loss_list)
    plt.plot(range(n_epochs), validation_loss_list)
    plt.legend(["Train loss", "Validation Loss"])
    plt.xlabel("Epochs")
    plt.ylabel("Loss value")
    plt.savefig('img3.png')
    plt.show()

    '''
    Q9 - Advanced model training
    '''
    # Setup advanced model
    model = CNNPro()
    learning_rate = 0.031
    optimizer = optim.SGD(model.parameters(), lr=learning_rate,weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    model = model.to(DEVICE)
    print("Working on", DEVICE)

    train_loss_list = []
    validation_loss_list = []
    n_epochs = 7

    # Training loop with early stopping
    for epoch in range(n_epochs):
        loss_train = 0
        for data, target in trainloader:
            model.train()
            data, target = data.to(DEVICE), target.to(DEVICE)
            
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss_train += loss.item()
            
            loss.backward()
            optimizer.step()
            
        loss_train = loss_train / len(trainloader)
        train_loss_list.append(loss_train)
        
        # Validation check
        with torch.no_grad():
            model.eval()
            for data, target in validloader: 
                data, target = data.to(DEVICE), target.to(DEVICE)
                output = model(data)
                validation_loss = loss_fn(output, target).item()
            print(f"Epoch {epoch + 1}: Train loss: {loss_train}, Validation loss {validation_loss}")
            validation_loss_list.append(validation_loss)
            
    
    # Final evaluation
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for data, target in testloader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            n_samples += target.size(0)
            n_correct += (predicted == target).sum().item()

        acc = 100.0 * n_correct / n_samples
    print("Accuracy on the test set:", acc, "%")

    # Plot final results
    plt.figure()
    plt.plot(range(len(train_loss_list)), train_loss_list)
    plt.plot(range(len(validation_loss_list)), validation_loss_list)
    plt.legend(["Train loss", "Validation Loss"])
    plt.xlabel("Epochs")
    plt.ylabel("Loss value")
    plt.savefig('img4.png')
    plt.show()

    '''
    Q10 - Multiple seed testing
    '''
    test_accuracies = []
    # Test different random seeds
    for seed in range(5, 10):
        torch.manual_seed(seed)
        print("\nSeed equal to", torch.random.initial_seed())
        
        # Initialize model
        model = CNNBasic()
        model = model.to(DEVICE)
        learning_rate = 0.034
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        loss_fn = nn.CrossEntropyLoss()
        
        # Training
        n_epochs = 4
        train_loss_list = []
        validation_loss_list = []
        
        for epoch in range(n_epochs):
            loss_train = 0
            for data, target in trainloader:
                model.train()
                data, target = data.to(DEVICE), target.to(DEVICE)
                optimizer.zero_grad()
                output = model(data)
                loss = loss_fn(output, target)
                loss_train += loss.item()
                loss.backward()
                optimizer.step()
                
            loss_train = loss_train / len(trainloader)
            train_loss_list.append(loss_train)
            
            # Validation
            with torch.no_grad():
                model.eval()
                for data, target in validloader:
                    data, target = data.to(DEVICE), target.to(DEVICE)
                    output = model(data)
                    validation_loss = loss_fn(output, target).item()
                print(f"Epoch {epoch + 1}: Train loss: {loss_train:.4f}, "
                        f"Validation loss: {validation_loss:.4f}")
                validation_loss_list.append(validation_loss)
        
        # Test evaluation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in testloader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        test_accuracy = 100 * correct / total
        test_accuracies.append(test_accuracy)
        print(f"Final Test Accuracy for seed {seed}: {test_accuracy:.2f}%")