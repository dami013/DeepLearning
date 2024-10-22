'''
Assignment 2
Student: NAME SURNAME
'''

# *** Packages ***
import torch

def out_dimensions(conv_layer, h_in, w_in):
    '''
    This function computes the output dimension of each convolutional layers in the most general way. 
    '''
    h_out = floor((h_in + 2 * conv_layer.padding[0] - conv_layer.dilation[0] * (conv_layer.kernel_size[0] - 1) - 1) /
                  conv_layer.stride[0] + 1)
    w_out = floor((w_in + 2 * conv_layer.padding[1] - conv_layer.dilation[1] * (conv_layer.kernel_size[1] - 1) - 1) /
                  conv_layer.stride[1] + 1)
    return h_out, w_out
    
# You can start by modifyng what we have done in class, or define your model from scratch
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3)) # Is in_channels = 1 what you want?
        h_out, w_out = out_dimensions(self.conv1, 28, 28) # Is 28 what you want?
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3)) 
        h_out, w_out = out_dimensions(self.conv2, h_out, w_out)
        self.pool1 = nn.MaxPool2d(2, 2)
        h_out, w_out = int(h_out/2), int(w_out/2)
        # You can double this block! 
        self.fc1 = nn.Linear(32 * h_out * w_out, 10) # What does 32 represent?
        # You can add one fully connected layer. What do you have to change?
        self.dimensions_final = (32, h_out, w_out)
​
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool1(x)
        n_channels, h, w = self.dimensions_final
        x = x.view(-1, n_channels * h * w)
        x = self.fc1(x)
        return x

if __name__ == "__main__":
    # Write your code here
    print("Hello World!")

    '''
    DON'T MODIFY THE SEED!
    '''
    # Set the seed for reproducibility
    manual_seed = 42
    torch.manual_seed(manual_seed)

    '''
    Q1 - Code
    '''
    pass

    
    '''
    Q2 - Code
    '''
    
    pass


    '''
    ........
    '''


    '''
    Q10 -  Code
    '''
    for seed in range(5,10):
        torch.manual_seed(seed)
        print("Seed equal to ", torch.random.initial_seed())
        # Train the models here
