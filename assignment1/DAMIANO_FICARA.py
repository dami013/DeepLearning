'''
Template for Assignment 1
'''

import numpy as np 
import matplotlib.pyplot as plt 
import torch 
import torch.nn as nn
import torch.optim as optim
import os
'''
Code for Q2
'''
def plot_polynomial(coeffs, z_range, color='b'):
    z_min, z_max = z_range
    z = np.linspace(z_min, z_max)

    y = np.polyval(coeffs[::-1], z)
    
    plt.figure(figsize=(10, 6))
    plt.plot(z, y, color=color)
    plt.xlabel('z')
    plt.ylabel('p(z)')
    plt.title('Polynomial Function')
    plt.grid(True)
    plt.savefig("Q2.png")
    plt.show()
'''
Code for Q3:
'''    
def create_dataset(coeffs, z_range, sample_size, sigma, seed=42):
    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    # Unpack z_range
    z_min, z_max = z_range
    # Generate evenly spaced z values
    z = torch.linspace(z_min, z_max, sample_size)
    # Create the design matrix X
    X = torch.stack([z**i for i in range(len(coeffs))]).T
    # Calculate y_hat (noiseless y)
    y_hat = X @ torch.tensor(coeffs, dtype=torch.float32)
    # Add Gaussian noise to get y
    noise = torch.normal(torch.zeros(sample_size), sigma*torch.ones(sample_size))
    y = y_hat + noise
    return X, y

def visualize_data(X, y, coeffs, z_range, title):
    z_min, z_max = z_range
    z = np.linspace(z_min, z_max)
    y_true = np.polyval(coeffs[::-1], z)

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot true polynomial
    ax.plot(z, y_true, color='r', label='True Polynomial', linewidth=2)
    ax.scatter(X[:, 1], y, alpha=0.5, label='Generated Data', color='b', edgecolor='k')
    ax.set_xlabel('z', fontsize=12)
    ax.set_ylabel('p(z)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("Q5.png")
    plt.show()


if __name__ == "__main__":
    '''
    Code for Q1
    '''
    stream = os.popen('pip list')
    pip_list = stream.read()
    packages = pip_list.split('\n')

    numpy_2_1_0_installed = False
    numpy_2_1_1_installed = False

    for package in packages:
        if package.startswith('numpy'):
            package_info = package.split()
            if len(package_info) >= 2:
                if package_info[1] == '2.1.0':
                    numpy_2_1_0_installed = True
                    break
                elif package_info[1] == '2.1.1':
                    numpy_2_1_1_installed = True
                    break

    if numpy_2_1_0_installed:
        print("numpy 2.1.0 is installed")
    elif numpy_2_1_1_installed:
        print("numpy 2.1.1 is installed")
    else:
        print("Neither numpy 2.1.0 nor 2.1.1 is installed")
    
    '''
    Code for Q2
    '''
    coeffs = [1, -1,5, -0.1, 1/30]  # [w4, w3, w2, w1, w0]
    plot_polynomial(coeffs, [-4, 4])
    '''
    Code for Q4
    '''
    # Training
    X_train, y_train = create_dataset(coeffs, [-2, 2], 500, 0.5, seed=0)

    # Validation
    X_val, y_val = create_dataset(coeffs, [-2, 2], 500, 0.5, seed=1)
    '''
    Code for Q5
    '''
    visualize_data(X_train, y_train, coeffs, [-2, 2], "Training Data")
    visualize_data(X_val, y_val, coeffs, [-2, 2], "Validation Data")
    '''
    Code for Q6
    '''
    model = nn.Linear(5, 1,False) 
    loss_fn = nn.MSELoss() 
    learning_rate = 0.01
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)


    print(X_train.shape,y_train.shape,X_val.shape,y_val.shape)

    y_train = y_train.reshape(-1, 1)
    y_val = y_val.reshape(-1, 1)
    train_loss_vals = []
    val_loss_vals = []
    weights = []
    n_steps = 600 # Number of updates of the gradient
    for step in range(n_steps):
        model.train() # Set the model in training mode
        # Set the gradient to 0
        optimizer.zero_grad() # Or model.zero_grad()
        # Compute the output of the model
        y_hat = model(X_train)
        # Compute the loss
        loss = loss_fn(y_hat, y_train)
        # Compute the gradient
        loss.backward()
        # Update the parameters
        optimizer.step()
        # *** Evaluation ***
        # Here we do things that do not contribute to the gradient computation
        model.eval() # Set the model in evaluation mode
        with torch.no_grad(): #
            # Compute the output of the model
            y_hat_val = model(X_val)
            # Compute the loss
            loss_val = loss_fn(y_hat_val, y_val)
            # Compute the output of the model
            val_loss_vals.append(loss_val.item())
            train_loss_vals.append(loss.item())
            # At every step, print the losses
            weights.append(model.weight.flatten().tolist())
            print("Step:", step, "- Loss eval:", loss_val.item())
            # Do also a very simple plot

    # Get the final value of the parameters
    print("Final w:", model.weight, "Final b:\n", model.bias)
    '''
    Code for Q7
    '''
    plt.plot(range(step + 1), train_loss_vals)
    plt.plot(range(step + 1), val_loss_vals)
    plt.legend(["Training loss", "Validation loss"])
    plt.xlabel("Steps")
    plt.ylabel("Loss value")
    plt.savefig("Q7.png")
    plt.show()
    print("Training done, with an evaluation loss of {}".format(loss_val.item()))
    '''
    Code for Q8
    '''
    z_min, z_max = [-2,2]
    z = np.linspace(z_min, z_max)
    # Compute the true polynomial
    y_true = np.polyval(coeffs[::-1], z)
    est_weight = model.weight.flatten().tolist()
    est_coeffs = est_weight[::-1]
    est_y = np.polyval(est_coeffs, z)


    plt.plot(z, y_true, color='r', label='True Polynomial')
    plt.plot(z, est_y, color='g', label='Estimated Polynomial')
    plt.title("True vs. Estimated Polynomial")
    plt.xlabel('z')
    plt.ylabel('p(z)')
    plt.legend()
    plt.savefig("Q8.png")
    plt.show()
    '''
    Code for Q9
    '''
    colors = ['blue', 'orange', 'green', 'red', 'gray']
    weight_labels = [f'Weight {i}' for i in range(5)] 
    weight_array = np.array(weights)
    plt.figure(figsize=(10, 6))
    for i in range(5):
        plt.plot(np.arange(n_steps), weight_array[:, i], label=weight_labels[i], color=colors[i], linewidth=2)
        plt.axhline(y=coeffs[i], linestyle='--', label=f'True {weight_labels[i]}', color=colors[i], linewidth=1.5)
    plt.xlabel('Training Steps')
    plt.ylabel('Weight Value')
    plt.title('Evolution of Weights During Training')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1, fontsize=10, title="Weights", frameon=True)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig("Q9.png")
    plt.show()
    '''
    Code for Bonus
    '''
    def create_dataset_log(f, x_range, sample_size, sigma, seed=42):
        torch.manual_seed(seed)
        x_min, x_max = x_range
        X = torch.rand(sample_size) * (x_max - x_min) + x_min
        y = f(X) + torch.normal(torch.zeros(sample_size), sigma*torch.ones(sample_size))
        return X.reshape(-1, 1), y.reshape(-1, 1)

    def f(x):
        return 2 * torch.log(x + 1) + 3

    def train_and_evaluate(X_train, y_train, X_val, y_val, learning_rate=0.01, n_steps=1000):
        model = nn.Linear(1, 1)
        loss_fn = nn.MSELoss()
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        
        train_losses = []
        val_losses = []
        
        for step in range(n_steps):
            model.train()
            optimizer.zero_grad()
            y_hat = model(X_train)
            loss = loss_fn(y_hat, y_train)
            loss.backward()
            optimizer.step()
            
            model.eval()
            with torch.no_grad():
                y_hat_val = model(X_val)
                loss_val = loss_fn(y_hat_val, y_val)
                val_losses.append(loss_val.item())
                train_losses.append(loss.item())
                print("Step:", step, "- Loss eval:", loss_val.item())
        
        return model, train_losses, val_losses

    # Case 1: a = 0.01
    x_range1 = (-0.05, 0.01)
    # Case 2: a = 10
    x_range2 = (-0.05, 10)

    sample_size = 1000 
    sigma = 0.5 

    X_train1, y_train1 = create_dataset_log(f, x_range1, sample_size, sigma, seed=0)
    X_val1, y_val1 = create_dataset_log(f, x_range1, sample_size , sigma, seed=1)

    X_train2, y_train2 = create_dataset_log(f, x_range2, sample_size, sigma, seed=0)
    X_val2, y_val2 = create_dataset_log(f, x_range2, sample_size , sigma, seed=1)

    model1, train_losses1, val_losses1 = train_and_evaluate(X_train1, y_train1, X_val1, y_val1)
    model2, train_losses2, val_losses2 = train_and_evaluate(X_train2, y_train2, X_val2, y_val2)

    print(f"Final validation loss for case 1 (a = 0.01): {val_losses1[-1]:.6f}")
    print(f"Final validation loss for case 2 (a = 10): {val_losses2[-1]:.6f}")

    # Plotting
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses1, label='Train Loss')
    plt.plot(val_losses1, label='Validation Loss')
    plt.title('Case 1: a = 0.01')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_losses2, label='Train Loss')
    plt.plot(val_losses2, label='Validation Loss')
    plt.title('Case 2: a = 10')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig("QB1.png")
    plt.show()

    # Plotting the true function and linear approximations
    x1 = torch.linspace(-0.05, 0.01, 100).reshape(-1, 1)
    x2 = torch.linspace(-0.05, 10, 100).reshape(-1, 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(X_train1, y_train1, alpha=0.5, label='Training Data')
    plt.plot(x1, f(x1), 'r-', label='True Function')
    plt.plot(x1, model1(x1).detach(), 'black', linestyle='--', label='Linear Approximation')
    plt.title('Case 1: a = 0.01')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.scatter(X_train2, y_train2, alpha=0.5, label='Training Data')
    plt.plot(x2, f(x2), 'r-', label='True Function')
    plt.plot(x2, model2(x2).detach(), 'black', linestyle='--', label='Linear Approximation')
    plt.title('Case 2: a = 10')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()

    plt.tight_layout()
    plt.savefig("QB2.png")
    plt.show()
