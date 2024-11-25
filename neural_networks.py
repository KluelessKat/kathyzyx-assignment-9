import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
import os
from functools import partial
from matplotlib.patches import Circle

result_dir = "results"
os.makedirs(result_dir, exist_ok=True)

# Define a simple MLP class
class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, lr, activation='tanh'):
        np.random.seed(0)
        self.lr = lr # learning rate
        self.activation_fn = activation # activation function
        assert self.activation_fn in ('sigmoid', 'tanh', 'relu')
        
        # TODO: define layers and initialize weights
        # self.W1 = np.random.randn(input_dim, hidden_dim)
        # self.b1 = np.zeros((1, hidden_dim))
        # self.W2 = np.random.randn(hidden_dim, output_dim)
        # self.b2 = np.zeros((1, output_dim))
        
        # Try Xavier W init and nonzero bias init
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(1 / input_dim)
        self.b1 = np.random.randn(1, hidden_dim) * 0.01
        self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(1 / hidden_dim)
        self.b2 = np.random.randn(1, output_dim) * 0.01
    
    def act(self, z):
        if self.activation_fn == 'tanh':
            A = np.tanh(z)
        elif self.activation_fn == 'relu':
            A = np.maximum(0, z)
        else :
            A = 1 / (1 + np.exp(-z))
        return A
        
    def forward(self, X):
        # TODO: forward pass, apply layers to input X
        # TODO: store activations for visualization
        
        #  X: Nx2 -> A1: Nx3
        self.Z1 = X @ self.W1 + self.b1  # (N, 3)
        self.A1 = self.act(self.Z1)
            
        # A1: Nx3 -> A2: Nx1
        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = np.tanh(self.Z2)
        # self.A2 = 1 / (1 + np.exp(-self.Z2))
        
        out = self.A2
        
        return out

    def backward(self, X, y):
        # TODO: compute gradients using chain rule
        # TODO: store gradients for visualization
        N = X.shape[0]
        
        # dZ2 = self.A2 - y  # Nx1
        dZ2 = 2 * (self.A2 - y) * (1 - np.tanh(self.Z2))
        
        self.dW2 = (self.A1.T @ dZ2) / N  # 3x1
        self.db2 = np.sum(dZ2, axis=0, keepdims=True) / N  # 1x1
        
        dA1 = dZ2 @ self.W2.T  # Nx3
        
        if self.activation_fn == 'tanh':
            dZ1 = dA1 * (1 - np.tanh(self.Z1) ** 2)
        elif self.activation_fn == 'relu':
            dZ1 = dA1 * (self.Z1 > 0).astype(float)
        else:
            assert self.activation_fn == 'sigmoid'
            sig = 1 / (1 + np.exp(-self.Z1))
            dZ1 = dA1 * sig * (1 - sig)
            
        self.dW1 = (X.T @ dZ1) / N  # 2x3
        self.db1 = np.sum(dZ1, axis=0, keepdims=True) / N  # 1x3
        
        # TODO: update weights with gradient descent
        self.W2 -= self.lr * self.dW2
        self.b2 -= self.lr * self.db2
        self.W1 -= self.lr * self.dW1
        self.b1 -= self.lr * self.db1

def generate_data(n_samples=100):
    np.random.seed(0)
    # Generate input
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 > 1).astype(int) * 2 - 1  # Circular boundary
    y = y.reshape(-1, 1)
    return X, y

# Visualization update function
def update(frame, mlp, ax_input, ax_hidden, ax_gradient, X, y):
    ax_hidden.clear()
    ax_input.clear()
    ax_gradient.clear()

    # perform training steps by calling forward and backward function
    for _ in range(10):
        # Perform a training step
        mlp.forward(X)
        mlp.backward(X, y)
    
    # Get input grid values
    x0_min, x0_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    x1_min, x1_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    grid_res = 50
    x0_grid, x1_grid = np.meshgrid(np.linspace(x0_min, x0_max, grid_res), 
                                   np.linspace(x1_min, x1_max, grid_res))
    X_grid_flat = np.stack((x0_grid.ravel(), x1_grid.ravel()), axis=1)
    
    # TODO: Plot hidden features
    # TODO: Hyperplane visualization in the hidden space
    # TODO: Distorted input space transformed by the hidden layer
    # Plot decision boundary
    xx, yy = np.meshgrid(np.linspace(-1, 1, grid_res), 
                         np.linspace(-1, 1, grid_res))
    zz = (-mlp.W2[0,0] * xx - mlp.W2[1,0] * yy - mlp.b2[0]) / mlp.W2[2,0]
    ax_hidden.plot_surface(xx, yy, zz, color='orange', alpha=0.4)
    
    # Plot manifold surf
    A1_grid_flat = mlp.act((X_grid_flat @ mlp.W1) + mlp.b1)
    ax_hidden.plot_surface(
        A1_grid_flat.reshape(grid_res, grid_res, 3)[:, :, 0],
        A1_grid_flat.reshape(grid_res, grid_res, 3)[:, :, 1],
        A1_grid_flat.reshape(grid_res, grid_res, 3)[:, :, 2],
        alpha=0.3,
        color='blue'
    )
    
    # Plot hidden feats scatterplot
    hidden_features = mlp.A1  # Nx3
    ax_hidden.scatter(hidden_features[:, 0], 
                      hidden_features[:, 1], 
                      hidden_features[:, 2], 
                      c=y.ravel(), cmap='bwr', alpha=0.7)
    
    
    # Plot input feats
    # TODO: Plot input layer decision boundary
    y_probs = mlp.forward(X_grid_flat).reshape(x0_grid.shape)  # 100x100
    y_preds = y_probs > 0  # final tanh act
    # ax_input.contourf(x0_grid, x1_grid, y_preds, 
    #                   levels=1, colors=['blue', 'red'], alpha=0.5)
    ax_input.contourf(x0_grid, x1_grid, y_probs,
                       levels=(-1., 0, 1.), 
                       cmap='bwr', alpha=0.5)
    ax_input.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='bwr', alpha=0.7,
                     edgecolor='black')
    ax_input.set_title(f"Input Space at Step {frame*10}")
    ax_input.set_xlim(x0_min, x0_max)
    ax_input.set_ylim(x1_min, x1_max)
    
    # TODO: Visualize features and gradients as circles and edges 
    # The edge thickness visually represents the magnitude of the gradient
    
    layer_sizes = [2, 3, 1]
    pos = {
        'x1': (0,   0),    
        'x2': (0,   1),
        'h1': (0.5, 0),  
        'h2': (0.5, 0.5),
        'h3': (0.5, 1),
        'y':  (1,   0.5)  
    }
    
    # Display Nodes
    node_size = 1000
    nodes = ['x1', 'x2', 'h1', 'h2', 'h3', 'y']
    for node in nodes:
        ax_gradient.scatter(pos[node][0], pos[node][1], 
                            s=node_size, c='blue', zorder=2)
        ax_gradient.text(pos[node][0], pos[node][1], node, 
                horizontalalignment='center', 
                verticalalignment='center',
                color='white',
                fontsize=10)
        
    max_grad = max(np.abs(mlp.dW1).max(), np.abs(mlp.dW2).max())
    
    # Input to hidden conns
    for i in range(2):  # input nodes
        for j in range(3):  # hidden nodes
            input_node = f'x{i+1}'
            hidden_node = f'h{j+1}'
            grad = np.abs(mlp.dW1[i,j])
            alpha = np.clip(grad / max_grad, 0.1, 1.0)
            ax_gradient.plot(
                [pos[input_node][0], pos[hidden_node][0]],
                [pos[input_node][1], pos[hidden_node][1]],
                'purple', 
                alpha=alpha, 
                linewidth=1, zorder=1
            )
    
    # Hidden to output conns
    for j in range(3):  # hidden nodes
        hidden_node = f'h{j+1}'
        grad = np.abs(mlp.dW2[j,0])
        alpha = np.clip(grad / max_grad, 0.1, 1.0)
        ax_gradient.plot(
            [pos[hidden_node][0], pos['y'][0]],
            [pos[hidden_node][1], pos['y'][1]],
            'purple', 
            alpha=alpha, linewidth=1, zorder=1)
    
    ax_gradient.set_title(f"Gradients at Step {frame*10}")
    ax_gradient.axis('off')


def visualize(activation, lr, step_num):
    X, y = generate_data()  # X: Nx2, y: Nx1
    mlp = MLP(input_dim=2, hidden_dim=3, output_dim=1, lr=lr, activation=activation)

    # Set up visualization
    matplotlib.use('agg')
    fig = plt.figure(figsize=(21, 7))
    ax_hidden = fig.add_subplot(131, projection='3d')
    ax_input = fig.add_subplot(132)
    ax_gradient = fig.add_subplot(133)

    # Create animation
    ani = FuncAnimation(fig, 
                        partial(
                            update, 
                            mlp=mlp, 
                            ax_input=ax_input, 
                            ax_hidden=ax_hidden, 
                            ax_gradient=ax_gradient, 
                            X=X, 
                            y=y
                            ), 
                        frames=step_num//10, 
                        repeat=False)

    # Save the animation as a GIF
    ani.save(os.path.join(result_dir, "visualize.gif"), writer='pillow', fps=10)
    plt.close()

if __name__ == "__main__":
    activation = "tanh"
    lr = 0.1
    step_num = 1000
    visualize(activation, lr, step_num)