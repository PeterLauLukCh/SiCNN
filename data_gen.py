import numpy as np
import matplotlib.pyplot as plt

def plot_datasets():
    # Load datasets
    dataset_X = np.load(r'C:\Users\chenl\Desktop\SiCNN\2_dim_experiments\data\X.npy')
    dataset_Y = np.load(r'C:\Users\chenl\Desktop\SiCNN\2_dim_experiments\data\Y.npy')

    # Create plot
    plt.figure(figsize=(10, 6))

    # Plot dataset X
    plt.scatter(dataset_X[:, 0], dataset_X[:, 1], c='blue', label='X (Base Measure)', alpha=0.5, edgecolors='none')

    # Plot dataset Y
    plt.scatter(dataset_Y[:, 0], dataset_Y[:, 1], c='red', label='Y (Target Measure)', alpha=0.5, edgecolors='none')

    # Add titles and labels
    plt.title('Visualization of Datasets X and Y')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.legend()
    plt.grid(True)

    # Show plot
    plt.show()
	
BATCH_SIZE = 256

dataset_X = []
for i in range(BATCH_SIZE):
	x = np.random.uniform(0, 600)
	y = np.random.uniform(0, 400)
	dataset_X.append([x, y])
dataset_X = np.array(dataset_X, dtype='float32')
np.save(r'C:\Users\chenl\Desktop\SiCNN\2_dim_experiments\data\X.npy', dataset_X)

dataset_Y = []
for i in range(BATCH_SIZE // 2):  # Splitting the batch between two Y boxes
	x = np.random.uniform(0, 580)
	y = np.random.uniform(600, 800)
	dataset_Y.append([x, y])
			
for i in range(BATCH_SIZE // 2):
	x = np.random.uniform(800, 900)
	y = np.random.uniform(-50, 450)
	dataset_Y.append([x, y])
dataset_Y = np.array(dataset_Y, dtype='float32')
np.save(r'C:\Users\chenl\Desktop\SiCNN\2_dim_experiments\data\Y.npy', dataset_Y)

plot_datasets()