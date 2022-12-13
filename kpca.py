from sklearn.decomposition import PCA, KernelPCA
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.datasets import mnist
from keras import Model
from keras import Sequential
from keras.layers import Dense, Dropout
from keras.losses import MeanSquaredLogarithmicError
from keras import layers, losses
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize pixel values to [0., 1.]
x_train = x_train / 255.
#x_test = x_test / 255.
x_test = x_test.astype('float32') / 255.

pca = PCA(n_components=32)
kernel_pca = KernelPCA(
    n_components=32, kernel="rbf", gamma=1e-3, fit_inverse_transform=True, alpha=5e-3
)

x_reshaped = x_train[0:5000].reshape(-1, 784)              # new shape is (500, 28*28) = (500, 784)
x_scaled = StandardScaler().fit_transform(x_reshaped)    # center and scale data (mean=0, std=1)
x_transformed = pca.fit(x_reshaped)#.transform(x_reshaped)
x_transformed_k = kernel_pca.fit(x_reshaped)#.transform(x_reshaped)

x_test_reshaped = x_test[0:500].reshape(-1,784)
x_scaled_test = x_test_reshaped#StandardScaler().fit_transform(x_test_reshaped)


X_reconstructed_kernel_pca = kernel_pca.inverse_transform(
    kernel_pca.transform(x_scaled_test)
)
X_reconstructed_pca = pca.inverse_transform(pca.transform(x_scaled_test))

def plotting(X, title=""):
    fig, axs = plt.subplots(nrows=1, ncols=10, figsize=(8, 1))
    for img, ax in zip(X, axs.ravel()):
        ax.imshow(img.reshape((28, 28)),cmap="gist_gray")
        ax.axis("off")
    fig.suptitle(title, fontsize=10)
	
plotting(x_test)#, "Test images")
plotting(
    X_reconstructed_kernel_pca,
    #"Kernel PCA reconstruction with"
f" MSE: {np.mean((x_test_reshaped - X_reconstructed_kernel_pca) ** 2):.2f}",
)
