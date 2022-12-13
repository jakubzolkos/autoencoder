import os
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
import matplotlib.pyplot as plt
from keras import backend as K
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image, ImageChops
from sklearn.neighbors import KernelDensity
import random
from sklearn import metrics


class DataGenerator(object):

    def __init__(self, directory):
        self.dir = directory
    
    def get_generators(self, img_size, batch_size):

        generator = ImageDataGenerator(rescale = 1./255, data_format = 'channels_last')

        train_gen = generator.flow_from_directory(
            os.path.join(self.dir, 'train'),
            target_size = img_size,
            batch_size = batch_size,
            class_mode = 'input'
            )

        validation_gen = generator.flow_from_directory(
            os.path.join(self.dir, 'test'),
            target_size = img_size,
            batch_size = batch_size,
            class_mode = 'input'
            )

        anomaly_gen = generator.flow_from_directory(
            os.path.join(self.dir, 'anomaly'),
            target_size = img_size,
            batch_size = batch_size,
            class_mode = 'input'
            )
        
        return train_gen, validation_gen, anomaly_gen


class Autoencoder(object):

    def __init__(self, input_shape, train_gen, validation_gen):
        self.input_shape = input_shape
        self.train_gen = train_gen
        self.validation_gen = validation_gen
        self.model = None
        self.encoder = None

    def get_model(self, train_new = False):
        # If model already exists, load it from models folder
        if "model.keras" in os.listdir(os.path.join(os.getcwd(), "models")) and train_new is False:
            model = keras.models.load_model(os.path.join(os.getcwd(), "models/model.keras"))
            self.model = model
            return model
        
        else:
            model = Sequential([

                Input(shape = self.input_shape),
                Conv2D(64, (3, 3), padding = 'same', activation = 'relu', input_shape = self.input_shape),
                MaxPooling2D(pool_size = (2, 2), padding = 'same'),
                Conv2D(32,(3, 3), activation = 'relu', padding = 'same'),
                MaxPooling2D(pool_size = (2, 2), padding = 'same'),
                Conv2D(8,(3, 3), activation = 'relu',  padding = 'same'),
                MaxPooling2D(pool_size = (4, 4), padding = 'same'),
                Conv2D(8,(3, 3), activation = 'relu', padding = 'same'),
                UpSampling2D((4, 4)),
                Conv2D(32,(3, 3),activation = 'relu',  padding = 'same'),
                UpSampling2D((2, 2)),
                Conv2D(64,(3, 3),activation = 'relu', padding = 'same'),
                UpSampling2D((2, 2)),
                Conv2D(3, (3, 3), activation = 'sigmoid', padding = 'same')
            ])

            model.summary()
            model.compile(optimizer = 'adam', loss = 'mean_squared_error')

            early_stop = keras.callbacks.EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience = 15) 
            best = keras.callbacks.ModelCheckpoint(os.path.join(os.getcwd(), "models/model.keras"), monitor = 'val_loss', save_best_only = True, mode = 'min') 

            model.fit(self.train_gen, steps_per_epoch = 10, epochs = 100, validation_data = self.validation_gen, validation_steps = 10, shuffle = True, callbacks = [early_stop, best])
            self.model = model

            return model

    def get_encoder(self):

        encoder = Sequential([
            Conv2D(64, (3, 3), padding = 'same',activation = 'relu', input_shape = self.input_shape, weights = self.model.layers[0].get_weights()),
            MaxPooling2D(pool_size = (2,2), padding = 'same'),
            Conv2D(32,(3, 3),activation = 'relu',  padding = 'same', weights = self.model.layers[2].get_weights()),
            MaxPooling2D(pool_size = (2,2), padding = 'same'),
            Conv2D(8,(3, 3),activation = 'relu',  padding = 'same', weights = self.model.layers[4].get_weights()),
            MaxPooling2D(pool_size = (4,4), padding = 'same')
        ])

        self.encoder = encoder
        return encoder

    def kde(self, data, bandwidth):

        if self.encoder is not None:
            encoded_imgs = self.encoder.predict(data)
            latent_vector = [np.reshape(img, (512)) for img in encoded_imgs]
            density = KernelDensity(kernel = 'gaussian', bandwidth = bandwidth).fit(latent_vector)
            estimate = density.score_samples(latent_vector)
            return estimate
        
        else:
            return None
    
    def evaluate(self, images, bandwidth, rec_thresh, dens_thresh, ground_truth):
        
        def mse(images, reconstruction):
            return np.mean((images - reconstruction)**2, axis = (1,2,3)) 

        if self.model is not None and self.encoder is not None:

            reconstructions = self.model.predict(images)

            MSE = mse(images, reconstructions)
            KDE = self.kde(images, bandwidth)

            reconstruction_anomalies = MSE > rec_thresh

            if dens_thresh is not None:
                density_anomalies = KDE > dens_thresh
            else:
                density_anomalies = [0]
            
            total_anomalies = np.maximum(reconstruction_anomalies, density_anomalies)
            detection_rate = np.mean(total_anomalies)
            
            print(f'Reconstruction anomalies = {sum(reconstruction_anomalies)}')
            print(f'Density anomalies = {sum(density_anomalies)}')
            print(f'Combined anomalies = {sum(total_anomalies)}')
            print(f'Anomaly detection rate = {np.round(detection_rate, 3)}')
            print(f"Ground Truth: {ground_truth}")
            
            return sum(total_anomalies), len(images)


if __name__ == "__main__":

    # Generate data
    data = DataGenerator(os.getcwd())
    train_gen, validation_gen, anomaly_gen = data.get_generators((128, 128), 85)

    # Get autoencoder model and the encoder
    autoencoder = Autoencoder(input_shape = (128, 128, 3), train_gen = train_gen, validation_gen = validation_gen)
    model = autoencoder.get_model(train_new = False)
    encoder = autoencoder.get_encoder()
    print(f"Validation Error: {model.evaluate(validation_gen)}")
    print(f"Anomaly Error: {model.evaluate(anomaly_gen)}")

    # Get latent space dsitribution and plot it
    training_density_scores = autoencoder.kde(train_gen, 0.85)
    validation_density_scores = autoencoder.kde(validation_gen, 0.85)
    anomaly_density_scores = autoencoder.kde(anomaly_gen, 0.85)

    plt.figure(figsize = (10,7))
    plt.title('Distribution of Density Scores')
    plt.hist(training_density_scores, 12, alpha = 0.5, label = 'Training Normal')
    plt.hist(validation_density_scores, 12, alpha = 0.5, label = 'Validation Normal')
    plt.hist(anomaly_density_scores, 12, alpha = 0.5, label = 'Anomalies')
    plt.legend(loc = 'upper right')
    plt.xlabel('Sample Log-Likelihood')
    plt.show()

    # Plot original image and reconstruction for a normal and anomaly sample images
    normal = train_gen.next()[0]
    normal_predict = model.predict(normal)
    anomaly = anomaly_gen.next()[0]
    anomaly_predict = model.predict(anomaly)
    plt.subplot(2, 2, 1)
    plt.imshow(normal[0])
    plt.subplot(2, 2, 2)
    plt.imshow(normal_predict[0])
    plt.subplot(2, 2, 3)
    plt.imshow(anomaly[0])
    plt.subplot(2, 2, 4)
    plt.imshow(anomaly_predict[0])
    plt.show()

    # Function for getting test images
    def get_ims(test_file):

        batch_size  = 0
        for _, _, filenames in os.walk(test_file):
            batch_size += len([file for file in filenames if file.endswith(".jpg")]) # Batchsize now is total images in folder
        
        generator = ImageDataGenerator(rescale = 1./255, data_format = 'channels_last')
        test_img_generator = generator.flow_from_directory(test_file, target_size = (128, 128), batch_size = batch_size, class_mode = 'input')
        images = test_img_generator.next()[0]
        return images
        
    # Get images for testing: normal and anomaly
    normal = get_ims('test')
    anomaly = get_ims('anomaly')

    # Evaluate the model performance based on detection rate
    anomalies, total = autoencoder.evaluate(normal, 0.04, 13.7, "Normal")
    anomalies, total = autoencoder.evaluate(anomaly, 0.04, 13.7, "Anomaly")


    # Plotting the ROC curve - might require high computational power
    # Modify start, stop, num values as needed

    kde_start, kde_stop, kde_num = -392.5, -391.5, 3
    rec_start, rec_stop, rec_num = 0, 0.05, 30 
    density_thresholds = np.linspace(kde_start, kde_stop, kde_num)
    reconstruction_thresholds = np.linspace(rec_start, rec_stop, rec_num)
    FPR = np.zeros((len(density_thresholds), len(reconstruction_thresholds)))
    TPR = np.zeros((len(density_thresholds), len(reconstruction_thresholds)))
    FPR_R = np.zeros((1, len(reconstruction_thresholds)))
    TPR_R = np.zeros((1, len(reconstruction_thresholds)))

    # ROC for models with density threshold
    for i, dens_thresh in enumerate(density_thresholds):
        for j, rec_thresh in enumerate(reconstruction_thresholds):
            fn, total = autoencoder.evaluate(normal, 0.85, rec_thresh, dens_thresh, "Normal")
            tp = total - fn
            tn, total = autoencoder.evaluate(anomaly, 0.85, rec_thresh, dens_thresh, "Anomaly")
            fp = total - tn
            FPR[0, j] = fp / (fp + tn)
            TPR[0, j] = tp / (tp + fn)
    
    # ROC for model with reconstruction threshold only
    for j, rec_thresh in enumerate(reconstruction_thresholds):
            fn, total = autoencoder.evaluate(normal, 0.85, rec_thresh, None, "Normal")
            tp = total - fn
            tn, total = autoencoder.evaluate(anomaly, 0.85, rec_thresh, None,  "Anomaly")
            fp = total - tn
            FPR_R[0, j] = fp / (fp + tn)
            TPR_R[0, j] = tp / (tp + fn)
        
    for i in range(kde_num):
        plt.plot(FPR[i, :], TPR[i, :], label = f'DT = {density_thresholds[i]}')

    plt.plot(FPR_R[0, :], TPR_R[0, :], label = "No DT")
    plt.plot(np.linspace(0, 1, 30), np.linspace(0, 1, 30), '--', label = 'Reference')
    plt.legend()
    plt.xlabel("FP Rate")
    plt.ylabel("TP Rate")
    plt.title("ROC - Reconstruction Threshold")
    plt.show()


