import tensorflow as tf
import data_helper
import model
import training
import os

data_folder_path = "GAN_data\\real_data_samples\\real_samples\\"
sample_prefix = "real_sample"
target_folder = "gan_images"

if not os.path.exists(target_folder):
    os.makedirs(target_folder)

# Get training data
x_train = data_helper.get_samples(data_folder_path, sample_prefix)

# Map training data in interval (-1, +1) for better training
x_train = x_train / 255.0 * 2 - 1

# The shape of training data is (10000, 110, 110) meaning
# there's 10000 images of 110x110 pixels
print("x_train.shape:", type(x_train))

# Taking the dimensionality of images from the shape function
# and map every sample on a single rows in a matrix of (10000,110x110)
N, Height, Width = x_train.shape
D = Height * Width
x_train = x_train.reshape(-1, D)

# Config
batch_size = 32
epochs = 10000
sample_period = 200 # every `sample_period` steps generate and save some data

# This is the latent space dimension, it's a hyper parameter and it can be change based on data dimensionality
latent_dim = 370

[discriminator, generator, combined_model] = model.compile(D, latent_dim)

# Training bt parsing the models and the configuration parameters
training.train(discriminator=discriminator,
               generator=generator,
               combined_model=combined_model,
               epochs=epochs,
               x_train=x_train,
               batch_size=batch_size,
               latent_dim=latent_dim,
               sample_period=sample_period,
               sample_height=Height,
               sample_width=Width)
