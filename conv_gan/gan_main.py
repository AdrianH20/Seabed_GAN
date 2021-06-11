import data_helper
import conv_gan.model as model
from conv_gan.training import train
from conv_gan.training import generate_fake_samples
from numpy import expand_dims
from tensorflow.keras.models import load_model as load_model

folder_path = 'C:\\Users\\adria\\Desktop\\structured\\rescaled_samples\\'
sample_prefix = 'rescaled_sample'

# Configurations
latent_dim = 100
epochs = 31
batch_size = 32


def save_gan_samples(generated_list, path):
    generated_list = (generated_list + 1) / 2.0
    i = 0
    for image in generated_list:
        data_helper.save_png(image, '{0}generated_image{1}.png'.format(path, str(i)), 0.42)
        i += 1


def load_real_samples():
    trainX = data_helper.get_samples(folder_path, sample_prefix, 2342)
    trainX = expand_dims(trainX, axis=-1)
    # convert from unsigned ints to floats
    trainX = trainX.astype('float32')
    # scale from [0,255] to [-1,1]
    trainX = (trainX - 127.5) / 127.5
    return trainX


def create_model():

    # Discriminator
    discriminator = model.build_discriminator()
    # Generator
    generator = model.build_generator(latent_dim)
    # Combined Model
    gan = model.build_gan(generator, discriminator)
    # load image data
    dataset = load_real_samples()
    # train model
    train(generator, discriminator, gan, dataset, latent_dim, epochs, batch_size)


# create_model()

