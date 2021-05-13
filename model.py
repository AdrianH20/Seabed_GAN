from tensorflow.keras.layers import Input, Dense, LeakyReLU, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam


# Generator Model
def build_generator(latent_dim, D):
    i = Input(shape=(latent_dim,))
    x = Dense(256, activation=LeakyReLU(alpha=0.2))(i)
    x = BatchNormalization(momentum=0.7)(x)
    x = Dense(512, activation=LeakyReLU(alpha=0.2))(x)
    x = BatchNormalization(momentum=0.7)(x)
    x = Dense(1024, activation=LeakyReLU(alpha=0.2))(x)
    x = BatchNormalization(momentum=0.7)(x)
    x = Dense(D, activation='tanh')(x)

    model = Model(i, x)
    return model


# Discriminator Model
def build_discriminator(img_size):
    i = Input(shape=(img_size,))
    x = Dense(512, activation=LeakyReLU(alpha=0.2))(i)
    x = Dense(256, activation=LeakyReLU(alpha=0.2))(x)
    x = Dense(1, activation='sigmoid')(x)
    model = Model(i, x)
    return model


def compile(D, latent_dim):
    # Build and compile the discriminator
    discriminator = build_discriminator(D)
    discriminator.compile(
        loss='binary_crossentropy',
        optimizer=Adam(0.0002, 0.5),
        metrics=['accuracy'])

    # Build and compile the combined model
    generator = build_generator(latent_dim, D)

    # Create an input to represent noise sample from latent space
    z = Input(shape=(latent_dim,))

    # Pass noise through generator to get an image
    img = generator(z)

    # Only the generator will be trained
    discriminator.trainable = False

    # The true output is fake, but we label them real
    fake_prediction = discriminator(img)

    # Create and compile the combined model
    combined_model = Model(z, fake_prediction)
    combined_model.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))

    return [discriminator, generator, combined_model]