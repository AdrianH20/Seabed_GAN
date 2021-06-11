import numpy as np
import matplotlib.pyplot as plt
import data_helper

def sample_images(generator, epoch, latent_dim, sample_height, sample_width):
    rows, cols = 5, 5
    noise = np.random.randn(rows * cols, latent_dim)
    images = generator.predict(noise)

    figure, axs = plt.subplots(rows, cols)
    idx = 0
    for i in range(rows):
        for j in range(cols):
            axs[i, j].imshow(images[idx].reshape(sample_height, sample_width))
            axs[i, j].axis('off')
            idx += 1
    figure.savefig("gan_images2/colored%d.png" % epoch)
    plt.close()


def select_real_samples(dataset, batch_size, dim):
    ix = np.random.randint(dataset.shape[0], size=batch_size)

    X = dataset[ix]
    X = X.reshape(batch_size, dim, dim)
    y = np.ones((batch_size, 1))
    return X, y


def generate_fake_images(generator, latent_dim, sample_height, sample_width, samples_number):
    path = 'C:\\Users\\adria\\Desktop\\dense_improvised_samples\\'
    for i in range(samples_number):
        noise = np.random.randn(1, latent_dim)
        image = generator.predict(noise)
        image = image.reshape(sample_height, sample_width)
        data_helper.save_png(image, '{0}generated_image{1}.png'.format(path, str(i)), 0.37)


def train(discriminator, generator, combined_model, epochs, x_train, batch_size, latent_dim,
          sample_period, sample_height, sample_width):
    # Create batch labels to use when calling train_on_batch
    ones = np.ones(batch_size)
    zeros = np.zeros(batch_size)

    # Store the losses
    d_losses = []
    g_losses = []
    for epoch in range(epochs):

        # Train Discriminator
        # Select a random batch of images
        idx = np.random.randint(0, x_train.shape[0], batch_size)
        real_imgs = x_train[idx]
        #X_real, y_real = select_real_samples(x_train, batch_size, sample_height)

        # Generate fake images
        noise = np.random.randn(batch_size, latent_dim)
        fake_imgs = generator.predict(noise)

        # Train the discriminator
        d_loss_real, d_acc_real = discriminator.train_on_batch(real_imgs, ones)
        d_loss_fake, d_acc_fake = discriminator.train_on_batch(fake_imgs, zeros)
        d_loss = 0.5 * (d_loss_real + d_loss_fake)
        d_acc = 0.5 * (d_acc_real + d_acc_fake)

        # Train generator 2 times in each epoch
        noise = np.random.randn(batch_size, latent_dim)
        g_loss = combined_model.train_on_batch(noise, ones)

        noise = np.random.randn(batch_size, latent_dim)
        g_loss = combined_model.train_on_batch(noise, ones)

        # Save the losses
        d_losses.append(d_loss)
        g_losses.append(g_loss)

        # Print current loses each 100 epochs
        if epoch % 100 == 0:
            print("Epoch: {}/{}, d_loss: {:.2f}, d_acc: {:.2f}, g_loss: {:.2f}".
                  format(epoch + 1, epochs, d_loss, d_acc, g_loss))

        if epoch % sample_period == 0:
            sample_images(generator=generator, epoch=epoch, latent_dim=latent_dim, sample_height=sample_height, sample_width=sample_width)
    #generate_fake_images(generator=generator, latent_dim=latent_dim, sample_height=sample_height, sample_width=sample_width, samples_number=1270)
