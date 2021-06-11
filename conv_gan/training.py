from numpy import zeros
from numpy import ones
from numpy.random import randn
from numpy.random import randint
from matplotlib import pyplot


plot_path = 'C:\\Users\\adria\\PycharmProjects\\Seabed\\gan_images\\'
model_path = 'C:\\Users\\adria\\PycharmProjects\\Seabed\\models\\'


def save_plot(generator, latent_dim, sample_number, epoch, n=7):
    examples, _ = generate_fake_samples(generator, latent_dim, sample_number)
    # From [-1,1] to [0,1]
    examples = (examples + 1) / 2.0
    # Plot samples
    for i in range(n * n):
        pyplot.subplot(n, n, 1 + i)
        pyplot.axis('off')
        pyplot.imshow(examples[i])
    # Saving
    filename = plot_path+'generator_model_3c_test_structured_plot_e{}.png'.format(epoch + 1)
    pyplot.savefig(filename)
    pyplot.close()


def select_real_samples(real_dataset, batch_size):
    # Take a random selection in the dataset and assigning labels of 1 to them
    random_selection = randint(0, real_dataset.shape[0], batch_size)
    real_data = real_dataset[random_selection]
    real_labels = ones((batch_size, 1))
    return real_data, real_labels


def build_latent_space(latent_dim, batch_size, label_class=0):
    # Create a latent space and label data for the generator
    latent_space = randn(latent_dim * batch_size)
    latent_space = latent_space.reshape(batch_size, latent_dim)
    if label_class == 1:
        fake_labels = ones((batch_size, 1))
    else:
        fake_labels = zeros((batch_size, 1))
    return latent_space, fake_labels


def generate_fake_samples(generator, latent_dim, n_samples):
    # Generated data will be labeled as fake data for training the discriminator
    generator_input, fake_labels = build_latent_space(latent_dim, n_samples)
    generated_data = generator.predict(generator_input)
    return generated_data, fake_labels


def train(generator, discriminator, gan, dataset, latent_dim, epochs, batch_size):
    bat_per_epo = int(dataset.shape[0] / batch_size) * 10
    half_batch = int(batch_size / 2)
    # Go through every epoch
    for i in range(epochs):
        # In each iteration epoch will have a sufficient number of batches to cover the entire epoch
        for j in range(bat_per_epo):
            # Discriminator Training #######################

            # Divide each batch in half.
            # First half it will be used to train the discriminator on real data
            real_data, real_labels = select_real_samples(dataset, half_batch)
            d_loss_real, d_acc_real = discriminator.train_on_batch(real_data, real_labels)
            # Second half will be used to train the discriminator on fake/generated data
            fake_data, fake_labels = generate_fake_samples(generator, latent_dim, half_batch)
            d_loss_fake, d_acc_fake = discriminator.train_on_batch(fake_data, fake_labels)
            # Generator Training     #######################

            # Swap labels for generator training
            train_data, train_labels = build_latent_space(latent_dim, batch_size, 1)
            # Update the GAN with the new weights
            g_loss = gan.train_on_batch(train_data, train_labels)

            # Print loss and accuracy
            d_loss = 0.5 * (d_loss_real + d_loss_fake)
            d_acc = 0.5 * (d_acc_real + d_acc_fake)
            if j % 20 == 0:
                print("Epoch: {}/{}, Batch:{}/{} d_loss: {:.2f}, d_acc: {:.2f}, g_loss: {:.2f}".
                      format(i + 1, epochs, j+1, bat_per_epo, d_loss, d_acc, g_loss))
        # Save models and plots from 10 to epochs
        if (i + 1) % 10 == 0:
            save_plot(generator, latent_dim, 50, i+1)
            generator_name = model_path+'generator_model_3c_{}.h5'.format(i + 1)
            generator.save(generator_name)
