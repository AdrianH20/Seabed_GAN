import numpy as np
import math
import matplotlib.pyplot as plt
from PIL import Image as Image
from numpy import asarray


def save_png(z_grid, name):
    plt.figure()
    plt.imshow(z_grid)
    plt.axis('off')
    plt.savefig(name, bbox_inches='tight', pad_inches=0)
    plt.close()


def create_samples(file_path, target_path, x_rows, y_cols, sample_size):
    data_buffer = np.empty((x_rows, y_cols), float)

    i = 0
    with open(file_path) as infile:
        for line in infile:
            if (i > 5):
                data_buffer[i - 6] = [float(data) for data in line.split()]
                if (i % 100 == 0):
                    print("{} rows from the file were read")
            i += 1

    N = math.floor(x_rows / sample_size)
    M = math.floor(y_cols / sample_size)

    k = 0
    for i in range(N):
        for j in range(M):
            Z = data_buffer[sample_size * i:sample_size * (i + 1), sample_size * j:sample_size * (j + 1)]
            save_png(Z, target_path.format(k))
            if k % 200 == 0:
                print("Sample#{} was created.".format(k))
            k += 1


def rescale_sample(folder_path, rescale):
    for i in range(10000):
        image = Image.open(folder_path+"real_fig{}.png".format(i))
        new_image = image.resize((rescale, rescale))
        new_image.save(folder_path+"real_samples\\real_sample{}.png".format(i))
        if i % 100 == 0:
            print("First {} samples were created.".format(i))


def get_samples(folder_path, sample_prefix):

    data_samples = []
    for i in range(10000):

        image = Image.open(folder_path+sample_prefix+str(i)+".png").convert("L")
        data = asarray(image)
        data_samples.append(data)
        if i % 200 == 0:
            print(i)
    train_x = np.array(data_samples)
    return train_x
