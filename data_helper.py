import numpy as np
import math
import matplotlib.pyplot as plt

import os
import glob


from PIL import Image as Image

# FID imports
from numpy import cov, trace as tr, iscomplexobj, asarray
from scipy.linalg import sqrtm
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from skimage.transform import resize


class File:
    def __init__(self, path, file_name, rows, x_diff, y_diff, cut, sample_size):
        self.path = path
        self.file_name = file_name
        self.rows = rows
        self.x_diff = x_diff
        self.y_diff = y_diff
        self.cut = cut
        self.sample_size = sample_size


def scale_samples(images, new_shape):
    images_list = list()
    for image in images:
        new_image = resize(image, new_shape, 0)
        # store
        images_list.append(new_image)
    return asarray(images_list)


def compute_fid(model, data_set1, data_set2):
    activation1 = model.predict(data_set1)
    activation2 = model.predict(data_set2)

    (mean1, mean2) = activation1.mean(axis=0), activation2.mean(axis=0)
    (sigma1, sigma2) = cov(activation1, rowvar=False), cov(activation2, rowvar=False)

    squared_diff = np.sum(mean1 - mean2)**2.0
    covariance_mean = sqrtm(sigma1.dot(sigma2))

    if iscomplexobj(covariance_mean):
        covariance_mean = covariance_mean.real

    fid = squared_diff + tr(sigma1 + sigma2 - 2.0*covariance_mean)
    fid = round(fid, 2)
    return fid

#    trainX = data_helper.get_samples("C:\\Users\\adria\\Desktop\\simple_images_for_gan\\rescaled_samples\\", 'rescaled_sample', 1269)


def get_fid_accuracy(path1, prefix1, path2, prefix2, samples_number, dim):
    inception_model = InceptionV3(include_top=False, pooling='avg', input_shape=(dim, dim, 3))
    sample_set1 = get_samples(path1, prefix1, samples_number - 1)
    sample_set2 = get_samples(path2, prefix2, samples_number - 1)

    sample_set1 = sample_set1.astype('float32')
    sample_set1 = preprocess_input(scale_samples(sample_set1, (dim, dim, 3)))
    sample_set2 = sample_set2.astype('float32')
    sample_set2 = preprocess_input(scale_samples(sample_set2, (dim, dim, 3)))

    fid = compute_fid(inception_model, sample_set1, sample_set2)
    return fid


def save_png(z_grid, name, dim=0):
    if dim == 0:
        plt.figure()
    else:
        plt.figure(figsize=(dim, dim))
    plt.imshow(z_grid)
    plt.axis('off')
    plt.savefig(name, bbox_inches='tight', pad_inches=0)
    plt.close()


def show_image(z_grid):
    plt.figure(figsize=(8, 6))
    plt.imshow(z_grid)
    plt.show()


def mapping_to_grid(point_clouds, rows_number, x_diff, y_diff, cut):
    """This function takes a point cloud array and map the given points into a self-sufficient grid that contains
    accurate data. The grid will be returned"""
    grid_size = math.floor(math.sqrt(rows_number)+1)
    np_image = np.tile(0, (grid_size, grid_size))
    X = point_clouds[:, 0].reshape(1, rows_number) * 100 - x_diff * 100
    Y = point_clouds[:, 1].reshape(1, rows_number) * 100 - y_diff * 100
    Z = point_clouds[:, 2].reshape(1, rows_number) * 100

    c = 0
    d = grid_size
    a = np.amin(X)
    b = np.amax(X)
    X = c + (d - c) / (b - a) * (X - a)
    X = X.astype(int)
    X[X > grid_size-1] = grid_size-1

    a = np.amin(Y)
    b = np.amax(Y)
    Y = c + (d - c) / (b - a) * (Y - a)
    Y = Y.astype(int)
    Y[Y > grid_size-1] = grid_size-1

    mean = np.mean(Z)

    Z = Z - mean
    mean = int(np.mean(Z))
    np_image[X, Y] = Z

    for i in range(grid_size):
        for j in range(grid_size):
            if np_image[i][j] > cut:
                np_image[i][j] = mean
            if np_image[i][j] < -cut:
                np_image[i][j] = mean

    c = 0
    d = 255

    a = np.amin(np_image)
    b = np.amax(np_image)

    np_image = c + (d - c) / (b - a) * (np_image - a)

    null_value = int(c + (d - c) / (b - a) * -a)
    np_image = np_image.astype(int)
    #m = np.mean(Z)

    show_image(np_image)
    return np_image, grid_size, null_value


def grid_tiling(grid_map, grid_size, sample_size, null_value):
    """This function take an grid map and tile the entire grid into smaller grids.
       Returned value will be a list of smaller grids that have a 70% area of coverage with real data"""
    tile_list = []
    percent = int(0.3*(sample_size**2))
    N = math.floor(grid_size/sample_size)
    for i in range(N):
        for j in range(N):
            tile = grid_map[i*sample_size:(i+1)*sample_size, j*sample_size:(j+1)*sample_size]
            null_count = np.count_nonzero(tile == null_value)
            if null_count < percent:
                #show_image(tile)
                tile_list.append(tile)
    return tile_list


def noise_cancellation(tile_list, tile_size, null_value):

    for tile in tile_list:
        non_zero_list = []
        for i in range(tile_size):
            for j in range(tile_size):
                if tile[i][j] != null_value:
                    non_zero_list.append(tile[i][j])
        mean = int(sum(non_zero_list)/len(non_zero_list))
        for i in range(tile_size):
            for j in range(tile_size):
                if tile[i][j] == null_value:
                    tile[i][j] = mean
        #show_image(tile)
    return tile_list


# ------------------------------LATER FOR GAN------------------------------------------------
def rescale_samples(folder_path, rescale):
    file_names = glob.glob(folder_path+'*.png')
    i = 0
    for file in file_names:
        image = Image.open(file)
        new_image = image.resize((rescale, rescale))
        new_image.save(folder_path+"rescaled_samples\\rescaled_sample{}.png".format(i))
        if i % 100 == 0:
            print("First {} samples were created.".format(i))
        i += 1


def rotate_samples(folder_path, angle):
    file_names = glob.glob(folder_path+'*.png')
    i = 0
    for file in file_names:
        image = Image.open(file)
        new_image = image.rotate(angle, expand=True)
        new_image.save(folder_path+"rotated_samples\\rotated_180_sample{}.png".format(i))
        if i % 100 == 0:
            print("First {} samples were created.".format(i))
        i += 1
# ------------------------------LATER FOR GAN------------------------------------------------
def get_samples(folder_path, sample_prefix, N):

    data_samples = []
    data_array = np.ndarray(shape=(32, 32, 1))
    for i in range(N):

        image = Image.open(folder_path+sample_prefix+str(i)+".png").convert('L')#convert('RGB')
        data = asarray(image)
        # for j in range(32):
        #     for k in range(32):
        #         data_array[j][k][0] = data[j][k]
        data_samples.append(data)
        if i % 200 == 0:
            print(i)
    train_x = np.array(data_samples)
    return train_x


def get_files(path):
    """This function will read data from the path text file and
    will return a list of File object

     The data inside the text file need to provide the following:
     1. Sample file name
     2. Number of rows
     3. x diff
     4. y diff
     5. cut (the cut is referring to those values of z that are too bigger from the mean of the other elements)
     6. sample size (the dimension of each tile which will result from the grid)
     7. Folder path where the sample file is located

     Important! x_diff and y_diff are required to build the grid map of each file
     These 2 values will be subtracted from each point to obtain cleaner variance of the values
     Example:
     for x values: 590341, 591572, 591653 -> x_diff = 590000
     for y values: 3591234 3585523, 3564981 -> y_diff = 3500000

     Example:
     cut : 1000 -> z values bigger then 1000 and less then -1000 will be set to mean of all elements"""
    file_list = []

    with open(path) as infile:
        for line in infile:
            line_buffer = line.split()
            file_list.append(
                File(
                    file_name=line_buffer[0],
                    rows=int(line_buffer[1]),
                    x_diff=int(line_buffer[2]),
                    y_diff=int(line_buffer[3]),
                    cut=int(line_buffer[4]),
                    sample_size=int(line_buffer[5]),
                    path=line_buffer[6]
                )
            )

    return file_list


def get_point_clouds(file_path, file_rows):
    X = []
    Y = []
    Z = []

    row_index = 0

    output = np.zeros(shape=(file_rows, 3))

    with open(file_path) as infile:
        for line in infile:
            line_buffer = line.split()
            X.append(float(line_buffer[0]))
            Y.append(float(line_buffer[1]))
            Z.append(float(line_buffer[2]))
            row_index += 1
    output[:, 0] = X
    output[:, 1] = Y
    output[:, 2] = Z
    return output


def save_tile_list(tile_list, file):
    k = 0
    for tile in tile_list:
        save_png(tile, '{}full_simple_images\\{}\\{}_{}.png'.format(file.path, file.file_name, file.file_name, str(k)))
        k += 1


def main_samples():

    files = get_files('config_info_files/file_info2.txt')



    i = 0
    samples = 0
    for file in files:
        # if not os.path.exists('C:\\Users\\adria\\Desktop\\Salton_test\\images\\full_images\\{}'.format(file)):
        #     os.makedirs('C:\\Users\\adria\\Desktop\\Salton_test\\images\\full_images\\{}'.format(file))
        if not os.path.exists('{}full_simple_images\\{}'.format(file.path, file.file_name)):
            os.makedirs('{}full_simple_images\\{}'.format(file.path, file.file_name))
        # Get all the point clouds from the file
        point_clouds = get_point_clouds('{}{}.txt'.format(file.path, file.file_name), file.rows)
        # Make the grid map
        (grid_map, grid_size, null_value) = mapping_to_grid(point_clouds, file.rows, file.x_diff, file.y_diff, file.cut)
        # Get all the tiles from the grid that have useful data
        tile_list = grid_tiling(grid_map, grid_size, file.sample_size, null_value)
        # Noise cancellation in each tile(null values from grid map will be replace with the local tile mean)
        #print(len(tile_list))
        samples += len(tile_list)
        tile_list = noise_cancellation(tile_list, file.sample_size, null_value)
        save_tile_list(tile_list, file)
        i += 1
        print('File({}/{}) done.'.format(i, 30))
    print(samples)


def record_fid():
    with open('fid_records.txt', 'w') as file:

        gaussian_fid_accuracy = get_fid_accuracy(
            path1='C:\\Users\\adria\\Desktop\\simple_images_for_gan\\rescaled_samples\\',
            prefix1='rescaled_sample',
            path2='C:\\Users\\adria\\Desktop\\Licenta\\Samples\\gaussian_samples\\',
            prefix2='gaussian',
            samples_number=1270,
            dim=299
        )
        dense_improvised_accuracy = get_fid_accuracy(
            path1='C:\\Users\\adria\\Desktop\\simple_images_for_gan\\rescaled_samples\\',
            prefix1='rescaled_sample',
            path2='C:\\Users\\adria\\Desktop\\Licenta\\Samples\\dense_improvised_samples\\rescaled_samples\\',
            prefix2='rescaled_sample',
            samples_number=1166,
            dim=299
        )

        dense_real_accuracy = get_fid_accuracy(
            path1='C:\\Users\\adria\\Desktop\\simple_images_for_gan\\rescaled_samples\\',
            prefix1='rescaled_sample',
            path2='C:\\Users\\adria\\Desktop\\Licenta\\Samples\\dense_real_samples\\rescaled_samples\\',
            prefix2='rescaled_sample',
            samples_number=1114,
            dim=299
        )
        conv_fid_accuracy = get_fid_accuracy(
            path1='C:\\Users\\adria\\Desktop\\simple_images_for_gan\\rescaled_samples\\',
            prefix1='rescaled_sample',
            path2='C:\\Users\\adria\\Desktop\\Licenta\\Samples\\conv1_generated_samples\\',
            prefix2='generated_image',
            samples_number=1270,
            dim=299
        )
        conv_structured_fid_accuracy = get_fid_accuracy(
            path1='C:\\Users\\adria\\Desktop\\structured\\rescaled_samples\\',
            prefix1='rescaled_sample',
            path2='C:\\Users\\adria\\Desktop\\Licenta\\Samples\\conv2_structured_samples\\',
            prefix2='generated_image',
            samples_number=2342,
            dim=299
        )
        file.write(
            'Gaussian Surface FID: {0}, Samples: {1}, Sample dimension->({2}, {2})'.
            format(gaussian_fid_accuracy, 1270, 110)
        )
        file.write(
            'Dense GAN Improvised Input FID: {0}, Samples: {1}, Sample dimension->({2}, {2})'.
            format(dense_improvised_accuracy, 1166, 28)
        )
        file.write(
            'Dense GAN Real Input FID: {0}, Samples: {1}, Sample dimension->({2}, {2})'.
            format(dense_real_accuracy, 1114, 28)
        )
        file.write(
            'Convolutional GAN Simple Dataset FID: {0}, Samples: {1}, Sample dimension->({2}, {2})'.
            format(conv_fid_accuracy, 1270, 32)
            )
        file.write(
            'Convolutional GAN Structured Dataset FID: {0}, Samples: {1}, Sample dimension->({2}, {2})'.
            format(conv_structured_fid_accuracy, 2342, 32)
        )

#record_fid()
#main_samples()


#rotate_samples('C:\\Users\\adria\\Desktop\\trees\\', 180)
#rescale_samples('C:\\Users\\adria\\Desktop\\structured\\', 32)
#rescale_samples('C:\\Users\\adria\\Desktop\\simple_and_complex\\', 32)
