
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.tri as mtri
from stl import mesh


class EnvironmentGenerator:
    def __init__(self, dim_x, dim_y, scale):
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.scale = scale

        self.X = 0
        self.Y = 0
        self.Z = 0
        self.random_array = 0
        self.output = None

#     Random Gauss Surface
    def generate(self):
        # self.x Size in pixels of the x axis on the grid
        # self.y Size in pixels of the x axis on the grid
        N = [self.dim_x, self.dim_y]
        self.random_array = np.random.randn(N[0], N[1])


        self.fourier_surface(self, self.random_array)

        #self.add_peak(self, 30, 30, 10)


        self.add_square(self, 30, 30, 7)
        #self.add_peak(self, 15, 15, 20)

        self.Z *= self.scale
        print(self.Z, type(self.Z))
        # Remember that Gazebo uses ENU (east-north-up) convention, so underwater
        # the Z coordinate will be negative
        #Z -= 3
        # Note: Gazebo will import your mesh in meters.

        # Flatten our interpolated data for triangulation
        self.output = np.zeros(shape=(self.X.size, 3))
        self.output[:, 0] = self.X.flatten()
        self.output[:, 1] = self.Y.flatten()
        self.output[:, 2] = self.Z.flatten()

        #self.create_3d_object(self, self.output, 'seabed1.stl')

        #self.show_surface(self, self.output)

    # Show the resulting heightmap as an image
    @staticmethod
    def show_image(self, z_grid):
        plt.figure(figsize=(6, 6))
        plt.show(z_grid)
        plt.show()

    @staticmethod
    def save_png(self, z_grid, name):
        plt.figure()

        plt.imshow(z_grid)
        plt.axis('off')
        plt.savefig(name, bbox_inches='tight', pad_inches= 0)
        plt.close()

    @staticmethod
    def fourier_surface(self, random):
        F = 2  # frequency-filter width
        [X, Y] = np.mgrid[1:self.dim_x + 1, 1:self.dim_y + 1]
        i = np.minimum(X - 1, self.dim_x - X + 1)
        j = np.minimum(Y - 1, self.dim_y - Y + 1)

        H = np.exp(-0.5 * (np.multiply(i, i) + np.multiply(j, j)) / F ** 2)

        fast_fourier = np.fft.fft2(random)
        inverse_fast_fourier = np.fft.ifft2(np.multiply(H, fast_fourier))
        Z = np.real(inverse_fast_fourier)
        [self.X, self.Y, self.Z] = [X, Y, Z]

    @staticmethod
    def show_surface(self, output):
        # Triangulation of the interpolated data
        tri = mtri.Triangulation(output[:, 0], output[:, 1])

        # Show the resulting surface
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_trisurf(tri, output[:, 2], cmap=plt.cm.CMRmap, shade=True, linewidth=0.1)
        plt.show()

    @staticmethod
    def add_peak(self, cx, cy, r):
        x = np.arange(0, self.dim_x)
        y = np.arange(0, self.dim_y)

        a = reversed(range(2, r+1))
        for index in a:
            mask = (x[np.newaxis, :] - cx) ** 2 + (y[:, np.newaxis] - cy) ** 2 < index ** 2
            self.Z[mask] += 0.06*index/(r+1)

    @staticmethod
    def add_square(self, cx, cy, length):
        x = np.arange(0, self.dim_x)
        y = np.arange(0, self.dim_y)

        start_x = int(cx-length/2)
        start_y = int(cy-length/2)

        self.Z[start_x:start_x+length, start_y:start_y+length] = np.amax(self.Z)*1.5#(np.amax(self.Z))**3/(0.01*self.dim_x)

    @staticmethod
    def create_3d_object(self, output, filename):

        # Triangulation of the interpolated data
        tri = mtri.Triangulation(output[:, 0], output[:, 1])

        # Create the mesh object
        seabed_mesh = mesh.Mesh(np.zeros(tri.triangles.shape[0], dtype=mesh.Mesh.dtype))

        # Set the vectors
        for i, f in enumerate(tri.triangles):
            for j in range(3):
                seabed_mesh.vectors[i][j] = output[f[j]]

        # Store the seabed as a STL file
        seabed_mesh.save(filename)

    def generate_samples(self, sample_number):
        # self.x Size in pixels of the x axis on the grid
        # self.y Size in pixels of the x axis on the grid
        N = [self.dim_x, self.dim_y]

        for i in range(sample_number):
            self.random_array = np.random.randn(N[0], N[1])
            self.fourier_surface(self, self.random_array)
            #self.add_square(self, 30, 30, 10)
            self.Z *= self.scale
            self.save_png(self, self.Z,'figures/fig{}'.format(i))
            if i% 200==0:
                print(i)
            # Flatten our interpolated data for triangulation
            # self.output = np.zeros(shape=(self.X.size, 3))
            # self.output[:, 0] = self.X.flatten()
            # self.output[:, 1] = self.Y.flatten()
            # self.output[:, 2] = self.Z.flatten()

            #self.create_3d_object(self, self.output, 'seabed_sample#'+str(i)+'.stl')
    def save_sample(self, zi, name):
        self.save_png(self, zi, name)
