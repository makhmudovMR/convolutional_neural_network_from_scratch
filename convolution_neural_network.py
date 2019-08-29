import numpy as np
import mnist


def load_mnist():
    train_images = mnist.train_images()
    train_labels = mnist.train_labels()

    test_images = mnist.test_images()
    test_labels = mnist.test_labels()

    return ((train_images, train_labels), (test_images, test_labels))

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

class ConvolutionNeuralNetwork:

    def __init__(self):
        
        self.W1_conv = np.random.randn(3,3,1,32) * 0.01
        self.W2_conv = np.random.randn(3,3,32,64) * 0.01
        self.W1_fc = np.random.randn(120, 5*5*64) * 0.01
        self.W2_fc = np.random.randn(10, 120) * 0.01

        self.stride = 1
        self.pad = 0

        self.max_pool_stride = 2
        self.max_pool_pad = 0
        self.max_pool_f = 2

    def conv_single_step(self, a_slice, W):
        s = np.multiply(a_slice, W)
        Z = np.sum(s)
        # Z = Z + b
        return Z

    def create_mask_from_window(self, x):
        return x == np.max(x)

    def max_pool(self, Z):
        (z_h, z_w, z_c) = Z.shape
        
        nz_h = int(1+(z_h - self.max_pool_f) / self.max_pool_stride)
        nz_w = int(1+(z_w - self.max_pool_f) / self.max_pool_stride)
        nz_c = z_c

        nZ = np.zeros((nz_h, nz_w, nz_c))

        for h in range(nz_h):
            for w in range(nz_w):
                for c in range(nz_c):
                    vert_start = h * self.max_pool_stride
                    vert_end = vert_start + self.max_pool_f
                    horiz_start = w * self.max_pool_stride
                    horiz_end = horiz_start + self.max_pool_f
                    z_slice = Z[vert_start:vert_end, horiz_start:horiz_end, c]
                    nZ[h, w, c] = np.max(z_slice)
        return nZ

    def forward(self, image):
        (img_h, img_w) = image.shape
        (W1_f, W1_f, imc_c, W1_c) = self.W1_conv.shape # img_c == W1_c_prev

        Z1_h = int((img_h - W1_f + 2 * self.pad) / self.stride) + 1
        Z1_w = int((img_w - W1_f + 2 * self.pad) / self.stride) + 1
        Z1 = np.zeros((Z1_h, Z1_w, W1_c))

        for h in range(Z1_h):
            for w in range(Z1_w):
                for c in range(W1_c):
                    vert_start = h * self.stride
                    vert_end = vert_start + W1_f
                    horiz_start = w * self.stride
                    horiz_end = horiz_start + W1_f

                    img_slice = image[vert_start:vert_end, horiz_start:horiz_end]

                    Z1[h, w, c] = self.conv_single_step(img_slice, self.W1_conv[:, :, :, c])

        print(Z1.shape)
        Z1 = self.max_pool(Z1)
        print(Z1.shape)
        print('''----''')
        z1_h, z1_w, z1_c = Z1.shape
        (W2_f, W2_f, z1_c, W2_c) = self.W2_conv.shape
        Z2_h = int((z1_h - W2_f + 2 * self.pad) / self.stride) + 1
        Z2_w = int((z1_w - W2_f + 2 * self.pad) / self.stride) + 1
        Z2 = np.zeros((Z2_h, Z2_w, W2_c))

        for h in range(Z2_h):
            for w in range(Z2_w):
                for c in range(W2_c):
                    vert_start = h * self.stride
                    vert_end = vert_start + W2_f
                    horiz_start = w * self.stride
                    horiz_end = horiz_start + W2_f

                    z1_slice = Z1[vert_start:vert_end, horiz_start:horiz_end, :]
                    Z2[h, w, c] = self.conv_single_step(z1_slice, self.W2_conv[:, :, :, c])

        # print(Z2.shape)
        Z2 = self.max_pool(Z2)
        # print(Z2.shape)

        Afc_1 = Z2.reshape(5*5*64, 1)
        Zfc_1 = np.dot(self.W1_fc, Afc_1)
        # calculate the signals emerging from hidden layer
        Afc_2 = sigmoid(Zfc_1)

        Zfc_2 = np.dot(self.W2_fc, Afc_2)
        out = sigmoid(Zfc_2)
        return out

    def train(self, image, target):
        '''Layer 1'''
        (img_h, img_w) = image.shape
        (W1_f, W1_f, imc_c, W1_c) = self.W1_conv.shape # img_c == W1_c_prev

        Z1_h = int((img_h - W1_f + 2 * self.pad) / self.stride) + 1
        Z1_w = int((img_w - W1_f + 2 * self.pad) / self.stride) + 1
        Z1 = np.zeros((Z1_h, Z1_w, W1_c))

        for h in range(Z1_h):
            for w in range(Z1_w):
                for c in range(W1_c):
                    vert_start = h * self.stride
                    vert_end = vert_start + W1_f
                    horiz_start = w * self.stride
                    horiz_end = horiz_start + W1_f

                    img_slice = image[vert_start:vert_end, horiz_start:horiz_end]

                    Z1[h, w, c] = self.conv_single_step(img_slice, self.W1_conv[:, :, :, c])

        Z1_pool = self.max_pool(Z1)
        '''Layer 1'''
        '''Layer 2'''
        z1_h, z1_w, z1_c = Z1_pool.shape
        (W2_f, W2_f, z1_c, W2_c) = self.W2_conv.shape
        Z2_h = int((z1_h - W2_f + 2 * self.pad) / self.stride) + 1
        Z2_w = int((z1_w - W2_f + 2 * self.pad) / self.stride) + 1
        Z2 = np.zeros((Z2_h, Z2_w, W2_c))

        for h in range(Z2_h):
            for w in range(Z2_w):
                for c in range(W2_c):
                    vert_start = h * self.stride
                    vert_end = vert_start + W2_f
                    horiz_start = w * self.stride
                    horiz_end = horiz_start + W2_f

                    z1_slice = Z1_pool[vert_start:vert_end, horiz_start:horiz_end, :]
                    Z2[h, w, c] = self.conv_single_step(z1_slice, self.W2_conv[:, :, :, c])

        # print(Z2.shape)
        Z2_pool = self.max_pool(Z2)
        # print(Z2.shape)
        '''Layer 2'''

        Afc_1 = Z2_pool.reshape(5*5*64, 1)
        Zfc_1 = np.dot(self.W1_fc, Afc_1)
        # calculate the signals emerging from hidden layer
        Afc_2 = sigmoid(Zfc_1)

        Zfc_2 = np.dot(self.W2_fc, Afc_2)
        out = sigmoid(Zfc_2)

        '''Backprop'''
        
        delta = (out - target)

        delta_W2_fc = np.dot(delta, Afc_2.T)
        delta = np.dot(self.W2_fc.T, delta)

        delta_W1_fc = np.dot(delta, Afc_1.T)
        delta = np.dot(self.W1_fc.T, delta)

        delta = delta.reshape(5, 5, 64)

        n_H, n_W, n_C = delta.shape
        dZ2 = np.zeros(Z2.shape)

        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    vert_start = h
                    vert_end = vert_start + self.max_pool_f
                    horiz_start = w 
                    horiz_end = horiz_start + self.max_pool_f

                    Z2_slice = Z2[vert_start:vert_end, horiz_start:horiz_end, c]
                    mask = self.create_mask_from_window(Z2_slice)
                    dZ2[vert_start:vert_end, horiz_start:horiz_end, c] = mask * delta[h, w, c]

        # print(dZ2.shape)

        n_H, n_W, n_C = dZ2.shape

        dZ1_pool = np.zeros(Z1_pool.shape)
        dW2_conv = np.zeros(self.W2_conv.shape)
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    vert_start = h
                    vert_end = vert_start + W2_f
                    horiz_start = w 
                    horiz_end = horiz_start + W2_f

                    z1_pool_slice = Z1_pool[vert_start:vert_end, horiz_start:horiz_end, :]
                    dZ1_pool[vert_start:vert_end, horiz_start:horiz_end, :] += self.W2_conv[:, :, :, c] * dZ2[h, w, c]
                    dW2_conv[:, :, :, c] += z1_pool_slice * dZ2[h, w, c]
        # print(dZ1_pool.shape)



        n_H, n_W, n_C = dZ1_pool.shape
        dZ1 = np.zeros(Z1.shape)

        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    vert_start = h
                    vert_end = vert_start + self.max_pool_f
                    horiz_start = w 
                    horiz_end = horiz_start + self.max_pool_f

                    Z1_slice = Z1[vert_start:vert_end, horiz_start:horiz_end, c]
                    mask = self.create_mask_from_window(Z1_slice)
                    dZ1[vert_start:vert_end, horiz_start:horiz_end, c] = mask * dZ1_pool[h, w, c]

        # print(dZ1.shape)

        n_H, n_W, n_C = dZ1.shape

        dW1_conv = np.zeros(self.W1_conv.shape)

        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    vert_start = h
                    vert_end = vert_start + W1_f
                    horiz_start = w 
                    horiz_end = horiz_start + W1_f

                    image_slice = image[vert_start:vert_end, horiz_start:horiz_end]
                    image_slice = image_slice.reshape(3,3,1)
                    # dZ1_pool[vert_start:vert_end, horiz_start:horiz_end, :] += self.W2_conv[:, :, :, c] * dZ2[h, w, c]
                    dW1_conv[:, :, :, c] += image_slice * dZ1[h, w, c]


        # print(self.W1_conv.shape, dW1_conv.shape)
        # print(self.W2_conv.shape, dW2_conv.shape)
        self.W2_fc = self.W2_fc - 0.01 * delta_W2_fc
        self.W1_fc = self.W1_fc - 0.01 * delta_W1_fc
        self.W2_conv = self.W2_conv - 0.01 * dW2_conv
        self.W1_conv = self.W1_conv - 0.01 * dW1_conv

        
        


        


def test():
    ((train_images, train_labels), (test_images, test_labels)) = load_mnist()
    cnn = ConvolutionNeuralNetwork()
    image = train_images[1]
    target = np.zeros((10, 1))
    target[train_labels[1]][0] += .99
    print(cnn.forward(image))
    for _ in range(15):
        cnn.train(image, target)
    print(cnn.forward(image))
    print(target)
    
    
    
                    
        
test()