import numpy as np

class ImageCompressor:
    """
      This class is responsible to
          1. Learn the codebook given the training images
          2. Compress an input image using the learnt codebook
    """
    def __init__(self):
        """
        Feel free to add any number of parameters here.
        But be sure to set default values. Those will be used on the evaluation server
        """
        # Here you can set some parameters of your algorithm, e.g.
        self.dtype = np.float16

        self.K = 19         # Dimensionality of the compressed code
        self.train_count = 100
        self.pixel_count = 96*96*3
        self.codebook = None
        self.mask = None
        self.vec_mask = None
        self.masked_pixel_count = self.pixel_count
        self.mean = None

    def set_k(self, k):
        self.K = k
        pass

    def get_codebook(self):
        """ Codebook contains all information needed for compression/reconstruction """
        return self.codebook

    def train(self, train_images):

        self.train_count = len(train_images)
        self.pixel_count = train_images[0].shape[0]*train_images[0].shape[1]*3
        # print(f"self.pixel_count: {self.pixel_count}")

        VT = np.zeros((self.train_count, self.pixel_count * 3))

        mask = self.get_mask(train_images)
        self.mask = mask
        vec_mask = self.vectorize_mask(mask)
        self.vec_mask = vec_mask
        self.masked_pixel_count = vec_mask.shape[0]*3
        # print(f"masked pixel count: {self.masked_pixel_count/3}")
        # print(f"vec_mask.shape: {vec_mask.shape}\nself.vec_mask: {vec_mask}")

        A = np.array(train_images).reshape((100, int(self.pixel_count/3), 3))
        Acomp = self.vec_mask_compression(A, self.vec_mask).reshape((100, self.masked_pixel_count))

        mean_comp = np.mean(Acomp, axis=0)

        Acomp -= mean_comp

        Ucomp, Scomp, VTcomp = np.linalg.svd(Acomp, full_matrices=False)
        VTcomp = np.array(VTcomp)
        # print(f"VTcomp.shape: {VTcomp.shape}")

        self.codebook = np.array((self.K + 2, self.masked_pixel_count))

        codebook = VTcomp[:self.K+2, :].reshape((self.K + 2, self.masked_pixel_count))
        codebook *= self.masked_pixel_count
        # print(f"max(codebook): {np.max(codebook)}")

        self.codebook = codebook.astype(np.int16)
        # print(f"type(codebook): {type(self.codebook[0, 0])}")
        # print(f"codebook.shape: {self.codebook.shape}\ncodebook type: {type(self.codebook[0, 0, 0])}")
        self.codebook[self.K, :] = np.zeros(self.codebook.shape[1:2])
        self.codebook[self.K, 0:int(self.masked_pixel_count/3)] = self.vec_mask
        self.codebook[self.K+1, :] = mean_comp
        self.mean = mean_comp
        # print(f"self.codebook[K,:,0]: {self.codebook[self.K, :, 0]}")

        """
        # Use for visualizing eigenimages
        U, S, VT = zip(*(np.linalg.svd(A[:, :, c], full_matrices=False)for c in range(3)))
        VT = np.array(VT)
        VT = np.stack([VT[c][:self.K, :] for c in range(3)], axis=2)

        Nr = 7
        imgs = np.zeros((Nr, 96, 96, 3)).astype(np.uint8)
        for i in range(Nr):
            img = np.clip(VT[i, :, :]*self.pixel_count/3, 0, 255)
            imgs[i] = img.reshape((96, 96, 3))
        show_images(imgs)
        """
        pass

    def compress(self, test_image):
        codebook = self.codebook / self.masked_pixel_count
        X = test_image.reshape((96*96*3))
        X = self.vec_mask_compression(X[None, :], self.vec_mask)[0]
        test_code = np.sum(codebook[:self.K, :] * (X-self.mean)[None, :], axis=1).astype(np.int16)
        # print(f"compressed image code: {test_code}\ncode type: {type(test_code[0])}")
        return test_code

    def get_mask(self, train_images):
        """
        In: train images of dimensions k, 96*96, 3
        Out: boolean mask of dimensions 96*96
        """

        A = np.zeros((self.train_count, int(self.pixel_count/3), 3)).astype(np.float32)
        for i, img in enumerate(train_images):
            A[i] = img.reshape((9216, 3))
        mask = ~np.min(np.floor(np.mean(A, axis=0) / 240), axis=1).astype(bool)
        # print(f"floored mask: {mask}")
        return mask

    def vectorize_mask(self, mask):
        """
        In: boolean mask of dims 96*96
        Out: a vector containing all the pixel coordinates affected by the filter
        vectorized_mask contains all the indices of nonzero mask pixels
        """
        filtered_img_size = np.sum(mask)
        vect_mask = np.zeros(filtered_img_size, dtype=np.uint16)
        vect_mask_pos = 0
        for px_index in range(mask.shape[0]):
            if mask[px_index]:
                vect_mask[vect_mask_pos] = int(px_index)
                vect_mask_pos += 1
        # print(f"vect_mask_infuntion: {vect_mask}\nvec_mask_dimension: {vect_mask.shape}")
        return vect_mask

    def vec_mask_compression(self, imgs_uncomp, vec_mask):
        """
        In:     imgs with shape k, 9216 * 3 (get transformed to k, 9216, 3),
                vec mask with shape v
        Out:    imgs_compressed with dims k, len(vec_mask) * 3
        """
        imgs_count = imgs_uncomp.shape[0]
        imgs = imgs_uncomp.reshape((imgs_count, int(self.pixel_count/3), 3))
        filtered_vec_size = vec_mask.shape[0]
        # print(f"vec_mask: {vec_mask}")
        imgs_compressed = np.zeros((imgs_count, filtered_vec_size, 3))
        for image in range(imgs.shape[0]):
            # print(f"image: {image}")
            for px_pos in range(filtered_vec_size):
                # print(f"px_pos: {px_pos},   vec_mask[px_pos]: {vec_mask[px_pos]}")
                imgs_compressed[image, px_pos] = imgs[image, vec_mask[px_pos]]
        return imgs_compressed.reshape((imgs_count, self.masked_pixel_count))


class ImageReconstructor:
    """ This class is used on the server to reconstruct images """
    def __init__(self, codebook):
        """ The only information this class may receive is the codebook """
        self.codebook = codebook[0:-2]
        self.K = self.codebook.shape[0]
        self.pixel_count = 96 * 96 * 3
        self.vec_mask = codebook[self.K, :].astype(np.uint16)
        self.masked_pixel_count = self.vec_mask.shape[0]
        self.mean = codebook[self.K+1, :].astype(np.uint16)

        # print(f"self.codebook_VT.shape: {self.codebook_VT.shape}")
        pass

    def reconstruct(self, test_code):
        """
        In: test code of shape K
        Given: codebook
        Out: image
        """
        codebook = self.codebook / self.masked_pixel_count
        vec_mask = self.vec_mask
        X = test_code @ codebook + self.mean
        img_compressed = np.clip(X.reshape(self.masked_pixel_count), 0, 255).astype(np.uint8)
        img = self.vec_mask_decompression(np.array([img_compressed]), vec_mask)[0]
        return img.reshape((96, 96, 3)).astype(np.uint8)

    def vec_mask_decompression(self, imgs_comp, vec_mask):
        """
        In      imgs_compressed with shape i, self.masked_pixel_count
                vec_mask with shape masked_pixel_dim
        Out     imgs decompressed with shape i, self.pixel_count
        """
        img_count = imgs_comp.shape[0]
        imgs_compressed = imgs_comp.reshape((img_count, int(self.masked_pixel_count/3), 3))
        imgs_decompressed = np.ones((img_count, int(self.pixel_count/3), 3))*255
        for image in range(img_count):
            for vec_pos in range(int(vec_mask.shape[0]/3)):
                imgs_decompressed[image, vec_mask[vec_pos]] = imgs_compressed[image, vec_pos]
        return imgs_decompressed

