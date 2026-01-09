import numpy as np
from helper_fcts import img_to_vec, vec_to_img, show_images, load_image

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
        self.codebook_mean = None
        self.codebook_VT = None
        self.codebook = None
        
        self.K = 20         # Dimensionality of the compressed code
        self.train_count = 100
        self.pixel_count = 96*96

    def set_k(self, K): 
        self.K = K
        pass


    def get_codebook(self):
        """ Codebook contains all information needed for compression/reconstruction """
        #print(f"Getting codebook: \n{self.codebook_VT}")
        return self.codebook
    

    def train(self, train_images):
        """
        Training phase of your algorithm - e.g. here you can perform PCA on training data
        
        Args:
            train_images  ... A list of NumPy arrays.
                              Each array is an image of shape H x W x C, i.e. 96 x 96 x 3
        """
        n = len(train_images)
        self.train_count = n
        self.pixel_count = train_images[0].shape[0]*train_images[0].shape[1]
        m = self.pixel_count

        A = np.zeros((3, n, m)).astype(np.float32)
        mean = np.zeros((3, m)).astype(np.float32)
        U = np.zeros((3, n, n))
        S = np.zeros((3, n))
        VT = np.zeros((3, n, m))

        self.codebook_mean = np.zeros((3, m))
        self.codebook_VT = np.zeros((3, self.K, m))
        self.codebook = {"mean" : self.codebook_mean, "VT" : self.codebook_VT}

        for i, img in enumerate(train_images):
            for c in range(3):
                A[c, i] = img[:,:,c].reshape(-1)
        
        mean = np.mean(A, axis = 1)

        A = A - mean[:,np.newaxis,:]

        U, S, VT = np.linalg.svd(A, full_matrices=False) 

        self.codebook_mean = mean
        self.codebook_VT = VT[:,:self.K,:] + mean[:,np.newaxis,:]
        self.codebook = {"mean" : self.codebook_mean, "VT" : self.codebook_VT}
        """
        Nr = 3
        imgs = np.zeros((Nr+1, 96,96,3)).astype(np.uint8)
        for i in range(Nr):
            img = np.clip(mean + self.codebook_VT[:,i,:]*m, 0, 255)
            imgs[i] = vec_to_img(img)
            #imgs[i] = vec_to_img(A[:,i,:]+mean)
        imgs[Nr] = vec_to_img(mean)
        show_images(imgs)

        print(f"codebook: \n{self.codebook_VT*m}")
        """
        pass
        
        
    def compress(self, test_image):
        X = img_to_vec(test_image)
        test_code = np.zeros((3, self.K))
        X = X - self.codebook_mean
        test_code = np.einsum('cm,ckm->ck', X, self.codebook_VT[:, :self.K, :])

        # print(f"compressed image code: {test_code}")
        return test_code.astype(self.dtype)


class ImageReconstructor:
    """ This class is used on the server to reconstruct images """
    def __init__(self, codebook):
        """ The only information this class may receive is the codebook """
        self.codebook_mean = codebook["mean"]
        self.codebook_VT = codebook["VT"]
        
        self.codebook = {"mean" : self.codebook_mean, "VT" : self.codebook_VT}

        self.K = self.codebook_VT.shape[1]
        self.pixel_count = self.codebook_mean.shape[1]
        
        # print(f"self.codebook_VT.shape: {self.codebook_VT.shape}")
        pass


    def reconstruct(self, test_code):
        """ Given a compressed code of shape K, reconstruct the original image """
        codebook_mean = self.codebook_mean
        codebook_VT = self.codebook_VT

        X = np.zeros((3, self.pixel_count))
        img = np.zeros((96, 96, 3))


        for c in range(3):
            VT = codebook_VT[c]
            mean = codebook_mean[c]
            X[c] = test_code[c]@VT
        
        img = np.clip(vec_to_img(X), 0, 255).astype(np.uint8)
        
        # TODO
        return img