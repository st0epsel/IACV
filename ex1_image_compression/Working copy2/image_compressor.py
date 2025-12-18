import numpy as np
from helper_fcts import show_images

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
        self.codebook = None
        
        self.K = 19         # Dimensionality of the compressed code
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
        VT = np.zeros((3, n, m))

        self.codebook = np.zeros((3, self.K, m))

        for i, img in enumerate(train_images):
            for c in range(3):
                A[c, i] = img[:,:,c].reshape(-1)

        mean = np.mean(A, axis = 1)
        median = np.median(A, axis = 1)
        print(f"\nmean: \n{mean}")
        floored_mask = np.floor(mean/240)*255

        final_mask = np.zeros((3, self.pixel_count))
        final_mask[0] = np.min(floored_mask, axis = 0)

        show_images([self.vec_to_img(floored_mask),
                     self.vec_to_img(final_mask)])

        """project_dir = Path(__file__).resolve().parent  # folder where this script is located
        filepath = project_dir / "mask.txt"

        np.savetxt(filepath, array, fmt="%.6f")  # choose format as needed
        print(f"✅ Saved array to: {filepath}")"""

        U, S, VT = np.linalg.svd(A, full_matrices=False) 

        self.codebook = VT[:,:self.K,:]

        Nr = 7
        imgs = np.zeros((Nr, 96,96,3)).astype(np.uint8)
        for i in range(Nr):
            img = np.clip(self.codebook[:,i+12,:]*m, 0, 255)
            imgs[i] = self.vec_to_img(img)
            #imgs[i] = vec_to_img(A[:,i,:]+mean)
        #show_images(imgs)
        pass
        
        
    def compress(self, test_image):
        X = self.img_to_vec(test_image)
        test_code = np.zeros((3, self.K))
        test_code = np.einsum('cm,ckm->ck', X, self.codebook[:, :self.K, :])

        # print(f"compressed image code: {test_code}")
        return test_code.astype(self.dtype)


    def img_to_vec(self, img):
        vec = np.zeros((3, 96*96))
        for c in range(3):
            vec[c] = img[:,:,c].reshape(-1)
        return vec
    
    def vec_to_img(self, vec):
        img = np.zeros((96,96,3))
        for c in range(3):
            img[:,:,c] = vec[c].reshape(96,96)
        return img.astype(np.uint8)



class ImageReconstructor:
    """ This class is used on the server to reconstruct images """
    def __init__(self, codebook):
        """ The only information this class may receive is the codebook """

        self.codebook = codebook
        self.K = self.codebook.shape[1]
        self.pixel_count = self.codebook.shape[2]
        
        # print(f"self.codebook_VT.shape: {self.codebook_VT.shape}")
        pass


    def reconstruct(self, test_code):
        """ Given a compressed code of shape K, reconstruct the original image """
        codebook = self.codebook

        X = np.einsum('ck,ckm->cm', test_code, codebook)
        img = np.clip(self.vec_to_img(X), 0, 255).astype(np.uint8)

        return img

        
    def vec_to_img(self, vec):
        img = np.zeros((96,96,3))
        for c in range(3):
            img[:,:,c] = vec[c].reshape(96,96)
        return img