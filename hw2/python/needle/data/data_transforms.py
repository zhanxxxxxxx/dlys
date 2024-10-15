import numpy as np

class Transform:
    def __call__(self, x):
        raise NotImplementedError


class RandomFlipHorizontal(Transform):
    def __init__(self, p = 0.5):
        self.p = p

    def __call__(self, img):
        """
        Horizonally flip an image, specified as an H x W x C NDArray.
        Args:
            img: H x W x C NDArray of an image
        Returns:
            H x W x C ndarray corresponding to image flipped with probability self.p
        Note: use the provided code to provide randomness, for easier testing
        """
        flip_img = np.random.rand() < self.p
        ### BEGIN YOUR SOLUTION
        if flip_img:
          img = np.flip(img, axis=1) 
        return img
        
        ### END YOUR SOLUTION


#对图像进行随机裁剪
class RandomCrop(Transform):
    def __init__(self, padding=3):
        self.padding = padding

    def __call__(self, img):
        """ Zero pad and then randomly crop an image.
        Args:
             img: H x W x C NDArray of an image
        Return 
            H x W x C NAArray of cliped image
        Note: generate the image shifted by shift_x, shift_y specified below
        """
        # Get original dimensions
        h, w, c = img.shape
        
        # Apply zero padding
        padded_img = np.pad(img, ((self.padding, self.padding), (self.padding, self.padding), (0, 0)), mode='constant')
        
        # Calculate random shifts
        shift_x, shift_y = np.random.randint(low=-self.padding, high=self.padding + 1, size=2)
        
        # Calculate the starting coordinates for cropping
        start_x = self.padding + shift_x
        start_y = self.padding + shift_y
        
        # Crop the image
        cropped_img = padded_img[start_x:start_x + h, start_y:start_y + w]
        # img = img[self.padding + shift_x:self.padding + shift_x + img_size[0], self.padding + shift_y:self.padding + shift_y + img_size[1], :]
        return cropped_img

        
