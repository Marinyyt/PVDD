import cv2

__all__ = ['filter2D']


def filter2D(img, kernel):
    """cv2.filter2D
    Args:
        img: (h, w, c), type: float32, 0-1
        kernel: (b, k, k)
    """
    img_blur = cv2.filter2D(img, ddepth=-1, kernel=kernel)
    return img_blur.clip(0, 1)
