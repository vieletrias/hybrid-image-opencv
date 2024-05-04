#Laviele Trias
#Lab02 - CMC 174
import sys
import cv2
import numpy as np
def cross_correlation(img, kernel):
    print("This cross-correlation")
    #This gets the image's and the kernel's dimensions for further calculation 
    img_height, img_width = img.shape[:2]
    kernel_height, kernel_width = kernel.shape

    # Padding
    #Centering so that there is equal amount of padding on all sides
    vpadding = (kernel_height - 1) // 2
    hpadding = (kernel_width - 1) // 2

    #Adds image into new array with the added vertical and horizontal padding
    image_padded = np.zeros((img_height + 2 * vpadding, img_width + 2 * hpadding, img.shape[2]))

    #For loop to copy the images to account for padding 
    for i in range(img_height):
        for j in range(img_width):
            for c in range(img.shape[2]):
                image_padded[i + vpadding, j + hpadding, c] = img[i, j, c]

    # Cross-correlation
    #Store result of cross cor
    result = np.zeros_like(img)
    #Iterates over channels, rows and columns
    for c in range(img.shape[2]):
        for i in range(img_height):
            for j in range(img_width):
                window = image_padded[i:i + kernel_height, j:j + kernel_width, c]
                #Checks if shape of window is same as kernel 
                if window.shape == kernel.shape:
                    #sums product of window and kernel
                    result[i, j, c] = np.sum(window * kernel)
                else:
                    result[i, j, c] = 0


    #print(result)
    return result


def convolution(img, kernel):
    # Use cross_correlation to carry out a 2D convolution
    #Image is flipped before passing cross-correlation
    print("This convolve")
    return cross_correlation(img, np.flipud(np.fliplr(kernel)))

def gaussian_kernel(sigma, size):
    print("This gaussian blur")
    center = size // 2
    kernel = np.zeros((size, size))

    for i in range(size):
        for j in range(size):
            kernel[i, j] = (1 / (2 * np.pi * sigma ** 2)) * np.exp(-((i - center) ** 2 + (j - center) ** 2) / (2 * sigma ** 2))
    kernel = kernel / np.sum(kernel)  # Normalize values so that the sum is 1.0
    #print(kernel)
    return kernel

def low_pass(img, sigma, size):
    # Filter the image as if it's filtered with a low pass filter
    #use gaussian blur
    kernel = gaussian_kernel(sigma, size)
    print("This low pass")
    #Then use correlation
    #print(cross_correlation(img, kernel))
    #return values
    return cross_correlation(img, kernel)

def high_pass(img, sigma, size):
    # Filter the image as if it's filtered with a high pass filter
    print("This high pass")
    #Use low pass then subtract it with original image
    low_pass_img = low_pass(img, sigma, size)
    #print(img - low_pass_img)
    return img - low_pass_img


def create_hybrid_image(img1, img2, sigma1, size1, high_low1, sigma2, size2, high_low2, mixin_ratio, scale_factor):
    high_low1 = high_low1.lower()
    high_low2 = high_low2.lower()

    if img1.dtype == np.uint8:
        #Convert image from int to float and divide by 255.0
        img1 = img1.astype(np.float32) / 255.0
        img2 = img2.astype(np.float32) / 255.0

    #If low, apply low pass filter; else apply high pass filter
    if high_low1 == 'low':
        img1 = low_pass(img1, sigma1, size1)
    else:
        img1 = high_pass(img1, sigma1, size1)

    #If low, apply low pass filter; else apply high pass filter
    if high_low2 == 'low':
        img2 = low_pass(img2, sigma2, size2)
    else:
        img2 = high_pass(img2, sigma2, size2)

    #image intensity adjustment according to mix in ration
    img1 *= (1 - mixin_ratio)
    img2 *= mixin_ratio

    #Adjusted scaling result acc to scale factor variable
    hybrid_img = (img1 + img2) * scale_factor

    
    return (hybrid_img * 255).clip(0, 255).astype(np.uint8)

def main():
    # Read the images using OpenCV
    image1 = cv2.imread("Trias_lab02_left.jpg")
    image2 = cv2.imread("Trias_lab02_right.jpg")

    image1 = cv2.resize(image1, (500, 500))
    image2 = cv2.resize(image2,(500,500))
    
    #Sigma => Higher = blurrier results 
    #Size => Dimensions of the kernel filter
    #mixin => image contribution to hybrid image
    #scale factor => overall brightness
    
    #Dog - Cat Parameters
    sigma1 = 10
    size1 = 40
    high_low1 = 'low'
    sigma2 = 10
    size2 = 7
    high_low2 = 'high'
    mixin_ratio = 0.7
    scale_factor = 1.7

    #For Einstein - Marilyn Parameters
    # sigma1 = 20
    # size1 = 25
    # high_low1 = 'low'
    # sigma2 = 5
    # size2 = 7
    # high_low2 = 'high'
    # mixin_ratio = 0.7
    # scale_factor = 1.7

    # Create a hybrid image
    hybrid_image = create_hybrid_image(image1, image2, sigma1, size1, high_low1, sigma2, size2, high_low2, mixin_ratio, scale_factor)

    # Display the images
    # cv2.imshow('Image 1', image1)
    # cv2.imshow('Image 2', image2)
    cv2.imwrite('Trias_lab02_hybrid.png', hybrid_image)
    cv2.imshow('Hybrid Image', hybrid_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

main()