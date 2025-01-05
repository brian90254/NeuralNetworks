import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

def quantimage(image, k, output_filename):
    i = np.float32(image).reshape(-1, 3)
    condition = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    ret, label, center = cv2.kmeans(i, k, None, condition, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    final_img = center[label.flatten()]
    final_img = final_img.reshape(image.shape)
    cv2.imwrite(output_filename, final_img)
    return final_img

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python 250104_ImageQuantization_6.py <input_filename>")
        sys.exit(1)

    input_filename = sys.argv[1]
    output_filename = input_filename.split('.')[0] + '_Quant_1.png'
    image = cv2.imread(input_filename)
    plt.imshow(quantimage(image, 3, output_filename))
    #plt.imshow(quantimage(image, 3))
    plt.show()