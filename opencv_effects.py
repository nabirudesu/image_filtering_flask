#install opencv using pip install opencv-contrib-python
import cv2
import numpy as np
import matplotlib.image as img
from matplotlib import pyplot as plt
import io
import base64
path='/home/kyoraku/Pictures/Pano_Robley2.jpg'
def oil_painting_effect(img_path):
    img = cv2.imread(img_path)
    res = cv2.xphoto.oilPainting(img, 7, 30)
    return res
def cartoon_effect(img_path):
    img = cv2.imread(img_path)


    #Create Edge Mask
    line_size = 7
    blur_value = 7
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.medianBlur(gray, blur_value)
    edges = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, line_size, blur_value)

    #colour quantization
    #k value determines the number of colours in the image
    total_color = 8
    k=total_color
    # Transform the image
    data = np.float32(img).reshape((-1, 3))
    # Determine criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)
    # Implementing K-Means
    ret, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    result = center[label.flatten()]
    result = result.reshape(img.shape)
    
    #Add blurr effect
    blurred = cv2.bilateralFilter(result, d=1, sigmaColor=250,sigmaSpace=250)
    
    #blurred and edges
    cartoon = cv2.bitwise_and(blurred, blurred, mask=edges)
    cartoon2=cartoon.reshape(img.shape)
    return cartoon


#image=cv2.imread(path)
#resized_image = cv2.resize(image, (700,400), interpolation= cv2.INTER_LINEAR)
#cv2.imshow('original image',resized_image)

#resized_cartoon_effect = cv2.resize(cartoon_effect, (700,400), interpolation= cv2.INTER_LINEAR)
#cv2.imshow('cartoon effect', resized_cartoon_effect)

#resized_oil_effect = cv2.resize(oil_effect, (700,400), interpolation= cv2.INTER_LINEAR)
#cv2.imshow('oil effect', resized_oil_effect)

def image_to_base64(image):
    _, im_arr = cv2.imencode('.jpeg', image)  # im_arr: image in Numpy one-dim array format.
    im_bytes = im_arr.tobytes()
    im_b64 = base64.b64encode(im_bytes)
    img_str = im_b64.decode('utf-8')
    return img_str

def return_result(type,path):
    if type=='cartoon':
        cart_effect=cartoon_effect(path)
        crt=image_to_base64(cart_effect)
        return crt
    if type == 'oil':
        oil_effect=oil_painting_effect(path)
        oil=image_to_base64(oil_effect)
        return oil