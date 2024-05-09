import numpy as np
import cv2
import matplotlib.pyplot as plt
import imutils
from imutils.perspective import four_point_transform


def resizer(image,width=500):
    # get widht and height of the image
    h,w,c = image.shape
    
    height = int((h/w)* width )
    size = (width,height)
    image = cv2.resize(image,(width,height))
    return image, size

def apply_brightness_contrast(input_img, brightness = 0, contrast = 0):
    
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow)/255
        gamma_b = shadow
        
        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()
    
    if contrast != 0:
        f = 131*(contrast + 127)/(127*(131-contrast))
        alpha_c = f
        gamma_c = 127*(1-f)
        
        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf

def document_scanner(image):
    orig = image.copy()
    image = np.pad(image,((50,50),(50,50),(0,0)),mode='constant',
    constant_values=0)
    img_re,size = resizer(image)

    try:
        detail = cv2.detailEnhance(img_re,sigma_s = 60, sigma_r = 0.5)
        gray = cv2.cvtColor(detail,cv2.COLOR_BGR2GRAY) # GRAYSCALE IMAGE
        bfilter = cv2.bilateralFilter(gray, 20, 17, 17) # NOISE REDUCTION
        blur = cv2.GaussianBlur(bfilter,(5,5),0)
        ret, thresh = cv2.threshold(blur,200,255,cv2.THRESH_BINARY)
    
        # edge detect
        edge_image = cv2.Canny(thresh,75,200)
        # morphological transform
        kernel = np.ones((5,5),np.uint8)
        dilate = cv2.dilate(edge_image,kernel,iterations=1)
        closing = cv2.morphologyEx(dilate,cv2.MORPH_CLOSE,kernel, iterations=3)

        # find the contours
        contours , hire = cv2.findContours(closing,
                                       cv2.RETR_LIST,
                                       cv2.CHAIN_APPROX_SIMPLE)

        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        for contour in contours:
            peri = cv2.arcLength(contour,True)
            approx = cv2.approxPolyDP(contour,0.03*peri, True)

            if len(approx) == 4:
                four_points = np.squeeze(approx)
                break

        cv2.drawContours(img_re,[four_points],-1,(0,255,0),3)
        

        if cv2.contourArea(four_points) < (img_re.shape[0] * img_re.shape[1])/5:
            four_points = np.array([[0,0],[size[0],0],[size[0],size[1]],[0,size[1]]])

    except:
        four_points = np.array([[0,0],[size[0],0],[size[0],size[1]],[0,size[1]]])

 
    img_re = img_re[50:-50,50:-50,:] # remove padding

    # find four points for original image
    multiplier = image.shape[1] / size[0]
    four_points_orig = four_points * multiplier
    four_points_orig = four_points_orig.astype(int)
    # print(four_points_orig)

    wrap_image = four_point_transform(image,four_points_orig)
    
    # if wrap_image > orig then 
    if wrap_image.shape[0] > orig.shape[0] and wrap_image.shape[1] > orig.shape[1]: 
        wrap_image = wrap_image[50:-50,50:-50,:]
        
    return wrap_image

def add_performance(wrpimg):
    cnt = 0
    # add bright till the image is light
    x = wrpimg
    while np.mean(x) < 127:
        x = apply_brightness_contrast(x, 0, 40)
        x = document_scanner(x)
        # print(np.mean(x), cnt)
        cnt += 1
        if cnt > 5 or np.mean(x) > 220:
            # bgr to rgb
            return cv2.cvtColor(wrpimg,cv2.COLOR_BGR2RGB)

    x = cv2.cvtColor(x,cv2.COLOR_BGR2RGB)
    wrpimg = x
    return wrpimg