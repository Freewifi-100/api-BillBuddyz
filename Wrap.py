import numpy as np
import cv2
import matplotlib.pyplot as plt
import imutils
from imutils.perspective import four_point_transform
import skew



def resizer(image,width=500):
    # get widht and height of the image
    h,w,c = image.shape
    
    height = int((h/w)* width )
    size = (width,height)
    image = cv2.resize(image,(width,height))
    return image, size

def apply_brightness_contrast(input_img, brightness, contrast):
    
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

def find_size(four_point):
    x = np.array(four_point)
    x = x.reshape(4,2)
    x = x.astype(int)
    # print(x)
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    x4 = x[3]
    # print(x1,x2,x3,x4)
    # find the width of the image
    width1 = np.sqrt((x1[0] - x2[0])**2 + (x1[1] - x2[1])**2)
    width2 = np.sqrt((x3[0] - x4[0])**2 + (x3[1] - x4[1])**2)
    width = max(int(width1),int(width2))
    # find the height of the image
    height1 = np.sqrt((x1[0] - x4[0])**2 + (x1[1] - x4[1])**2)
    height2 = np.sqrt((x2[0] - x3[0])**2 + (x2[1] - x3[1])**2)
    height = max(int(height1),int(height2))
    
    return width, height, width1, width2, height1, height2

def scale_contour(cnt, scale):
    M = cv2.moments(cnt)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])

    cnt_norm = cnt - [cx, cy]
    cnt_scaled = cnt_norm * scale
    cnt_scaled = cnt_scaled + [cx, cy]
    cnt_scaled = cnt_scaled.astype(np.int32)

    return cnt_scaled

def four_point_transform(image, pts):
    # Function to perform the four point perspective transform
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def order_points(pts):
    # Function to order points in a consistent order: top-left, top-right, bottom-right, bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

def document_scanner(image, brightness, contrast, blur, val_threshold, threshold_method, contours_method, similarity, scaled=1.0, stp=0, expansion_factor=1.1):
    count = 0
    orig = image.copy()
    image = np.pad(image, ((50, 50), (50, 50), (0, 0)), mode='constant', constant_values=0)
    img_re, size = resizer(image)
    img_re = apply_brightness_contrast(img_re, brightness, contrast)
    
    try:
        detail = cv2.detailEnhance(img_re, sigma_s=60, sigma_r=0.5)
        gray = cv2.cvtColor(detail, cv2.COLOR_BGR2GRAY)  # Grayscale image
        bfilter = cv2.bilateralFilter(gray, 20, 17, 17)  # Noise reduction
        blur = cv2.GaussianBlur(bfilter, (blur, blur), 0)

        # Thresholding
        _, thresh = cv2.threshold(blur, val_threshold, 255, threshold_method)

        # Edge detection
        edge_image = cv2.Canny(thresh, 75, 200, apertureSize=3)

        # Hough Line Transformation
        lines = cv2.HoughLinesP(edge_image, rho=1, theta=np.pi/180, threshold=100, minLineLength=150, maxLineGap=20)
        if lines is not None:
            lines_image = img_re.copy()
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(lines_image, (x1, y1), (x2, y2), (255, 255, 255), 5)

        # Morphological transformation
        kernel = np.ones((3, 3), np.uint8)
        dilate = cv2.dilate(edge_image, kernel, iterations=1)
        closing = cv2.morphologyEx(dilate, cv2.MORPH_CLOSE, kernel)

        # Find the contours
        contours, _ = cv2.findContours(closing, cv2.RETR_LIST, contours_method)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        four_points = None
        for contour in contours:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, similarity * peri, True)

            if len(approx) == 4:
                four_points = np.squeeze(approx)
                break
        
        if four_points is None or cv2.contourArea(four_points) < (img_re.shape[0] * img_re.shape[1]) / 5:
            four_points = np.array([[0, 0], [size[0], 0], [size[0], size[1]], [0, size[1]]])

    except Exception as e:
        count = 1
        four_points = np.array([[0, 0], [size[0], 0], [size[0], size[1]], [0, size[1]]])

    # print(four_points)
    
    # Expand the detected four points
    # expanded_four_points = scale_contour(four_points.reshape(-1, 1, 2), expansion_factor).reshape(-1, 2)
    
    # # Check if expanded contour is within image bounds
    # if all(0 <= p[0] < size[0] and 0 <= p[1] < size[1] for p in expanded_four_points):
    #     four_points = expanded_four_points
    
    # If re_four_points is outside the image, adjust them
    i = stp
    while True:
        re_four_points = scale_contour(four_points.reshape(-1, 1, 2), scaled).reshape(-1, 2)
        
        if not (0 <= re_four_points[0][1] < 660 and 0 <= re_four_points[1][1] < 660 and
                0 <= re_four_points[2][1] < 660 and 0 <= re_four_points[3][1] < 660 and
                0 <= re_four_points[0][0] < 500 and 0 <= re_four_points[1][0] < 500 and
                0 <= re_four_points[2][0] < 500 and 0 <= re_four_points[3][0] < 500):
            i += 1
            scaled -= (0.01 / i)
            if i == 5:
                break
        else:
            break

    four_points = re_four_points

    cv2.drawContours(img_re, [four_points.reshape(-1, 1, 2)], 0, (0, 255, 0), 5)

    # Find four points for original image
    multiplier = image.shape[1] / size[0]
    four_points_orig = four_points * multiplier
    four_points_orig = four_points_orig.astype(int)

    wrap_image = four_point_transform(image, four_points_orig)

    if count == 1:
        wrap_image = wrap_image[50:-50, 50:-50, :]
    elif any(p[0] < 0 or p[1] < 0 or p[0] > size[0] or p[1] > size[1] for p in four_points):
        wrap_image = orig.copy()

    return img_re, detail, gray, blur, edge_image, dilate, closing, wrap_image, four_points_orig, four_points, size

def scan(image):
    angle, corrected = skew.correct_skew(image)
    #bgr to rgb
    corrected = cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB)
    if np.mean(corrected) > 150:
        b = 80
    else:
        b = np.mean(corrected)
    
    img_re, detail, gray, blur, edge_image, dilate, closing, wrpimg, points_ori, points_re, size = document_scanner(corrected, 0, 0, 5, b+70, cv2.THRESH_TRUNC,cv2.CHAIN_APPROX_SIMPLE, 0.02)

    # adjust contrast
    wrpimg = apply_brightness_contrast(wrpimg, 20, 40)
    wrpimg = cv2.detailEnhance(wrpimg, sigma_s=60, sigma_r=0.5)
    # convert to rgb
    wrpimg = cv2.cvtColor(wrpimg, cv2.COLOR_BGR2RGB)
    
    return wrpimg