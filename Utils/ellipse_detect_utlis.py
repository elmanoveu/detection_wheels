import cv2
import numpy as np
import math
import os
from PIL import Image, ImageFont, ImageDraw

def get_parent_dir(n=1):
    """returns the n-th parent dicrectory of the current
    working directory"""
    current_path = os.path.dirname(os.path.abspath(__file__))
    for _ in range(n):
        current_path = os.path.dirname(current_path)
    return current_path


# Set up folder names for default values
data_folder = os.path.join(get_parent_dir(n=1), "Data")

image_folder = os.path.join(data_folder, "Source_Images")

image_test_folder = os.path.join(image_folder, "Test_Images")
####описание функций
def checking_border(ell):
    (d1,d2),(xc,yc),ang=ell
    rmajor = max(d1,d2)/2
    if ang > 90:
        ang = ang - 90
    else:
        ang = ang + 90
    xtop = xc + math.cos(math.radians(ang))*rmajor
    ytop = yc + math.sin(math.radians(ang))*rmajor
    xbot = xc + math.cos(math.radians(ang+90*2))*rmajor
    ybot = yc + math.sin(math.radians(ang+90*2))*rmajor
    return int(xtop),int(ytop),int(xbot),int(ybot)
# Draw elipsis on image
def draw_ellipse(mask):
    try:
        image = Image.open(mask)
        if image.mode != "RGB":
            image = image.convert("RGB")
        image_array = np.array(image)
    except:
        print("File Open Error! Try again!")
        return None, None

    gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                       cv2.THRESH_BINARY_INV, 31, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    has_ellipse = len(contours) > 0
    if has_ellipse:
        for cnt in contours:
            if len(cnt)>5:
                ellipse = cv2.fitEllipse(cnt)
                (x, y), (MA, ma), angle = ellipse
                x2,y2,x1,y1=checking_border(ellipse)
                if (x>140 and x<160) and (y>250 and y<270):
                    if x1<image_array.shape[0] and y1<image_array.shape[1]:
                        S=3.14*MA*ma
                        if S>=20**4:
                            draw=ellipse
                    try:
                        cv2.ellipse(image_array,draw,(255,0,0),3)
                    except Exception:
                        pass
    return has_ellipse, mask, ellipse
def find_contours(mask):
    thresh = cv2.adaptiveThreshold(mask, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                       cv2.THRESH_BINARY_INV, 31, 0)
    contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    # draw contours on the original image
    image_copy = mask.copy()
    cv2.drawContours(image=image_copy, contours=contours,
                     contourIdx=-1, color=(255, 100, 255), thickness=2, lineType=cv2.LINE_AA)
    return contours,image_copy
def matched_area(shablon,scene):
    # инициализируем детектор точек
    orb = cv2.ORB_create()
    # запускаем поиск точек и вычисление дескрипторов
    kp1, des1 = orb.detectAndCompute(shablon, None)
    kp2, des2 = orb.detectAndCompute(scene, None)
    list_coord=[]
    for coord in kp2:
        list_coord.append(coord.pt)
    newY=[]
    newX=[]
    newXY=[]
    for i in list_coord[160:210]:
        newX.append(i[0])
        newY.append(i[1])
    newXY=np.array([newX,newY])
    max_x=newXY[:,np.argmax(newXY[0])]###точка максимума по икс
    max_y=newXY[:,np.argmax(newXY[1])]###точка максимума по игрек
    min_x=newXY[:,np.argmin(newXY[0])]###точка минимума по икс
    min_y=newXY[:,np.argmin(newXY[1])]###точка минимума по иигрек
    x_left,x_right,y_high,y_low=(min_x[0],(max_y[1]+min_y[1])//2),(max_x[0],(max_y[1]+min_y[1])//2),(max_y[0],max_y[1]),(min_y[0],min_y[1]) 
    return x_left,x_right,y_high,y_low
