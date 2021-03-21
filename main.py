from cv2 import cv2
import numpy as np

capture = cv2.VideoCapture(1)
capture.set(10, 160)
capture.set(3, 1920)
capture.set(4, 1080)
scale = 3
scaledWidth = 210 * scale
scaledHeight = 297 * scale
wanted_values = [5,5]
tolerances = [0.0,0.3]
hit = True

def getContours(img, thresholds=[100, 100], min_area=2000):

    contours = []
    #blurred_img = cv2.blur(img,ksize=(5,5))
    #med_val = np.median(blurred_img) 
    #lower = int(max(0 ,0.7*med_val))
    #upper = int(min(255,1.3*med_val))
    image_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(image_gray, (5, 5), 1)
    img_canny = cv2.Canny(img_blur, thresholds[0], thresholds[1])
    img_dilated = cv2.dilate(img_canny, np.ones((5, 5)), iterations=3)
    img_eroded = cv2.erode(img_dilated, np.ones((5, 5)), iterations=2)
    #if thresholds[1] == 100:
    #    cv2.imshow('eroded',img_eroded)
    all_contours, ret = cv2.findContours(img_eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in all_contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            arc_length = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.1*arc_length,True)
            bounding_box = cv2.boundingRect(approx)
            if len(approx) == 4:
                contours.append([len(approx), area, approx, bounding_box, contour])
    contours = sorted(contours, key=lambda x: x[1], reverse=True)
    return img, contours

def reorderPoints(myPoints, itsPaper=False):
    myPointsNew = np.zeros_like(myPoints)
    myPoints = myPoints.reshape((4, 2))
    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    if itsPaper:
        if (myPointsNew[1][0][0]-myPointsNew[0][0][0]) > (myPointsNew[2][0][1]-myPointsNew[0][0][1]):
            myPointsReversed = np.zeros_like(myPointsNew)
            myPointsReversed[1] = myPointsNew[0]
            myPointsReversed[2] = myPointsNew[3]
            myPointsReversed[3] = myPointsNew[1]
            myPointsReversed[0] = myPointsNew[2]
            return myPointsReversed

    return myPointsNew

def warpImg (img,points,w,h,padding=60):
    points = reorderPoints(points,True)
    pts1 = np.float32(points)
    pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    img_wrapped = cv2.warpPerspective(img, matrix, (w, h))
    img_wrapped = img_wrapped[padding:img_wrapped.shape[0]-padding, padding:img_wrapped.shape[1]-padding]
    return img_wrapped

def findDis(pts1,pts2):
    return ((pts2[0]-pts1[0])**2 + (pts2[1]-pts1[1])**2)**0.5

while True:
    ret,image = capture.read()
    imgContours, conts = getContours(image,min_area=50000)
    if len(conts) != 0:
        biggest = conts[0][2]
        img_wrapped = warpImg(image, biggest, scaledWidth,scaledHeight)
        imgContours2, conts2 = getContours(img_wrapped, thresholds=[50,50])
        if len(conts) != 0:
            for contour in conts2:
                nPoints = reorderPoints(contour[2])

                nW = round((findDis(nPoints[0][0]//scale, nPoints[1][0]//scale)/10), 1)
                nH = round((findDis(nPoints[0][0]//scale, nPoints[2][0]//scale)/10), 1)
                current_color_w = (0, 255, 0)
                current_color_h = (0, 255, 0)
                has_bad_card = False
                if abs(nW-wanted_values[0]) > tolerances[0]:
                    has_bad_card = True
                    hit = False
                    current_color_w = (0, 0, 255)
                else:
                    hit = True
                if abs(nH-wanted_values[1]) > tolerances[1]:
                    has_bad_card = True
                    hit = False
                    current_color_h = (0, 0, 255)
                else:
                    hit = True
                if not hit:
                    cv2.arrowedLine(imgContours2, (nPoints[0][0][0], nPoints[0][0][1]), (nPoints[1][0][0], nPoints[1][0][1]),
                                    current_color_w, 3, 8, 0, 0.1)
                    cv2.arrowedLine(imgContours2, (nPoints[0][0][0], nPoints[0][0][1]), (nPoints[2][0][0], nPoints[2][0][1]),
                                    current_color_h, 3, 8, 0, 0.1)
                    x, y, w, h = contour[3]

                    cv2.putText(imgContours2, '{}cm'.format(nW), (x + 30, y - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                                current_color_w, 2)
                    cv2.putText(imgContours2, '{}cm'.format(nH), (x - 70, y + h // 2), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                                current_color_h, 2)
                    if has_bad_card:
                        cv2.line(imgContours2,(nPoints[1][0][0], nPoints[1][0][1]),(nPoints[2][0][0], nPoints[2][0][1]),(0, 0, 255),4)
                        cv2.line(imgContours2,(nPoints[0][0][0], nPoints[0][0][1]),(nPoints[3][0][0], nPoints[3][0][1]),(0, 0, 255),4)
                else:
                    cv2.arrowedLine(imgContours2, (nPoints[0][0][0], nPoints[0][0][1]), (nPoints[1][0][0], nPoints[1][0][1]),
                                    (0, 255, 0), 3, 8, 0, 0.1)
                    cv2.arrowedLine(imgContours2, (nPoints[0][0][0], nPoints[0][0][1]), (nPoints[2][0][0], nPoints[2][0][1]),
                                    (0, 255, 0), 3, 8, 0, 0.1)
                    x, y, w, h = contour[3]

                    cv2.putText(imgContours2, '{}cm'.format(wanted_values[0]), (x + 30, y - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                                (0, 255, 0), 2)
                    cv2.putText(imgContours2, '{}cm'.format(wanted_values[1]), (x - 70, y + h // 2), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                                (0, 255, 0), 2)

        cv2.imshow('A4', imgContours2)

    image = cv2.resize(image, (0, 0), None, 0.8, 0.8)
    cv2.imshow('Original', image)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
