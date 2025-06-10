import cv2
import numpy as np

####################
# Detecting the size of the object on A4 sheet paper
####################

webcam = False
path = "your_image.jpg"
cap = cv2.VideoCapture(0)  # camera object
cap.set(10, 160)  # brightness id, value
cap.set(3, 1920)  # Height
cap.set(4, 1080)  # Width
scale = 3  # so the result image is not very small (210, 297) pixels
w_paper = 210 * scale
h_paper = 297 * scale


def getContours(img, cThr=None, show=False, minArea=1000, fltr=0, draw=False):
    if cThr is None:
        cThr = [150, 150]
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert to grayscale
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    imgCanny = cv2.Canny(imgBlur, cThr[0], cThr[1])  # apply canny edge detector
    kernel = np.ones((5, 5))
    imgDial = cv2.dilate(imgCanny, kernel, iterations=3)  # dilate the image
    imgThre = cv2.erode(imgDial, kernel, iterations=2)  # Thresholding
    if show:
        cv2.imshow("Canny", imgThre)
    finalContours = []
    contours, hierarchy = cv2.findContours(
        imgThre, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    # external-outer edges and contour approximation method-simple chain approximation
    for i in contours:
        area = cv2.contourArea(i)
        if area > minArea:
            peri = cv2.arcLength(i, True)  # perimeter
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)  # find corner points
            bbox = cv2.boundingRect(approx)
            if (
                fltr > 0
            ):  # if user only wants to detect some shapes like triangle or rectangle
                if (
                    len(approx) == fltr
                ):  # only the contours having corners equal to the user input (fltr)
                    finalContours.append([len(approx), area, approx, bbox, i])
            else:
                finalContours.append([len(approx), area, approx, bbox, i])

    # Sorting contours based on size
    finalContours = sorted(
        finalContours, key=lambda x: x[1], reverse=True
    )  # key only accepts function args
    if draw:
        for con in finalContours:
            cv2.drawContours(img, con[4], -1, (0, 0, 255), 3)

    return img, finalContours


def reorder(my_points):
    # we need to sort out the corners such that the first point is always the top left, 2-top right, 3-bottom left etc
    print(my_points.shape)  # (4, 1, 2)
    my_points = my_points.reshape((4, 2))
    add = my_points.sum(
        1
    )  # sum along 1st axis. smallest will be top left corner(1). largest will be bottom right(4)
    points_new = np.zeros_like(my_points)
    points_new[0] = my_points[np.argmin(add)]
    points_new[3] = my_points[np.argmax(add)]
    diff = np.diff(my_points, axis=1)
    points_new[1] = my_points[np.argmin(diff)]
    points_new[2] = my_points[np.argmax(diff)]
    return points_new


def warpImg(img, points, w, h, pad=20):
    # print(points)
    points = reorder(points)
    pt1 = np.float32(points)
    pt2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])  # background
    matrix = cv2.getPerspectiveTransform(pt1, pt2)
    print("Matrix \n", matrix)
    imgWarp = cv2.warpPerspective(img, matrix, (w, h))
    imgWarp = imgWarp[pad : imgWarp.shape[0] - pad, pad : imgWarp.shape[1] - pad]
    return imgWarp


def Dist(pts1, pts2):
    return ((pts2[0] - pts1[0]) ** 2 + (pts2[1] - pts1[1]) ** 2) ** 0.5


if __name__ == "__main__":

    while True:
        if webcam:
            success, img = cap.read()
        else:
            img = cv2.imread(path)

        imgConts, conts = getContours(
            img, show=True, minArea=50000, fltr=4, draw=True
        )  # find A4 sheet contour
        if len(conts) != 0:  # make sure it's not an empty list
            biggest = conts[0][2]  # get the approximation of the corners
            print(biggest)
            imgWarp = warpImg(img, biggest, w_paper, h_paper)

            imgConts2, conts2 = getContours(
                imgWarp, minArea=2000, fltr=4, cThr=[50, 50]
            )
            if len(conts2) != 0:
                for obj in conts2:
                    cv2.polylines(imgConts2, [obj[2]], True, (0, 255, 0), 2)
                    nPoints = reorder(obj[2])
                    width = round(Dist(nPoints[0] / scale, nPoints[1] / scale) / 10, 1)
                    height = round(Dist(nPoints[0] / scale, nPoints[2] / scale) / 10, 1)

                    # Display ______________
                    cv2.arrowedLine(
                        imgConts2,
                        (nPoints[0][0], nPoints[0][1]),
                        (nPoints[1][0], nPoints[1][1]),
                        (255, 0, 255),
                        3,
                        8,
                        0,
                        0.05,
                    )
                    cv2.arrowedLine(
                        imgConts2,
                        (nPoints[0][0], nPoints[0][1]),
                        (nPoints[2][0], nPoints[2][1]),
                        (255, 0, 255),
                        3,
                        8,
                        0,
                        0.05,
                    )
                    x, y, w, h = obj[3]
                    cv2.putText(
                        imgConts2,
                        "{}cm".format(width),
                        (x + 30, y - 10),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        1,
                        (255, 0, 255),
                        2,
                    )
                    cv2.putText(
                        imgConts2,
                        "{}cm".format(height),
                        (x - 70, y + h // 2),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        1,
                        (255, 0, 255),
                        2,
                    )
            cv2.imshow("Warped image A4", imgWarp)

        img = cv2.resize(img, (0, 0), None, 0.5, 0.5)
        cv2.imshow("Original", img)
        cv2.waitKey(1)
