import numpy as np
import cv2

class Document:
    def __init__(self,name,link):
        self.name = name
        self.link = link
        self.image_blank = None
        self.image = cv2.imread(link)
        self.corners_detected = None
        self.contours_page = None
        self.pts = None
        self.contour = None
        self.destination_corners = None
        self.final = None
        self.width = self.image.shape[0]
        self.height = self.image.shape[1]
        self.warped = None


    def blank_space(self,iter):
        kernel = np.ones((5, 5), np.uint8)
        img = cv2.imread("Document.jpg")
        self.image_blank = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=iter)

        cv2.imshow("first",self.image_blank)
        cv2.waitKey(0)
        cv2.destroyAllWindows()



    def show_original(self,link):
        image = cv2.imread(link)
        self.image = image
        cv2.imshow('Image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    def corner_detection(self):
        #смена цвета картинки
        gray = cv2.cvtColor(self.image_blank, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (11, 11), 0)
        # определение краев интересующей области
        canny = cv2.Canny(gray, 0, 200)
        canny = cv2.dilate(canny, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        self.corners_detected = canny

        cv2.imshow("second", self.corners_detected)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    def contour_detection(self):
        # пустой фон
        con = np.zeros_like(self.image_blank)
        # находим контуры отдельных ребер
        contours, hierarchy = cv2.findContours(self.corners_detected, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        # оставляем только самый большой контур
        page = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
        self.contours_page = page
        con = cv2.drawContours(con, page, -1, (0, 255, 255), 3)

        cv2.imshow("first", con)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def Corner_Points_Detection(self):
        # Blank canvas.
        con = np.zeros_like(self.image_blank)
        # Loop over the contours.
        for q in self.contours_page:
            # Approximate the contour.
            epsilon = 0.02 * cv2.arcLength(q, True)
            corners = cv2.approxPolyDP(q, epsilon, True)
            print("corners \n",corners)
            # If our approximated contour has four points
            if len(corners) == 4:
                c = q
                break
        cv2.drawContours(con, c, -1, (0, 255, 255), 3)
        cv2.drawContours(con, corners, -1, (0, 255, 0), 10)
        # Sorting the corners and converting them to desired shape.
        corners = sorted(np.concatenate(corners).tolist())
        self.corners_detected = corners
        self.pts = corners
        self.contour = con
        #Демонстрация углов
        corners = self.rearranging_corners()
        self.pts = corners
        for index, c in enumerate(corners):
            character = chr(65 + index)
            cv2.putText(con, character, tuple(c), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)

        cv2.imshow("first", con)
        cv2.waitKey(0)
        cv2.destroyAllWindows()



    def rearranging_corners(self):
        '''Rearrange coordinates to order:
        top-left, top-right, bottom-right, bottom-left'''
        rect = np.zeros((4, 2), dtype='float32')
        pts = np.array(self.pts)
        s = pts.sum(axis=1)
        # Top-left point will have the smallest sum.
        rect[0] = pts[np.argmin(s)]
        # Bottom-right point will have the largest sum.
        rect[2] = pts[np.argmax(s)]

        diff = np.diff(pts, axis=1)
        # Top-right point will have the smallest difference.
        rect[1] = pts[np.argmin(diff)]
        # Bottom-left will have the largest difference.
        rect[3] = pts[np.argmax(diff)]
        # Return the ordered coordinates.
        #rect = rect.reshape(-1, 1, 2)
        rect = rect.astype('int')
        return rect.astype('int').tolist()

    #def Destination_coordinates(self):
    #    (tl, tr, br, bl) = self.pts
    #    # Finding the maximum width.
    #    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    #    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    #    maxWidth = max(int(widthA), int(widthB))
    #    # Finding the maximum height.
    #    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    #    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    #    maxHeight = max(int(heightA), int(heightB))
    #    # Final destination co-ordinates.
    #    destination_corners = [[0, 0], [maxWidth, 0], [maxWidth, maxHeight], [0, maxHeight]]
    #    self.destination_corners = destination_corners
    def final_destination(self):
        print(self.height)
        print("\n")
        print(self.width)
    #def Perspective_transform(self):
    #    # Getting the homography.
    #   M = cv2.getPerspectiveTransform(np.float32(self.corners_detected), np.float32(self.destination_corners))
    #    # Perspective transform using homography.
    #    final = cv2.warpPerspective(self.image, M, (self.destination_corners[2][0], self.destination_corners[2][1]),
    #                                flags=cv2.INTER_LINEAR)
    #    self.final = final
    #    cv2.imshow("first", final)
    #    cv2.waitKey(0)
    #    cv2.destroyAllWindows()

    def perspective_transform(self):
        points_src = np.float32(self.pts)
        points_dst = np.float32([[0, 0], [self.height, 0], [self.height, self.width], [0, self.width]])
        matrix = cv2.getPerspectiveTransform(points_src, points_dst)
        warped_image = cv2.warpPerspective(self.image, matrix, (self.height, self.width))
        self.warped = warped_image

    def full_sborka(self):
        self.blank_space(3)
        print("blank_space")
        self.corner_detection()
        print("corner_detection")
        self.contour_detection()
        print("contour_detection")
        self.Corner_Points_Detection()
        print("Corner_Points_Detection")
        self.rearranging_corners()
        print("perspective transform")
        self.perspective_transform()
        cv2.imshow("warped",self.warped)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imwrite("warped.jpg",self.warped)
        #self.Destination_coordinates()
        #print("destination_coordinates")
        #self.Perspective_transform()
        #print("Persepctive treansform")
        #cv2.imshow("final result",self.final)



myDocument = Document("scan","Document.jpg")
myDocument.full_sborka()