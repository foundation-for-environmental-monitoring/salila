#!/usr/bin/python
import numpy as np
import argparse
import imutils
import cv2
import math

from salila import predict_result

labels = {"Fluoride" : {1:0, 2:0.5, 3:1, 4:1.5, 5:2, 6:0, 7:0.5, 8:1, 9:1.5, 10:2}, "pH" : {1:4, 2:5, 3:6, 4:7, 5:8, 6:9, 7:10, 8:4, 9:5, 10:6, 11:7, 12:8, 13:9, 14:10}}

#Reorder coordinates in a cyclic order
def re_order_points(pts):
	rect = np.zeros((4, 2), dtype = "float32")
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]

	return rect

def get_warped_image(image, pts):
	rect = re_order_points(pts)
	(tl, tr, br, bl) = rect
 
	wA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	wB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(wA), int(wB))
 
	hA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	hB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(hA), int(hB))
 
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
 
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
 
	return warped

def process_image(filename, test, show = False, debug = False):

    with open(filename, 'rb') as f:
        check_chars = f.read()[-2:]

    # https://stackoverflow.com/questions/33548956/detect-avoid-premature-end-of-jpeg-in-cv2-python
    if check_chars != b'\xff\xd9':
            print('Corrupt image')
            return 0,0,0,""
    else:
        image = cv2.imread(filename)

    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # if show:
    #     cv2.imshow("img", img)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

    # Incoming image is roughly 2k px wide and is cropped roughly to include only the bar codes, left & right grids & inner bottle area
    # Try and extract the left grid and right grid from the image
    img = cv2.medianBlur(img,1)
    dst = cv2.medianBlur(img,1)

    if show:
        cv2.imshow("img", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    threshold = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2, dst)

    if show:
        cv2.imshow("img", dst)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # print (threshold)

    contours, _ = cv2.findContours(threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
     
    # print (contours)

    i = 0
    bottle_enclosure_area = 100000000
    font = cv2.FONT_HERSHEY_SIMPLEX
    inner_approx = None
    out = test + "_colors.csv"
    f = open(out, "w")
    f.write("R,G,B,Label\n")

    height, width = img.shape
    print(f'Image width: {width}')

    for cnt in contours:

        # print(f"cnt arcLength: {cv2.arcLength(cnt, True)}")

        #Ignore too small contours
        if cv2.arcLength(cnt,True) < 800 or cv2.arcLength(cnt,True) > 5000:
            continue

        print(f"cnt arcLength: {cv2.arcLength(cnt, True)}")

        approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
        x, y, w, h = cv2.boundingRect(approx)
        ar = w / float(h)

        if show:
            print (f"x: {x}")
            print (f"y: {y}")
            print (f"w: {w}")
            print (f"h: {h}")
            print (f"ar: {ar}")
            print("----------------")

        #Drop the contours that are not squares, not four-sided and criss-crossed
        #This should roughly be the left and right outer contours.
        if len(approx) == 4 and ar <= 0.22:

            # if show:
            #     draw_contour = cv2.drawContours(img, [cnt], -1, (255, 140, 240), 2)
            #     cv2.imshow("Contour " + str(cv2.arcLength(cnt,True)), img)
            #     cv2.waitKey(0)
            #     cv2.destroyAllWindows()

            print ("countours ok")
            pts = np.zeros((4, 2), dtype = "float32")
            for idx in range(4):
                pts[idx] = approx[idx]
            warped = get_warped_image(image, pts)
            #Assuming that these are the outer grids, we can apply binary threshold
            #to get the inner grids
            xxx = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            thresh1 = cv2.adaptiveThreshold(xxx, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)
            innerc, _ = cv2.findContours(thresh1, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            yes = {0}
            xes = {0}
            xes.remove(0)
            yes.remove(0)
            rsum = 0
            count = 0
            for ic in innerc:
                
                if cv2.arcLength(ic,True) > 500 or cv2.arcLength(ic,True) < 350:
                    continue

                innera = cv2.approxPolyDP(ic, 0.08*cv2.arcLength(ic, True), True)
                xin, yin, win, hin = cv2.boundingRect(innera)
                arin = win / float(hin)
                #Let us try to locate the centroid of each grid (each of which are polygons with four sides
                #aspect ratio roughly equal to 1 and not criss-cross shapes. Once the centroids are 
                #located, we can cluster the centroids and identify general x-coord of each column 
                # and similarly identify the general y-coord of each row. Using which we can locate
                # all the 25 (x,y) of the inner grids
                if len(innera) == 4 \
                        and cv2.pointPolygonTest(ic, (xin + win / 2, yin + hin / 2), False) > 0  \
                        and cv2.pointPolygonTest(ic, (xin + win / 3, yin + hin * 2 / 3), False) > 0:

                    if show:
                        draw_contour = cv2.drawContours(warped, [ic], -1, (255, 140, 240), 2)
                        cv2.imshow("Inner " + str(cv2.arcLength(ic,True)), warped)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()

                    (cx, cy), radius = cv2.minEnclosingCircle(ic)
                    rsum += radius
                    count += 1
                    if len(xes) == 0:
                        xes.add(int(cx))
                    else:
                        added = False
                        for item in xes:
                            if cx < item + 20 and cx > item - 20:
                                added = True
                                break
                        if added == False:
                            xes.add(int(cx)) 
                    if len(yes) == 0:
                        yes.add(int(cy))
                    else:
                        added = False
                        for item in yes:
                            if cy < item + 20 and cy > item - 20:
                                added = True
                                break
                        if added == False:
                            yes.add(int(cy)) 
            xpos = 1
            for xx in sorted(xes):
                for yy in sorted(yes):     
                    if labels[test][xpos] == -1:
                        xpos += 1
                        continue
                    rd = int(rsum*7/(12*count))                
                    # Once we have located the centroid, we can create a bounding rectangle large enough to get
                    # enough pixels, but smaller than the bounding black line to avoid color errors. Each pixel
                    # is now a sample for learning, let us write these pixels along with labels for preparing 
                    # the learning set.
                    cv2.rectangle(warped, (xx-rd, yy-rd), (xx+rd, yy+rd), (255, 255, 255))
                    cv2.putText(warped, str(labels[test][xpos]), (xx, yy), font, 1,(255,255,255),2,cv2.LINE_AA)
                    for ycoord in range(xx-rd+1, xx+rd-1):
                        for xcoord in range(yy-rd+1, yy+rd-1): 
                            f.write(str(warped[xcoord][ycoord][2]) + "," +  str(warped[xcoord][ycoord][1]) + "," + str(warped[xcoord][ycoord][0]) + "," + test + str(labels[test][xpos]) + '\n')
                    xpos+=1
            if show:
                # Show grid with values for each box
                cv2.imshow("Warped" + str(i), warped)
                # print(f'Warped: {warped[xcoord][ycoord][2]}')
            i = i + 1
        else:
           #These are large enough polygons, but not the left and right grids, let us identify the bottle
           #enclosure and pick that
           
           if show:
                print (x)
                print(f'Bottle area: {bottle_enclosure_area}')
                print(f'Contour area: {cv2.contourArea(cnt)}')
                print(len(approx))
                print(image.shape[1]/3)

           if bottle_enclosure_area > cv2.contourArea(cnt) and len(approx) == 4 and x > image.shape[1]/3:
               inner_approx = approx
               bottle_enclosure_area = cv2.contourArea(cnt)
   
    ex, ey, ew, eh = cv2.boundingRect(inner_approx)

    print ('--------------------------------------')

    ex = int(ex + ew / 3)
    ey = int(ey + eh / 2)
    ew = int(ew/3)
    eh = int(eh/4)   
    
    cnt = 1
    rs = 0
    gs = 0
    bs = 0

    for bxcoord in range(ex, ex+ew):
        for bycoord in range(ey, ey+eh):
            cnt+=1
            rs+=image[bycoord][bxcoord][2]
            gs+=image[bycoord][bxcoord][1]
            bs+=image[bycoord][bxcoord][0]

    if debug and show:
        cv2.rectangle(image, (ex, ey), (ex+ew, ey+eh), (0, 0, 0), 0)
 
    f.close()
    if show:
        cv2.imshow("img", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return int(rs/cnt), int(gs/cnt), int(bs/cnt), out

def xmain():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--name", required = True,
            help = "name of the test")
    ap.add_argument("-i", "--image", required = True,
            help = "path to the image file")
    ap.add_argument("-x", "--show", required = False,
            help = "Display images")
    args = vars(ap.parse_args())
    show = True if args["show"] == None else args["show"] == 'True'
    r,g,b,out_file = process_image(args["image"], args["name"], show, True)
    print (r, g, b, out_file) 
    print (predict_result.salila_ml(out_file,r,g,b))

if __name__== "__main__":
    xmain()

