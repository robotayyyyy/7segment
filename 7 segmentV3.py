import numpy as np
import cv2
import imutils
import os
import matplotlib.pyplot as plt

os.chdir('F:/Users/K-GIFT/Desktop/7segment/vid')
#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('13.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = 0
maskGreen = []
maskGreenFat = []
first = True
text = []

#param#-------------------------------------
green1 = (35, 80, 80) #green1(HSV)
green2 = (140, 255, 255) #green2(HSV)
#HSV H=hue S=saturation V=value or brightness, low S or V mean can't distunguish color
brightness = float(0.65)
paramW = 34 #digit width for 50px high
bwThresh = 100 #binary threshold
numOfCnt = 2 #amount of screen
digitThresh = 0.10 #digit threshold

def resizeH(img,high):
    h = img.shape[0]
    w = img.shape[1]
    img = cv2.resize(img,( int(high*w/h) ,high))
    return(img)

def preprocess(img):
    img = resizeH(img,50)
    img = cv2.bilateralFilter(img, 20, 50, 100)#crucial
    img = cv2.multiply(img, np.array([brightness]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.threshold(img, bwThresh, 255, cv2.THRESH_BINARY)[1]
    return(img)

def preprocessGreenFilter(img):
    img = resizeH(img,50)
    img = cv2.bilateralFilter(img, 20, 50, 100)#crucial
    img = cv2.multiply(img, np.array([0.8]))
    img = cv2.inRange(img, green1, green2)
    img = cv2.Canny(img, 50,255)
    return(img)

while(True):
    # 0) read image, preprocess
    ret, frame = cap.read()
    original = frame.copy()
    frame = cv2.bilateralFilter(frame, 20, 50, 100)#crucial
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #gray = cv2.bilateralFilter(gray, 20, 50, 100)
    #bw = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)[1]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    #detect line ----------------------------------------------------------
    # 1) get green area ------------------------
    maskGreen = cv2.inRange(hsv, green1, green2)
    maskGreen = cv2.GaussianBlur(maskGreen,(201,201),0) #need really big kernel for bluring
    maskGreen = cv2.threshold(maskGreen, 30, 255, cv2.THRESH_BINARY)[1] #threshold 50
    if first:
        maskGreenFat = maskGreen*0
        first = False
    maskGreenFat = cv2.bitwise_or(maskGreen,maskGreenFat) #bitwise or will be summation of 2 arrays for given range 0 1
    result = cv2.bitwise_and(frame, frame, mask=maskGreenFat)


    # 2) convert roi to line of green color-----------
    result = cv2.multiply(result, np.array([brightness]))#reduce noise, multiply by 0.7 mean reduce brigness for wile image. However hue still remain. 0.6 is crucial!!!
    result = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
    result = cv2.inRange(result, green1, green2)# then select only pixel that in range
    edge = cv2.Canny(result, 50,255)#non-sensitive param
    bolderY = cv2.GaussianBlur(edge,(201,5),0)# 5 for make the case number 0.0 not to be splited


    # 3) grab contour of line
    cnts = cv2.findContours(bolderY, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cntsList = []
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        cntsList.append([x, y, w, h])
    #sort contour by size
    cntsSize = [ c[2]*c[3] for c in cntsList ]
    cntsSortedIndex = np.sort(cntsSize)[::-1]
    cntsSortedIndex = [cntsSize.index(size) for size in  cntsSortedIndex]
    #get only the i_th biggest cnts
    if len(cntsSortedIndex)>=numOfCnt:
        cntsSortedIndex = cntsSortedIndex[0:numOfCnt]
    #draw it
    outputImg = original.copy()
    tempImg = original.copy()
    sectionList = []
    yList = []
    for i in cntsSortedIndex:
        x, y, w, h = cntsList[i]
        sectionList.append( preprocess(tempImg[y:y+h,x:x+w])  )
        cv2.rectangle(outputImg,(x,y),(x+w,y+h),(255,0,255),3)
        yList.append(y)


    # 4) get number from the line
    imgList = []
    for tempImg in sectionList:
        #try to draw vertical lines
        imgW = tempImg.shape[1]
        vLine = np.arange(0,imgW,paramW)
        fStop = imgW - vLine[-1] -1
        sum = [ np.sum( tempImg[ :,[vLine+i] ] ) for i in np.arange(0,fStop)]
        sum += [np.sum( tempImg[ :,[vLine[:-1]+i] ] ) for i in np.arange(fStop,paramW)]
        vLine = vLine + np.argmin(sum)
        if np.argmin(sum) >= fStop:
            vLine = vLine[:-1]

        xRange = [ np.arange(vLine[i],vLine[i+1]) for i in range(len(vLine)-1)]
        imgList.append( [ tempImg[:,xRange[i] ] for i in range(len(xRange)) if np.sum(tempImg[:,xRange[i] ])> digitThresh*(50*paramW)*255 ] )
        for x in vLine:
            cv2.line(tempImg,(x,0),(x,1000), (255,255,255) )

    #manual adjust---------(mouyWAT method)
    if( yList[0] < yList[0]):
        topNum = imgList[0]
        botNum = imgList[1]
    else:
        topNum = imgList[1]
        botNum = imgList[0]

    outputTop = np.zeros((50,50))
    outputBot = np.zeros((50,50))
    if len(topNum)>0:
        outputTop =  np.hstack( tuple( [topNum[i] for i in range(len(topNum))] ) )
    if len(botNum)>0:
        outputBot =  np.hstack( tuple( [botNum[i] for i in range(len(botNum))] ) )


    # plt.imshow(tempImg[:,xRange[0] ])
    # plt.show()

    # FFF) display
    #for numberImg in imgList:
    #    outputImg2 =  np.hstack( tuple( [numberImg[i] for i in range(len(numberImg))] ) )


    print( [ len(imgList[i]) for i in range(len(imgList))])
    #outputImg2 = np.hstack( (sectionList[0],sectionList[1]) )
    #outputImg2 = cv2.resize(outputImg2,(640,outputImg2.shape[0]))


    frame_count +=1
    duration = int(frame_count/fps)
    cv2.putText(outputImg,str(duration),(30,30), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),5)
    cv2.imshow('frame',outputImg)
    cv2.imshow('top',outputTop)
    cv2.imshow('bot',outputBot)

    if cv2.waitKey(1) & 0xFF == ord(' '):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()





