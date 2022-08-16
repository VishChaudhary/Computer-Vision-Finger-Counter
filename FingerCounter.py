#   Finger Counter:
#
#       This project uses your webcam and utilizes computer vision and hand detection to accurately predict how many
#   fingers are up. This works for either hand. It will display the number of how many fingers are up and a cartoon
#   image of how many fingers up. NOTE: the pictures are just a visual indicator of how many fingers up, the actual
#   picture does not correspond to the actual fingers up. Additionally, you can't use both hands at once, only one hand
#   at a time can be used.
#
#   Author: Vish Chaudhary
#   Github: https://github.com/VishChaudhary




import mediapipe as mp
import cv2
import time
import os
import HandTrackingModule as mod



cap = cv2.VideoCapture(0)
hands = mp.solutions.hands.Hands()
previous_time = 0

detector = mod.handDetector(detectionCon=0.8)

folderPath = 'Finger Images' #folder path
folder_content = os.listdir(folderPath) #folder_content is a list with the name of the contents of the folder as its values

sorted_content = sorted(folder_content)

overlay_list = []   #empty list that will be used to store the overlays. The folder_content will fill the list
            #one by one using a for loop.


for imPath in sorted_content:   #for image path in folder_content
    image = cv2.imread(f'{folderPath}/{imPath}')    #imports the different images from the folder one at a time. This is
                                            #complete path that needs to be read from
    overlay_list.append(image)  # adds each of the 6 different selection images onto our list of overlays

tipIds = [4,8,12,16,20] #list containing the landmark id numbers of each finger tip

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)  #flips image
    img = detector.findHands(img)
    lmList = detector.findPosition(img, False)
    imgColor = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgColor)
    handedness = detector.handedness(img)
    if len(lmList) != 0:
        finger = []

        #Special Thumb Case- If thumb is to the left of the middle of the thumb then its closed
        #This works only for the right hand
        if len(handedness) != 0:
            if handedness == 'Right':
                if lmList[tipIds[0]][1] < lmList[tipIds[0] - 1][1]:
                    finger.append(1)  # adds 1 to the fingers list for every finger that is open
                else:
                    finger.append(0)  # adds 0 to the fingers list for every finger that is closed.
            if handedness == 'Left':
                if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
                    finger.append(1)  # adds 1 to the fingers list for every finger that is open
                else:
                    finger.append(0)  # adds 0 to the fingers list for every finger that is closed.

        #Four fingers
        for id in range (1,5):  #from 1 to 5 because thumb gets its own special case
            if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2 ][2]:  #Its less than rather than open because in open cv the top of the screen has a
                #lower value than the bottom, so its backwards. lmlist has 3 values in it: 0.landmark numer (0-20) 1. x-pixel coordinate
                                            #2. y-pixel coordinate. Here the 2 refers to the y coordinate. This is essentially saying if the tip
                                             #of the finger(tipIds[id]) is less than the middle of that finger(tipIds[id] -2 )<--- this
                                                # is because the middle of each finger is two landmarks below its tip.
                finger.append(1)  #adds 1 to the fingers list for every finger that is open
            else:
                finger.append(0)  #adds 0 to the fingers list for every finger that is closed.
        total_fingers = finger.count(1)

        h, w, c = overlay_list[total_fingers].shape  # h-height, w-width, c-channel. We get the dimmensions of our overlay
        img[0:h, 0:w] = overlay_list[total_fingers] # splicing the image and setting the image from  y-from 0 to h, x-from o to w to
        # our overlay

        cv2.rectangle(img, (0,200), (118, 400), (0,255,0), cv2.FILLED)
        cv2.putText(img, str(total_fingers), (10, 355), cv2.FONT_HERSHEY_COMPLEX, 5, (255,0,0), 3)

    current_time = time.time()
    fps = 1 / (current_time-previous_time)
    previous_time = current_time

    cv2.putText(img, 'Fps:' + str(int(fps)), (150, 70), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 0), 2) #displays fps

    cv2.imshow('My Webcam', img)
    cv2.waitKey(1)
