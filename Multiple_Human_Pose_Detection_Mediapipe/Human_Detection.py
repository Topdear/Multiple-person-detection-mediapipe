import cv2
import numpy as np
import imutils
from imutils.object_detection import non_max_suppression

import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

## Histogram of Oriented Gradients Detector
HOGCV = cv2.HOGDescriptor()
HOGCV.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


LEFT_ELBOW_angles=[]
LEFT_SHOULDER_angles=[]
LEFT_KNEE_angles=[]
RIGHT_ELBOW_angles=[]
RIGHT_SHOULDER_angles=[]
RIGHT_KNEE_angles=[]
person_pose=[]
person_hand=[]

def angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)

    if angle >180.0:
        angle = 360-angle
        
    return angle

def similarity_calculator(arr):
    try:
        value=(int(arr[0])*100)/int(arr[1])
        return int(value)
    except:
        pass



def mediapipe_pose(image):
    with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
        try:
            # Convert the BGR image to RGB before processing.
            results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            # Draw pose landmarks on the image.
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            landmarks = results.pose_landmarks.landmark
                
            # Get coordinates
            LEFT_SHOULDER = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            LEFT_ELBOW = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            LEFT_WRIST = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            RIGHT_SHOULDER = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            RIGHT_ELBOW = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            RIGHT_WRIST = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            RIGHT_HIP= [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            LEFT_HIP= [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            LEFT_KNEE = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            LEFT_ANKLE = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            RIGHT_KNEE = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            RIGHT_ANKLE = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
            # Calculate angle
            
            LEFT_ELBOW_angle = angle(LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST)
            RIGHT_ELBOW_angle = angle(RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST)
            RIGHT_SHOULDER_angle = angle(RIGHT_HIP, RIGHT_SHOULDER, RIGHT_ELBOW)
            LEFT_SHOULDER_angle = angle(LEFT_HIP, LEFT_SHOULDER, LEFT_ELBOW)
            LEFT_KNEE_angle = angle(LEFT_HIP, LEFT_KNEE, LEFT_ANKLE)
            RIGHT_KNEE_angle = angle(RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE)


            if LEFT_ELBOW_angle > 40  and LEFT_SHOULDER_angle > 90 : hand = "Hand up" 
            else: hand= "Hand Down"
                
            if RIGHT_ELBOW_angle > 45 and RIGHT_SHOULDER_angle > 90 : hand = "Hand up"
            else: hand= "Hand down"  
                
            if LEFT_KNEE_angle < 145 and RIGHT_KNEE_angle < 145: state = "Sitting"
            if RIGHT_KNEE_angle > 145 and LEFT_KNEE_angle > 145: state="Standing"
            
            LEFT_ELBOW_angles.append(LEFT_ELBOW_angle)
            LEFT_SHOULDER_angles.append(LEFT_SHOULDER_angle)
            LEFT_KNEE_angles.append(LEFT_KNEE_angle)
            RIGHT_ELBOW_angles.append(RIGHT_ELBOW_angle)
            RIGHT_SHOULDER_angles.append(RIGHT_SHOULDER_angle)
            RIGHT_KNEE_angles.append(RIGHT_KNEE_angle)
            person_pose.append(state)
            person_hand.append(hand)
        except:
            pass
        return(image)


def Detector(frame):
    ## USing Sliding window concept
    rects, weights = HOGCV.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=1.03)
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    for x,y,w,h in rects:
        crop = frame[y:y+h , x:x+w]
        crop= mediapipe_pose(crop)
        frame[y:y+h,x:x+w] = crop
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
    c = 1
    for x, y, w, h in pick:
        cv2.rectangle(frame, (x, y), (w, h), (139, 34, 104), 2)
        cv2.rectangle(frame, (x, y - 20), (w,y), (139, 34, 104), -1)
        cv2.putText(frame, f'P{c}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        c += 1
        

    #cv2.putText(frame, f'Total Persons : {c - 1}', (20, 450), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255,255), 2)
    cv2.imshow('output', frame)

    LEFT_ELBOW_angles.sort()
    LEFT_SHOULDER_angles.sort()
    LEFT_KNEE_angles.sort()
    RIGHT_ELBOW_angles.sort()
    RIGHT_SHOULDER_angles.sort()
    RIGHT_KNEE_angles.sort()
    try:
        similarity=similarity_calculator(LEFT_ELBOW_angles)
        similarity=similarity + similarity_calculator(LEFT_SHOULDER_angles)
        similarity=similarity + similarity_calculator(LEFT_KNEE_angles)
        similarity=similarity + similarity_calculator(RIGHT_ELBOW_angles)
        similarity=similarity + similarity_calculator(RIGHT_SHOULDER_angles)
        similarity=similarity + similarity_calculator(RIGHT_KNEE_angles)
        print(similarity)
        similarity /= 6
    except:
        pass

    result=cv2.imread('/home/adrbck/Documents/Pedestrian_Detection_OpenCV/results.jpg')
    img = imutils.resize(result, width=700, height=200)
    try:
        cv2.putText(result, f'Person1 pose: {person_pose[0]}', (15, 20), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(result, f'Person1 hand: {person_hand[0]}', (15, 50), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(result, f'Person2 pose: {person_pose[1]}', (500, 20), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(result, f'Person2 hand: {person_hand[1]}', (500, 50), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(result, f'Pose similarity: {similarity}%', (250, 100), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 0), 2)
    except:
        pass
    cv2.imshow('Results', result)
    return frame