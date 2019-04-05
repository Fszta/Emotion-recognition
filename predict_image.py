# !/usr/bin/env python
# coding: utf-8

import cv2
import numpy as np
import argparse
from keras.models import load_model

emotions = ['Angry', 'Disgut', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


def get_emotion(list_proba):

    emotion_proba = np.max(list_proba)
    emotion = emotions[np.argmax(list_proba)]

    if emotion == 'Angrey':
        color = emotion_proba * np.asarray((255, 0, 0))
    elif emotion == 'Disgust':
        color = emotion_proba * np.asarray((0, 0, 255))
    elif emotion == 'Fear':
        color = emotion_proba * np.asarray((255, 255, 0))
    elif emotion == 'Happy':
        color = emotion_proba * np.asarray((0, 100, 255))
    else:
        color = emotion_proba * np.asarray((0, 255, 0))

    return emotion, color, emotion_proba


def predict_emotion():

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path',
                        help='image path',
                        required='True')
    args = vars(parser.parse_args())

    # Load face classifier 
    face_cascade = cv2.CascadeClassifier('model/haarcascade_frontalface_alt.xml')

    # Load keras model
    model = load_model("model/model.h5")

    # Load an color image in grayscale
    img = cv2.imread(args['path'])

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Get faces
    faces = face_cascade.detectMultiScale(img, 1.3, 5)

    for (x,y,w,h) in faces:

        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        roi_gray = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)

        # Predict proba for the 7 classes
        predicted_emotion = model.predict(roi_gray)
        emotion , color, proba = get_emotion((predicted_emotion))
        proba = '%.2f'%(proba)

        # Add emotion name and proba on frame 
        cv2.putText(img, emotion, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1,color,2)
        cv2.putText(img,str(proba) , (x,y+h+30), cv2.FONT_HERSHEY_SIMPLEX, 1,color,2)

        cv2.rectangle(img,(x,y),(x+w,y+h),color,2)

    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

if __name__ == '__main__':
    predict_emotion()
