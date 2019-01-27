# !/usr/bin/env python
# coding: utf-8

import cv2
import numpy as np
import argparse
from keras.models import load_model


def get_emotion(list_proba):

    # Define list of emotion
    emotions = ['Angry', 'Disgut', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    # Get higher probability prediction
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

    # Load face classifier
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

    # Load keras model
    model = load_model("model.h5")

    cap = cv2.VideoCapture(0)

    while (True):

        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Get faces
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]
            roi_gray = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)

            # Predict proba for the 7 classes
            predicted_emotion = model.predict(roi_gray)
            emotion, color, proba = get_emotion((predicted_emotion))
            proba = '%.2f' % (proba)

            # Add emotion name and proba on frame
            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(frame, str(proba), (x, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    predict_emotion()
