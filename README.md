# Emotion-recognition
Real time emotion recognition with Keras CNN and OpenCv. Keras is used to train the emotion classification model on face and OpenCv to make simple face detection using Haarcascade classifier.

Model has been trained using google colab : https://colab.research.google.com/, it's a cloud service which provides free GPU

## Dependencies

* Keras
* Seaborn
* Sklearn
* OpenCv


Install dependencies :

```
pip install -r requirements.txt
```

## Docker
Real time emotion recognition can be run from docker. First, build the image :

```
docker build -t emotion-recognition .
```

Run the project with pre-trained model :

```
docker run --privileged --device=/dev/video0 -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=unix$DISPLAY emotion-recognition

```


## Usage 
    
To predict face emotion from webcam's frame:
``` python 
python predict_webcam.py
  ```
  
To predict face emotion from local image
``` python 
python predict_image.py -p path_to_image:
  ```


## Dataset description 
  
The model has been trained using <a href="https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data">fer2013 dataset </a> which is composed of 35887 samples splited between 7 classes:
  
  * Classe 0 : Angry 4593 images
  * Classe 1 : Disgust 547 images
  * Classe 2 : Fear 5121 images
  * Classe 3 : Happy 6077 images
  * Classe 4 : Sad 6077 images
  * Classe 5 : Surprise 4002 images
  * Classe 6 : Neutral 6198 images
  
<p float="left">
  <img src="img/data_ex.png" alt="hi" class="inline" width = 60% height = 60% />
  <img src="img/emotion.png" alt="hi" class="inline" width = 35%  height = 35% />
</p>  

All samples are 48x48 grayscale face images


## Model evaluation
	
* Accuracy :64,59 % on validation set 
	
  
<p float="left">
  <img src="img/accuracy.png" alt="hi" class="inline" height = 45% width = 45% /> 
  <img src="img/loss.png" alt="hi" class="inline" height = 45% width = 45% />
</p>


