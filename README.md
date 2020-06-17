# Scikit-learn-like-API-for-Facial-Expression-Recognition
Implementations of Neural Network Models with Scikit-learn-like Interface for facial expression recognition.

## Software
* Python 3.7.6
* Numpy 1.18.1
* Tensorflow 1.15

## Dataset
This data fer2013.csv is the training dataset of the [Facial Expression Recognition Challenge](
https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge) in Kaggle.

The data consists of 28,709 48x48 pixel grayscale images of faces. The faces have been automatically registered so that the face is more or less centered and occupies about the same amount of space in each image. fer2013.csv contains two columns, "emotion" and "pixels". The "emotion" column contains a numeric code ranging from 0 to 6 (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral) inclusive, for the emotion that is present in the image. The "pixels" column contains a string surrounded in quotes for each image. The contents of this string a space-separated pixel values in row major order.

## Outline
* Binary Classification (Sigmoid)
  * ANN with 0 Hidden Layer (Logistic Regression)
  * ANN with 1 Hidden Layer
  
* Multiclass Classification (Softmax)
  * ANN with 0 Hidden Layer (Logistic Regression)
  * ANN with 1 Hidden Layer
  * ANN with >1 Hidden Layer (Tensorflow)
  * CNN (Tensorflow)
