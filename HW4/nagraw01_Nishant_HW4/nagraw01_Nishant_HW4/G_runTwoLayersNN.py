import os
import time
import numpy as np

# Load the CIFAR10 dataset
from keras.datasets import cifar10
baseDir = os.path.dirname(os.path.abspath(__file__)) + '/'
classesName = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
(xTrain, yTrain), (xTest, yTest) = cifar10.load_data()
xVal = xTrain[49000:, :].astype(np.float)
yVal = np.squeeze(yTrain[49000:, :])
xTrain = xTrain[:49000, :].astype(np.float)
yTrain = np.squeeze(yTrain[:49000, :])
yTest = np.squeeze(yTest)
xTest = xTest.astype(np.float)

# Show dimension for each variable
print ('Train image shape:    {0}'.format(xTrain.shape))
print ('Train label shape:    {0}'.format(yTrain.shape))
print ('Validate image shape: {0}'.format(xVal.shape))
print ('Validate label shape: {0}'.format(yVal.shape))
print ('Test image shape:     {0}'.format(xTest.shape))
print ('Test label shape:     {0}'.format(yTest.shape))

# Normalize the data by subtract the mean image
meanImage = np.mean(xTrain, axis=0)
xTrain -= meanImage
xVal -= meanImage
xTest -= meanImage

# Reshape data from channel to rows
xTrain = np.reshape(xTrain, (xTrain.shape[0], -1))
xVal = np.reshape(xVal, (xVal.shape[0], -1))
xTest = np.reshape(xTest, (xTest.shape[0], -1))
print ('Train image shape after reshape:   {0}'.format(xTrain.shape))
print ('Val image shape after reshape:     {0}'.format(xVal.shape))
print ('Test image shape after reshape:    {0}'.format(xTest.shape))

# Two layer Neural Network
from G_twoLayersNN import TwoLayersNN
numClasses = np.max(yTrain) + 1
hiddenNeurons = 100

# Training classifier with Naive Update
print ('\n############ Naive Update ############')
classifier = TwoLayersNN(xTrain.shape[1], hiddenNeurons, numClasses, 0)
startTime = time.time()
classifier.train(xTrain, yTrain, iterations=1500 ,verbose=True)
print ('Training time:  {0:.2f}s'.format(time.time() - startTime))
print ('Training acc:   {0:.2f}%'.format(classifier.calAccuracy(xTrain, yTrain)))
print ('Validating acc: {0:.2f}%'.format(classifier.calAccuracy(xVal, yVal)))
print ('Testing acc:    {0:.2f}%'.format(classifier.calAccuracy(xTest, yTest)))

# Training classifier with Momentum Update
print ('\n############ Momentum Update ############')
classifier = TwoLayersNN(xTrain.shape[1], hiddenNeurons, numClasses, 1)
startTime = time.time()
classifier.train(xTrain, yTrain, iterations=1500 ,verbose=True)
print ('Training time:  {0:.2f}s'.format(time.time() - startTime))
print ('Training acc:   {0:.2f}%'.format(classifier.calAccuracy(xTrain, yTrain)))
print ('Validating acc: {0:.2f}%'.format(classifier.calAccuracy(xVal, yVal)))
print ('Testing acc:    {0:.2f}%'.format(classifier.calAccuracy(xTest, yTest)))

# Training classifier with Nesterov Update
print ('\n############ Nesterov Update ############')
classifier = TwoLayersNN(xTrain.shape[1], hiddenNeurons, numClasses, 2)
startTime = time.time()
classifier.train(xTrain, yTrain, iterations=1500 ,verbose=True)
print ('Training time:  {0:.2f}s'.format(time.time() - startTime))
print ('Training acc:   {0:.2f}%'.format(classifier.calAccuracy(xTrain, yTrain)))
print ('Validating acc: {0:.2f}%'.format(classifier.calAccuracy(xVal, yVal)))
print ('Testing acc:    {0:.2f}%'.format(classifier.calAccuracy(xTest, yTest)))

# Training classifier with AdaGrad Update
print ('\n############ AdaGrad Update ############')
classifier = TwoLayersNN(xTrain.shape[1], hiddenNeurons, numClasses, 3)
startTime = time.time()
classifier.train(xTrain, yTrain, iterations=1500 ,verbose=True)
print ('Training time:  {0:.2f}s'.format(time.time() - startTime))
print ('Training acc:   {0:.2f}%'.format(classifier.calAccuracy(xTrain, yTrain)))
print ('Validating acc: {0:.2f}%'.format(classifier.calAccuracy(xVal, yVal)))
print ('Testing acc:    {0:.2f}%'.format(classifier.calAccuracy(xTest, yTest)))

# Training classifier with RMSProp Update
print ('\n############ RMSProp Update ############')
classifier = TwoLayersNN(xTrain.shape[1], hiddenNeurons, numClasses, 4)
startTime = time.time()
classifier.train(xTrain, yTrain, iterations=1500 ,verbose=True)
print ('Training time:  {0:.2f}s'.format(time.time() - startTime))
print ('Training acc:   {0:.2f}%'.format(classifier.calAccuracy(xTrain, yTrain)))
print ('Validating acc: {0:.2f}%'.format(classifier.calAccuracy(xVal, yVal)))
print ('Testing acc:    {0:.2f}%'.format(classifier.calAccuracy(xTest, yTest)))

# Training classifier with Adam Update
print ('\n############ Adam Update ############')
classifier = TwoLayersNN(xTrain.shape[1], hiddenNeurons, numClasses, 5)
startTime = time.time()
classifier.train(xTrain, yTrain, iterations=1500 ,verbose=True)
print ('Training time:  {0:.2f}s'.format(time.time() - startTime))
print ('Training acc:   {0:.2f}%'.format(classifier.calAccuracy(xTrain, yTrain)))
print ('Validating acc: {0:.2f}%'.format(classifier.calAccuracy(xVal, yVal)))
print ('Testing acc:    {0:.2f}%'.format(classifier.calAccuracy(xTest, yTest)))
