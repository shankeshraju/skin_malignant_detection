from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.metrics import BinaryAccuracy, Precision, Recall
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf
import pandas as pd
import numpy as np
import configparser
import warnings
import keras


warnings.filterwarnings("ignore")
tf.keras.backend.clear_session()
config = configparser.ConfigParser()
config.read('/content/drive/MyDrive/Sivaraj_skin_malignant/config.ini')
epochs = int(config['MODEL_PARAMS']['EPOCHS'])
inputW = int(config['MODEL_PARAMS']['INPUT_DIM_WIDTH'])
inputH = int(config['MODEL_PARAMS']['INPUT_DIM_WIDTH'])
inputCh = int(config['MODEL_PARAMS']['INPUT_DIM_CHANNEL'])
inputTrain = config['INPUT_PATH']['TRAIN']
inputTest = config['INPUT_PATH']['TEST']
savePath_model = config['OUTPUT_PATH']['MODEL']


# preprocessing the training and testing images 
def data_preprocess():
	testData = keras.utils.image_dataset_from_directory(inputTest)
	trainData =  keras.utils.image_dataset_from_directory(inputTrain)
	trainData = trainData.map(lambda x,y: (x/255, y))
	testData = testData.map(lambda x,y: (x/255, y))
	return trainData, testData


def model_init():
	dataAug = Sequential([
	    layers.RandomFlip("horizontal_and_vertical", input_shape=(inputW,inputH,inputCh)),
	    layers.RandomZoom(0.3),
	    layers.RandomContrast(0.3),
	    layers.RandomRotation(0.4)
	])
	model = Sequential([
	    dataAug,
	    Conv2D(16, (3,3), 1, activation="relu", padding="same"),
	    Conv2D(16, (3,3), 1, activation="relu", padding="same"),
	    MaxPooling2D(),
	    Conv2D(32, (5,5), 1, activation="relu", padding="same"),
	    Conv2D(32, (5,5), 1, activation="relu", padding="same"),
	    MaxPooling2D(),
	    Conv2D(16, (3,3), 1, activation="relu", padding="same"),
	    Conv2D(16, (3,3), 1, activation="relu", padding="same"),
	    MaxPooling2D(),
	    
	    Flatten(),
	    Dense(128, activation="relu"),
	    Dense(1, activation="sigmoid")
	])
	model.compile(loss='binary_crossentropy', 
             optimizer='adam', 
             metrics=[tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), "acc"])
	model.summary()
	return model


def modelTest(inputTest, model):
	binAcc = BinaryAccuracy()
	recAcc = Recall()
	precAcc = Precision()

	for item in inputTest.as_numpy_iterator():
	    X, y = item
	    yHat = model.predict(X)
	    binAcc.update_state(y, yHat)
	    recAcc.update_state(y, yHat)
	    precAcc.update_state(y, yHat)
	print("Accuracy:", binAcc.result().numpy(), "\nRecall:", recAcc.result().numpy(), "\nPrecision:", precAcc.result().numpy())


def save_model(model):
	model.save(f"{savePath_model}/model.keras")
	print(f'Model and Parameters saved in {savePath_model}')


def main():
	trainData, testData = data_preprocess()
	model = model_init()
	history = model.fit(trainData, epochs=epochs, validation_data=testData)
	modelTest(testData, model)
	save_model(model)


if __name__ == '__main__':
	main()