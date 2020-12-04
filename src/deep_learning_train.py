import yaml
import argparse
from utils import split_data
import numpy as np
from tensorflow.keras import layers, models
from keras.utils import to_categorical
from tensorflow import keras
import tensorflow as tf
import os
import pandas as pd
from sklearn.metrics import accuracy_score
from kymatio.keras import Scattering1D


#Shapes list:
#------------
# Basic preprocesed : (30000,8000)
# MFCC : (30000,20,16)
# Chroma_stft : (30000, 12, 16)
# Root Mean Square : (30000, 1, 16)



def fast_cnn(n_classes=10, sequence_length=8000):
	""" Fast CNN to use for experiments

	Args:
		n_classes (int): Number of classes
		sequence_length (int): Length of input sequences
	Returns:
		(keras model): Model that groups layers into an object with training and inference features.
	"""
	inputs = keras.Input(shape=(sequence_length, 1), name="record")
	x = layers.Conv1D(filters=1, kernel_size=1, strides=1, activation='relu', padding='valid', dilation_rate=1, use_bias=True)(inputs)
	x = layers.Flatten()(x)
	prediction = layers.Dense(10, activation='softmax')(x)
	model = models.Model(inputs=inputs, outputs=prediction)
	return model


 
def audionet(n_classes=10, sequence_length=8000, pool_size=2, pool_strides=2):
	""" AudioNet CNN

	Args:
		n_classes (int): Number of classes to be predicted
		sequence_length (int): Length of inputs sequences
		pool_size (int): Size of the max pooling window
		pool_strides (int): Specifies how much the pooling window moves for each pooling step. If None, it will default to pool_size
	
	Returns:
		(keras model): Model that groups layers into an object with training and inference features.
	"""
	inputs = keras.Input(shape=(sequence_length, 1), name="record")
	x = layers.Conv1D(filters=100, kernel_size=3, strides=1, activation='relu', padding='valid', dilation_rate=1, use_bias=True)(inputs)
	x = layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(x)
	
	x = layers.Conv1D(filters=64, kernel_size=3, strides=1, activation='relu', padding='valid', dilation_rate=1, use_bias=True)(x)
	x = layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(x)
   
	x = layers.Conv1D(filters=128, kernel_size=3, strides=1, activation='relu', padding='valid', dilation_rate=1, use_bias=True)(x)
	x = layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(x)

	x = layers.Conv1D(filters=128, kernel_size=3, strides=1, activation='relu', padding='valid', dilation_rate=1, use_bias=True)(x)
	x = layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(x)

	x = layers.Conv1D(filters=128, kernel_size=3, strides=1, activation='relu', padding='valid', dilation_rate=1, use_bias=True)(x)
	x = layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(x)

	x = layers.Conv1D(filters=128, kernel_size=3, strides=1, activation='relu', padding='valid', dilation_rate=1, use_bias=True)(x)
	x = layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(x)

	x = layers.Flatten()(x)
	x = layers.Dense(1024, activation="relu")(x)
	x = layers.Dense(512, activation="relu")(x)
	prediction = layers.Dense(10, activation='softmax')(x)
	model = models.Model(inputs=inputs, outputs=prediction)


	return model


def scattering_transform1d(n_classes, sequence_length):
	""" Scattering transform
	"""
	log_eps = 1e-6
	x_in = layers.Input(shape=(sequence_length))
	x = Scattering1D(8, 12)(x_in)
	x = layers.Lambda(lambda x: x[..., 1:, :])(x)
	x = layers.Lambda(lambda x: tf.math.log(tf.abs(x) + log_eps))(x)
	x = layers.GlobalAveragePooling1D(data_format='channels_first')(x)
	x = layers.BatchNormalization(axis=1)(x)
	x_out = layers.Dense(n_classes, activation='softmax')(x)
	model = tf.keras.models.Model(x_in, x_out)
	return model


def scattering_transform1d_big(n_classes, sequence_length):
	""" Scattering transform with more parameters
	"""
	log_eps = 1e-6
	x_in = layers.Input(shape=(sequence_length))
	x = Scattering1D(8, 12)(x_in)
	x = layers.Lambda(lambda x: x[..., 1:, :])(x)
	x = layers.Lambda(lambda x: tf.math.log(tf.abs(x) + log_eps))(x)
	x = layers.GlobalAveragePooling1D(data_format='channels_first')(x)
	x = layers.BatchNormalization(axis=1)(x)
	x = layers.Dense(512, activation='softmax')(x)
	x_out = layers.Dense(n_classes, activation='softmax')(x)
	model = tf.keras.models.Model(x_in, x_out)
	return model


def train_dl_model(path_to_data, save_model_path, epochs, algorithm):
	# X_train, X_test, y_train, y_test = split_data(path_to_data, 0.33)
	X_train, X_test, X_valid, y_train, y_test, y_valid =  split_data(path_to_data, 0.33, 42, 0.2, True)
	

	###################################################
	#NORMALIZE FOR SCATERRING TRANSFORM ONLY
	row_sums = np.array(X_train).sum(axis=1)
	X_train = X_train / row_sums[:, np.newaxis]

	row_sums = np.array(X_valid).sum(axis=1)
	X_valid = X_valid / row_sums[:, np.newaxis]

	row_sums = np.array(X_test).sum(axis=1)
	X_test = X_test / row_sums[:, np.newaxis]
	#####################################################

	y_train = to_categorical(y_train)
	y_test = to_categorical(y_test)
	y_valid = to_categorical(y_valid)
	
	model = globals()[algorithm["name"]](**algorithm["args"])
	model.summary()

	model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=["accuracy"])
	history = model.fit(np.array(X_train), np.array(y_train), batch_size=128, epochs=epochs, validation_data=(np.array(X_valid), np.array(y_valid)))
	model.save(save_model_path)

	hist_json_file = os.path.join(save_model_path, 'history.json')
	with open(hist_json_file, mode='w') as f:
	    pd.DataFrame(history.history).to_json(f, indent=4)



	# Evaluation
	pred = model.predict(np.array(X_test))
	print(np.mean(pred.argmax(1) == y_test.argmax(1)))





if __name__ == "__main__":
	np.random.seed(42)
	parser = argparse.ArgumentParser()
	parser.add_argument("-e", "--config_file", default="configs/config.yaml", type=str, help = "Path to the configuration file")

	args = parser.parse_args()
	dl_based_training = yaml.safe_load(open(args.config_file))["deep_learning_based_training"]
	# # features_based_model_train(**features_based_training)
	
	train_dl_model(**dl_based_training)
	