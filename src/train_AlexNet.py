import os
import time 
import datetime
import socket
import pickle as pkl
import argparse
import yaml

from utils import split_data
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPool2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
import tensorflow as tf


def process_images(image, label):
	# Normalize images to have a mean of 0 and standard deviation of 1
	image = tf.image.per_image_standardization(image)
	# Resize images to 224x224
	image = tf.image.resize(image, (224,224))
	return image, label

def format_data(X_train, X_test, X_valid, y_train, y_test, y_valid):
	X_train = np.array(X_train)
	X_test = np.array(X_test)
	X_valid = np.array(X_valid)
	y_train = np.array(y_train)
	y_test = np.array(y_test)
	y_valid = np.array(y_valid)

	#One hot encoding classes array
	y_train = tf.one_hot(y_train, 10)
	y_test = tf.one_hot(y_test, 10)
	y_valid = tf.one_hot(y_valid, 10)

	#Making the images 3 dimensions
	X_train = X_train[..., np.newaxis]
	X_test = X_test[..., np.newaxis]
	X_valid = X_valid[..., np.newaxis]

	#Create Tensorflow dataset representation 
	train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
	test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
	valid_ds = tf.data.Dataset.from_tensor_slices((X_valid, y_valid))

	#Dataset partition sizes
	train_ds_size = tf.data.experimental.cardinality(train_ds).numpy()
	test_ds_size = tf.data.experimental.cardinality(test_ds).numpy()
	valid_ds_size = tf.data.experimental.cardinality(valid_ds).numpy()

	#Data processing pipeline
	train_ds = (train_ds
				.map(process_images)
				.shuffle(buffer_size=train_ds_size)
				.batch(batch_size=32, drop_remainder=True))
	test_ds = (test_ds
					.map(process_images)
					.shuffle(buffer_size=train_ds_size)
					.batch(batch_size=32, drop_remainder=True))
	valid_ds = (valid_ds
					.map(process_images)
					.shuffle(buffer_size=train_ds_size)
					.batch(batch_size=32, drop_remainder=True))


	return train_ds, test_ds, valid_ds

def train_alexNet_model(processed_data_path, model_save_path, test_size, validation_size, input_size, n_classes, epochs):
	X_train, X_test, X_valid, y_train, y_test, y_valid = split_data(processed_data_path, test_size, 42, validation_size, True)
	train_ds, test_ds, valid_ds = format_data(X_train, X_test, X_valid, y_train, y_test, y_valid)

	#Model architecture 
	model = tf.keras.models.Sequential([
		# 1st Convolutional Layer
		Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=input_size, padding="valid"),
		MaxPool2D(pool_size=(2,2), strides=(2,2),padding="valid"),
		BatchNormalization(),

		# 2nd Convolutional Layer
		Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), activation='relu', padding="valid"),
		MaxPool2D(pool_size=(2,2), strides=(2,2), padding="valid"),
		BatchNormalization(),

		# 3rd Convolutional Layer
		Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="valid"),
		BatchNormalization(),

		# 4th Convolutional Layer
		Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="valid"),
		BatchNormalization(),

		# 5th Convolutional Layer
		Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="valid"),
		BatchNormalization(),
		MaxPool2D(pool_size=(2,2), strides=(2,2), padding="valid"),

		Flatten(),

		#1st Dense Layer
		Dense(4096, activation='relu', input_shape=input_size),
		Dropout(0.4),

		#2nd Dense Layer
		Dense(4096, activation='relu'),
		Dropout(0.4),

		#3rd Dense Layer
		Dense(1000, activation='relu'),
		Dropout(0.4),
		Dense(n_classes, activation='softmax')
	])

	model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
	t1 = time.time()	
	model.fit(train_ds,
		epochs=epochs,
		validation_data=valid_ds)
	t2 = time.time()

	model.save(model_save_path)


	return t2-t1

if __name__ == "__main__":
	np.random.seed(42)
	parser = argparse.ArgumentParser()
	parser.add_argument("-c", "--config_file", default="configs/config.yaml", type=str, help = "Path to the configuration file")

	args = parser.parse_args()
	training_cfg = yaml.safe_load(open(args.config_file))["features_based_training"]
	if not os.path.exists(training_cfg['save_model_path']):
		os.makedirs(training_cfg['save_model_path'])

	algorithm_args = training_cfg['algorithm']['args']
	input_size_tuple = tuple(algorithm_args['input_size'])
	feature_type = training_cfg['path_to_data'].split("/")
	model_path_name = "{}{}_{}.h5".format(training_cfg['save_model_path'],training_cfg['algorithm']['name'],feature_type[-1])

	#train AlexNet
	print("Training AlexNet for "+str(feature_type[-1]))
	runTime = train_alexNet_model(training_cfg['path_to_data'], model_path_name, test_size=algorithm_args['test_size'], validation_size=algorithm_args['validation_size'], input_size=input_size_tuple, n_classes=algorithm_args['n_classes'], epochs=algorithm_args['epochs'])
	with open("logs/logs.csv", "a") as myfile:
		myfile.write("{:%Y-%m-%d %H:%M:%S},{},{},{},{:.2f}\n".format(datetime.datetime.now(),"Training AlexNet for "+str(feature_type[-1]),socket.gethostname(),os.cpu_count(),runTime))
