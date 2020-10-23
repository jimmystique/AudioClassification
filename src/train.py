import numpy as np
from  sklearn.ensemble import RandomForestClassifier
from utils import split_data

import pickle as pkl
import argparse
import yaml
import time 
import datetime
import socket
import os

def train_rf_model(processed_data_path, model_save_path, test_size, n_estimators, min_samples_split, max_depth=None):
	X_train, X_test, y_train, y_test = split_data(processed_data_path, test_size)

	flatten_train = []

	for input in X_train:
		flatten_train.append(input.flatten())

	t1 = time.time()
	model = RandomForestClassifier(n_estimators=n_estimators, min_samples_split=min_samples_split, max_depth=max_depth, n_jobs=-1)
	model.fit(flatten_train, y_train)
	pkl.dump(model, open(model_save_path, "wb"))
	t2 = time.time()

	return t2-t1

if __name__ == "__main__":
	np.random.seed(42)
	parser = argparse.ArgumentParser()
	parser.add_argument("-c", "--config_file", default="configs/config.yaml", type=str, help = "Path to the configuration file")

	args = parser.parse_args()
	training_cfg = yaml.safe_load(open(args.config_file))["training"]
	if not os.path.exists(training_cfg['model_save_path']):
		os.makedirs(training_cfg['model_save_path'])

	#train spectrogram
	print("Training random forest for spectrogram")
	runTime = train_rf_model(training_cfg['spectrogram_data_path'], training_cfg['spectrogram_rf_model_save_path'], training_cfg['test_size'], training_cfg['spectrogram_rf_n_estimators'], training_cfg['spectrogram_rf_min_samples_split'])
	with open("logs/logs.csv", "a") as myfile:
		myfile.write("{:%Y-%m-%d %H:%M:%S},{},{},{},{:.2f}\n".format(datetime.datetime.now(),"train random forest for spectrogram",socket.gethostname(),os.cpu_count(),runTime))

	#train mfcc
	print("Training random forest for mfcc")
	runTime = train_rf_model(training_cfg['mfcc_data_path'], training_cfg['mfcc_rf_model_save_path'], training_cfg['test_size'], training_cfg['mfcc_rf_n_estimators'], training_cfg['mfcc_rf_min_samples_split'])
	with open("logs/logs.csv", "a") as myfile:
		myfile.write("{:%Y-%m-%d %H:%M:%S},{},{},{},{:.2f}\n".format(datetime.datetime.now(),"train random forest for mfcc",socket.gethostname(),os.cpu_count(),runTime))

	#train spectrogram descriptors
	print("Training random forest for spectrogram descriptors")
	runTime = train_rf_model(training_cfg['spectrogram_descriptors_data_path'], training_cfg['spectrogram_descriptors_rf_model_save_path'], training_cfg['test_size'], training_cfg['spectrogram_descriptors_rf_n_estimators'], training_cfg['spectrogram_descriptors_rf_min_samples_split'], training_cfg['spectrogram_descriptors_rf_max_depth'])
	with open("logs/logs.csv", "a") as myfile:
		myfile.write("{:%Y-%m-%d %H:%M:%S},{},{},{},{:.2f}\n".format(datetime.datetime.now(),"train random forest for spectrogram descriptors",socket.gethostname(),os.cpu_count(),runTime))
