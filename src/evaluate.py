import pickle as pkl
import numpy as np 
import argparse
import yaml
from utils import split_data


def evaluate_model(load_model_path, input_data_path, test_size, configuration):
	X_train, X_test, y_train, y_test = split_data(input_data_path, test_size)
	model = pkl.load(open(load_model_path, "rb" ))

	flatten_train = []
	flatten_test = []

	for input in X_train:
		flatten_train.append(input.flatten())
	
	for input in X_test:
		flatten_test.append(input.flatten())
	
	pred_test = model.predict(flatten_test)
	print("Testing accuracy for {}: {}".format(configuration,np.mean(np.array(pred_test) == np.array(y_test))))

	pred_train = model.predict(flatten_train)
	print("Training accuracy for {}: {}".format(configuration,np.mean(np.array(pred_train) == np.array(y_train))))


if __name__ == "__main__":
	np.random.seed(42)

	parser = argparse.ArgumentParser()
	parser.add_argument("-c", "--config_file", default="configs/config.yaml", type=str, help = "Path to the configuration file")

	args = parser.parse_args()
	evaluation_cfg = yaml.safe_load(open(args.config_file))["evaluation"]

	#Evaluate spectrogram rf
	evaluate_model(evaluation_cfg['spectrogram_rf_model_save_path'], evaluation_cfg['spectrogram_data_path'], evaluation_cfg['test_size'], "spectrogram respresentation with randon forest")

	#Evaluate mfcc rf
	evaluate_model(evaluation_cfg['mfcc_rf_model_save_path'], evaluation_cfg['mfcc_data_path'], evaluation_cfg['test_size'], "MFCC with randon forest")

	#Evaluate spectrogram descriptor rf
	evaluate_model(evaluation_cfg['spectrogram_descriptors_rf_model_save_path'], evaluation_cfg['spectrogram_descriptors_data_path'], evaluation_cfg['test_size'], "spectrogram respresentation with randon forest")
	
	# load_model_path = "models/model.pkl"
	# model = pkl.load(open(load_model_path, "rb" ))




	
