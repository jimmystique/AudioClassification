import numpy as np
from  sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import ComplementNB, GaussianNB
from sklearn.svm import SVC
from utils import split_data

import pickle as pkl
import argparse
import yaml


def train_model(processed_data_path, model_save_path, test_size):
	X_train, X_test, y_train, y_test = split_data(processed_data_path, test_size)

	model = RandomForestClassifier()
	model.fit(X_train, y_train)
	pkl.dump(model, open(model_save_path, "wb"))



if __name__ == "__main__":
	np.random.seed(42)
	parser = argparse.ArgumentParser()
	parser.add_argument("-c", "--config_file", default="configs/config.yaml", type=str, help = "Path to the configuration file")

	args = parser.parse_args()
	training_cfg = yaml.safe_load(open(args.config_file))["training"]

	train_model(**training_cfg)
	
