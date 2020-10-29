import yaml
import argparse
import os
from utils import split_data
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pickle as pkl

import time 
import datetime
import socket


def random_forest(X_train, y_train, n_estimators=100, criterion='gini', max_depth=None, n_jobs=1):
	model = RandomForestClassifier(n_jobs=n_jobs)
	model.fit(X_train, y_train)
	return model



def print_testing_training_accuracies(model, X_train, y_train, X_test, y_test):
	testing_preds = model.predict(X_test)
	testing_acc = np.mean(np.array(testing_preds) == np.array(y_test))
	training_preds = model.predict(X_train)
	training_acc = np.mean(np.array(training_preds) == np.array(y_train))

	print("Testing accuracy : {} ".format(testing_acc))
	print("Training accuracy : {} ".format(training_acc))



def features_based_model_train(path_to_data, save_model_path, algorithm):
	X_train, X_test, y_train, y_test = split_data(path_to_data, 0.33)
	
	X_train = [sample.flatten() for sample in X_train]
	X_test = [sample.flatten() for sample in X_test]

	t1 = time.time()
	model = globals()[algorithm["name"]](X_train, y_train, **algorithm["args"])
	t2 = time.time()

	if algorithm["args"]["n_jobs"] == -1:
		n_jobs = os.cpu_count()
	else:
		n_jobs = algorithm["args"]["n_jobs"]

	feature_name = path_to_data.split("/")

	with open("logs/logs.csv", "a") as myfile:
		myfile.write("{:%Y-%m-%d %H:%M:%S},Training {} for {},{},{},{:.2f}\n".format(datetime.datetime.now(),algorithm["name"],feature_name[2],socket.gethostname(),n_jobs,t2-t1))


	pkl.dump(model, open("{}{}_{}.pkl".format(save_model_path,algorithm["name"],feature_name[2]), "wb" )) 
	print_testing_training_accuracies(model, X_train, y_train, X_test, y_test)




if __name__ == "__main__":
	np.random.seed(42)
	parser = argparse.ArgumentParser()
	parser.add_argument("-e", "--config_file", default="configs/config.yaml", type=str, help = "Path to the configuration file")

	args = parser.parse_args()
	features_based_training = yaml.safe_load(open(args.config_file))["features_based_training"]

	features_based_model_train(**features_based_training)