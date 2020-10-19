import pickle as pkl
import numpy as np 
import argparse
import yaml
from utils import split_data


def evaluate_model(load_model_path, input_data_path, test_size):
	X_train, X_test, y_train, y_test = split_data(input_data_path, test_size)
	model = pkl.load(open(load_model_path, "rb" ))

	pred_test = model.predict(X_test)
	print("Testing accuracy: " , np.mean(np.array(pred_test) == np.array(y_test)))

	pred_train = model.predict(X_train)
	print("Training accuracy: ", np.mean(np.array(pred_train) == np.array(y_train)))


if __name__ == "__main__":
	np.random.seed(42)

	parser = argparse.ArgumentParser()
	parser.add_argument("-c", "--config_file", default="configs/config.yaml", type=str, help = "Path to the configuration file")

	args = parser.parse_args()
	evaluation_cfg = yaml.safe_load(open(args.config_file))["evaluation"]

	evaluate_model(**evaluation_cfg)
	print(evaluation_cfg)
	# load_model_path = "models/model.pkl"
	# model = pkl.load(open(load_model_path, "rb" ))




	
