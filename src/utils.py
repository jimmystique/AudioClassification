import os
from sklearn.model_selection import train_test_split
import pickle as pkl

def get_users_processed_data_filenames(path):
	""" Returns a list of all the .pkl files at path
	"""
	processed_data_paths = []
	for file in sorted(os.listdir(path)):
		if file.endswith(".pkl"):
			processed_data_paths += [file]
	return processed_data_paths


def split_data(processed_data_path, test_size=0.33, random_state=42):
	processed_users_data_filenames = get_users_processed_data_filenames(processed_data_path)
	train_filenames, test_filenames = train_test_split(processed_users_data_filenames, test_size=test_size)
	X_train, X_test, y_train, y_test = [],[],[],[]

	# Use this to test
	train_filenames = [train_filenames[0]]
	for filename in train_filenames:
		user_data = pkl.load(open(os.path.join(processed_data_path, filename), "rb" ))
		X_train += list(user_data["data"])
		y_train += list(user_data["label"])

	for filename in test_filenames:
		user_data =  pkl.load(open(os.path.join(processed_data_path, filename), "rb" ))
		X_test += list(user_data["data"])
		y_test += list(user_data["label"])
	print(test_filenames)
	return X_train, X_test, y_train, y_test