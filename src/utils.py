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


def split_data(processed_data_path, test_size=0.33, random_state=42, validation_size=0.2,  validation_set=False):
	processed_users_data_filenames = get_users_processed_data_filenames(processed_data_path)

	# Use this to test
	#train_filenames = [train_filenames[0]]
	if validation_set:
		train_valid_filenames, test_filenames = train_test_split(processed_users_data_filenames, test_size=test_size)
		train_filenames, valid_filenames = train_test_split(train_valid_filenames, test_size=validation_size)
		X_train, X_test, X_valid, y_train, y_test, y_valid = [],[],[],[],[],[]

		for filename in train_filenames:
			user_data = pkl.load(open(os.path.join(processed_data_path, filename), "rb" ))
			X_train += list(user_data["data"])
			y_train += list(user_data["label"])

		for filename in test_filenames:
			user_data =  pkl.load(open(os.path.join(processed_data_path, filename), "rb" ))
			X_test += list(user_data["data"])
			y_test += list(user_data["label"])

		for filename in valid_filenames:
			user_data =  pkl.load(open(os.path.join(processed_data_path, filename), "rb" ))
			X_valid += list(user_data["data"])
			y_valid += list(user_data["label"])

		return X_train, X_test, X_valid, y_train, y_test, y_valid

	else:
		train_filenames, test_filenames = train_test_split(processed_users_data_filenames, test_size=test_size, random_state=random_state)
		X_train, X_test, y_train, y_test = [],[],[],[]

		for filename in train_filenames:
			user_data = pkl.load(open(os.path.join(processed_data_path, filename), "rb" ))
			X_train += list(user_data["data"])
			y_train += list(user_data["label"])

		for filename in test_filenames:
			user_data =  pkl.load(open(os.path.join(processed_data_path, filename), "rb" ))
			X_test += list(user_data["data"])
			y_test += list(user_data["label"])

		return X_train, X_test, y_train, y_test




def ensure_dir(path_to_dir, remove_if_exists = False):
    if remove_if_exists and os.path.exists(path_to_dir) and os.path.isdir(path_to_dir):
        shutil.rmtree(path_to_dir)
        
    if not os.path.exists(path_to_dir):
        os.makedirs(path_to_dir)

