import numpy as np
import os 
import pickle as pkl
from sklearn.model_selection import train_test_split
from  sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import ComplementNB, GaussianNB
from sklearn.svm import SVC


def get_users_processed_data_filenames(path):
	""" Returns a list of all the .pkl files at path
	"""
	processed_data_paths = []
	for file in sorted(os.listdir(path)):
		if file.endswith(".pkl"):
			processed_data_paths += [file]
	return processed_data_paths




if __name__ == "__main__":
	processed_data_path = "data/processed"
	processed_users_data_filenames = get_users_processed_data_filenames(processed_data_path)
	
	train_filenames, test_filenames = train_test_split(processed_users_data_filenames, test_size=0.33, random_state=42)

	X_train, X_test, y_train, y_test = [],[],[],[]

	for filename in train_filenames:
		user_data = pkl.load(open(os.path.join(processed_data_path, filename), "rb" ))
		X_train += list(user_data["data"])
		y_train += list(user_data["label"])

	for filename in test_filenames:
		user_data =  pkl.load(open(os.path.join(processed_data_path, filename), "rb" ))
		X_test += list(user_data["data"])
		y_test += list(user_data["label"])

	model = SVC()
	model.fit(X_train, y_train)

	pred_test = model.predict(X_test)
	print("Testing accuracy: " , np.mean(np.array(pred_test) == np.array(y_test)))


	pred_train = model.predict(X_train)
	print("Training accuracy: ", np.mean(np.array(pred_train) == np.array(y_train)))

	 