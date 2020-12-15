from utils import get_users_processed_data_filenames
import pickle as pkl
import os

split = {	"train":[  
					 set([28, 56,  7, 19, 35,  1,  6, 16, 23, 34, 46, 53, 36, 57,  9, 24, 37,  2, 8, 17, 29, 39, 48, 54, 43, 58, 14, 25, 38,  3, 10, 20, 30, 40, 49, 55]),
					 set([36, 57,  9, 24, 37,  2,  8, 17, 29, 39, 48, 54, 43, 58, 14, 25, 38,  3,  10, 20, 30, 40, 49, 55, 12, 47, 59, 15, 27, 41,  4, 11, 21, 31, 44, 50]),
					 set([43, 58, 14, 25, 38,  3, 10, 20, 30, 40, 49, 55, 12, 47, 59, 15, 27, 41, 4, 11, 21, 31, 44, 50, 26, 52, 60, 18, 32, 42,  5, 13, 22, 33, 45, 51]),
					 set([12, 47, 59, 15, 27, 41,  4, 11, 21, 31, 44, 50, 26, 52, 60, 18, 32, 42, 5, 13, 22, 33, 45, 51, 28, 56,  7, 19, 35,  1,  6, 16, 23, 34, 46, 53]),
					 set([26, 52, 60, 18, 32, 42,  5, 13, 22, 33, 45, 51, 28, 56,  7, 19, 35,  1, 6, 16, 23, 34, 46, 53, 36, 57,  9, 24, 37,  2,  8, 17, 29, 39, 48, 54])],

			"validate":[set([12, 47, 59, 15, 27, 41,  4, 11, 21, 31, 44, 50]),
						set([26, 52, 60, 18, 32, 42,  5, 13, 22, 33, 45, 51]),
						set([28, 56,  7, 19, 35,  1,  6, 16, 23, 34, 46, 53]),
						set([36, 57,  9, 24, 37,  2,  8, 17, 29, 39, 48, 54]),
						set([43, 58, 14, 25, 38,  3, 10, 20, 30, 40, 49, 55])],

			"test":[    set([26, 52, 60, 18, 32, 42,  5, 13, 22, 33, 45, 51]),
						set([28, 56,  7, 19, 35,  1,  6, 16, 23, 34, 46, 53]),
						set([36, 57,  9, 24, 37,  2,  8, 17, 29, 39, 48, 54]),
						set([43, 58, 14, 25, 38,  3, 10, 20, 30, 40, 49, 55]),
						set([12, 47, 59, 15, 27, 41,  4, 11, 21, 31, 44, 50])]}
if __name__ == "__main__":
	extracted_features_dir = [ 
								# "data/preprocessed_fft",
								# "data/preprocessed_downsampling",

								# "data/extracted_features/downsampling_processing/chroma_stft",
							 	# "data/extracted_features/downsampling_processing/mfcc",
								# "data/extracted_features/downsampling_processing/root_mean_square",
								# "data/extracted_features/downsampling_processing/spectrogram",
								# "data/extracted_features/downsampling_processing/spectrogram_bandwith",
								# "data/extracted_features/downsampling_processing/spectrogram_centroid",
								# "data/extracted_features/downsampling_processing/spectrogram_flatness",
								# "data/extracted_features/downsampling_processing/spectrogram_rolloff",

								"data/extracted_features/fft_processing/chroma_stft",
							    "data/extracted_features/fft_processing/mfcc",
								"data/extracted_features/fft_processing/root_mean_square",
								"data/extracted_features/fft_processing/spectrogram",
								"data/extracted_features/fft_processing/spectrogram_bandwith",
								"data/extracted_features/fft_processing/spectrogram_centroid",
								"data/extracted_features/fft_processing/spectrogram_flatness",
								"data/extracted_features/fft_processing/spectrogram_rolloff",
	]
	for path_to_data in extracted_features_dir:
		if not os.path.exists(path_to_data):
			raise ValueError("Path ", path_to_data, " do not exist")


	for path_to_data in extracted_features_dir:
		pth_spl = path_to_data.split("/")
		save_features_path = os.path.join("data/train_val_test_split/", "/".join(pth_spl[1:]))
		if not os.path.exists(save_features_path):
			os.makedirs(save_features_path)
		files = get_users_processed_data_filenames(path_to_data)
		for i in range(len(split["train"])):
			train_users_ids = split["train"][i]
			val_users_ids = split["validate"][i]
			tests_users_ids = split["test"][i]
			X_train, y_train, X_val, y_val, X_test, y_test = [], [], [], [], [], []

			for filename in files:
				user_id = int(filename.split("_")[0])
				if user_id in train_users_ids:
					print(user_id, " in user_id for split ", i)
					user_data = pkl.load(open(os.path.join(path_to_data, filename), "rb" ))
					X_train += list(user_data["data"])
					y_train += list(user_data["label"])
				elif user_id in val_users_ids:
					user_data = pkl.load(open(os.path.join(path_to_data, filename), "rb" ))
					X_val += list(user_data["data"])
					y_val += list(user_data["label"])
				elif user_id in tests_users_ids:
					user_data = pkl.load(open(os.path.join(path_to_data, filename), "rb" ))
					X_test += list(user_data["data"])
					y_test += list(user_data["label"])

			
			pkl.dump({"X_train": X_train, "y_train": y_train, "X_val": X_val, "y_val": y_val, "X_test": X_test, "y_test":y_test}, open(os.path.join(save_features_path, f"split_{i}.pkl"), "wb" )) 