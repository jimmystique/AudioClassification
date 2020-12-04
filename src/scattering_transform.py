import multiprocessing
import os
from scipy.io import wavfile
import os
import pandas as pd
import pickle as pkl
import numpy as np
import argparse
import yaml

import librosa
import scipy
from scipy.io import wavfile
import multiprocessing

import time 
import datetime
import socket


import scipy
def resample_wav_data(wav_data, orig_sr, target_sr):
	""" Resample wav_data from sampling rate equals to orig_sr to a new sampling rate equals to target_sr
	"""
	# resampled_wav_data = librosa.core.resample(y=wav_data.astype(np.float32), orig_sr=orig_sr, target_sr=target_sr)
	resampled_wav_data = scipy.signal.resample(wav_data, target_sr)
	# print(wav_data, resampled_wav_data, len(resampled_wav_data))
	return resampled_wav_data 



def pad_wav_data(wav_data, target_sr):
	""" Pad wav data 
	"""
	if (len(wav_data) > target_sr):
		raise ValueError("")
	elif len(wav_data) < target_sr:
		padded_wav_data = np.zeros(target_sr)
		starting_point = np.random.randint(low=0, high=target_sr-len(wav_data))
		padded_wav_data[starting_point:starting_point+len(wav_data)] = wav_data
	else:
		padded_wav_data = wav_data

	return padded_wav_data


def preprocess_user_data_at_pth(user_data_path, preprocessed_data_path, target_sr):
	print(user_data_path)
	user_df = pd.DataFrame(columns=['data', "user_id", "record_num", "label"])
	wav_user_id = 0
	for file in sorted(os.listdir(user_data_path)):
		if file.endswith(".wav"):
			wav_label, wav_user_id, wav_record_n =  os.path.splitext(file)[0].split("_")
			wav_sr, wav_data = wavfile.read(os.path.join(user_data_path, file))

			# resampled_wav_data = resample_wav_data(wav_data, wav_sr, target_sr)
			# scipy.io.wavfile.write(os.path.join("test_wav", file), 8000, resampled_wav_data)

			y,s = librosa.load(os.path.join(user_data_path, file), 8000)
			# scipy.io.wavfile.write(os.path.join("test_wav", "librosa" + file), 8000, y)

			padded_wav_data = pad_wav_data(y, target_sr)

			new_row = {"data": padded_wav_data, "user_id": wav_user_id, "record_num": wav_record_n, "label": wav_label}
			user_df = user_df.append(new_row, ignore_index=True)
	pkl.dump(user_df, open("{}_preprocessed.pkl".format(os.path.join(preprocessed_data_path, str(wav_user_id))), "wb" ) )




if __name__ == "__main__":
	raw_data_path = "data/raw/"
	preprocessed_data_path = "data/scattering_preprocessed/"
	n_process =  10
	target_sr = 8000
	
	if not os.path.exists(preprocessed_data_path):
		os.makedirs(preprocessed_data_path)

	users_data_path = sorted([folder.path for folder in os.scandir(raw_data_path) if folder.is_dir() and any(file.endswith(".wav") for file in os.listdir(folder))])
	print(users_data_path)


	pool=multiprocessing.Pool(n_process)
	pool.starmap(preprocess_user_data_at_pth, [[folder, preprocessed_data_path, target_sr] for folder in users_data_path if os.path.isdir(folder)], chunksize=1)


