from librosa.feature import chroma_stft, rms, mfcc, spectral_centroid
from librosa import feature
import argparse
import yaml
import os
import multiprocessing
import pickle as pkl
import numpy as np
from utils import ensure_dir



def chroma_stft(processed_data_path, save_path, n_processes, sr=22050, S=None,  n_fft=2048, hop_length=512, win_length=None, window='hann', center=True, pad_mode='reflect', tuning=None, n_chroma=12):
	""" Extract chroma features using STFT on all files at processed_data_path and save the extracted features at save_path

	Args:
		processed_data_path (str): Path to the directory containing the processed data
		save_path (str): Path to the directory where to save the extracted features
		n_processed (int): Number of processed to run at the same time to exctract features faster
		sr (inter): sampling rate
		S (np.ndarray): power spectrogram 
		norm (float or None): column-wise normalization
		n_fft (int): FFT window size
		hop_length (int): hop length
		win_length (int): Each frame of audio is windowed by window(). The window will be of length win_length and then padded with zeros to match n_fft.
		window (string, tuple, number, function, or np.ndarray [shape=(n_fft,)]): - a window specification (string, tuple, or number); see scipy.signal.get_window
																				  - a window function, such as scipy.signal.windows.hann
																				  - a vector or array of length n_fft
		center (bool): - if True, the signal y is padded so that frame t is centered at y[t * hop_length].
					   - if False, then frame t begins at y[t * hop_length]
		pad_mode (str): If center=True, the padding mode to use at the edges of the signal. By default, STFT uses reflection padding.
		tuning (float): Deviation from A440 tuning in fractional chroma bins. If None, it is automatically estimated.
		n_chroma (int): Number of chroma bins to produce (12 by default).
	"""
	print("Extracting Chroma Features with Short Time Fourier Transform ...")
	ensure_dir(save_path)
	processed_data_files = sorted([f.path for f in os.scandir(processed_data_path)])
	pool=multiprocessing.Pool(processes=n_processes)
	pool.starmap(_chroma_stft, [[processed_file_path, save_path, sr, S, n_fft, hop_length, win_length, window, center, pad_mode, tuning, n_chroma] for processed_file_path in processed_data_files], chunksize=1)



def _chroma_stft(processed_file_path, save_path, sr, S, n_fft, hop_length, win_length, window, center, pad_mode, tuning, n_chroma):
	""" Extract chroma features for the file at processed_file_path and save the features extracted at save_path
	Args:
		processed_data_path (str): Path to the directory containing the processed data
		save_path (str): Path to the directory where to save the extracted features
		sr (inter): sampling rate
		S (np.ndarray): power spectrogram 
		norm (float or None): column-wise normalization
		n_fft (int): FFT window size
		hop_length (int): hop length
		win_length (int): Each frame of audio is windowed by window(). The window will be of length win_length and then padded with zeros to match n_fft.
		window (string, tuple, number, function, or np.ndarray [shape=(n_fft,)]): - a window specification (string, tuple, or number); see scipy.signal.get_window
																				  - a window function, such as scipy.signal.windows.hann
																				  - a vector or array of length n_fft
		center (bool): - if True, the signal y is padded so that frame t is centered at y[t * hop_length].
					   - if False, then frame t begins at y[t * hop_length]
		pad_mode (str): If center=True, the padding mode to use at the edges of the signal. By default, STFT uses reflection padding.
		tuning (float): Deviation from A440 tuning in fractional chroma bins. If None, it is automatically estimated.
		n_chroma (int): Number of chroma bins to produce (12 by default).
	"""
	processed_data = pkl.load(open(processed_file_path, "rb" ))
	extracted_features = processed_data.copy(deep=True)

	for index, row in processed_data.iterrows():
		data = row["data"]
		data_extracted_features = feature.chroma_stft(y=data, sr=sr, S=S, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, center=center, pad_mode=pad_mode, tuning=tuning, n_chroma=n_chroma)
		extracted_features.loc[index, "data"] = data_extracted_features

	save_filename = "{}_chroma_stft_features.pkl".format(os.path.splitext(os.path.basename(processed_file_path))[0].split("_")[0])
	save_file_path = os.path.join(save_path, save_filename)
	pkl.dump(extracted_features, open(save_file_path, "wb" ) )
	print("- Chroma stft features extraction on {} Saved in {}".format(processed_file_path, save_file_path))



def root_mean_square(processed_data_path, save_path, n_processes, S=None, frame_length=2048, hop_length=512, center=True, pad_mode='reflect'):
	print("Extracting features with Root Mean Square ...")
	ensure_dir(save_path)
	processed_data_files = sorted([f.path for f in os.scandir(processed_data_path)])
	pool=multiprocessing.Pool(processes=n_processes)
	pool.starmap(_root_mean_square, [[processed_file_path, save_path, S, frame_length, hop_length, center, pad_mode] for processed_file_path in processed_data_files], chunksize=1)



def _root_mean_square(processed_file_path, save_path, S, frame_length, hop_length, center, pad_mode):
	processed_data = pkl.load(open(processed_file_path, "rb" ))
	extracted_features = processed_data.copy(deep=True)

	for index, row in processed_data.iterrows():
		data = row["data"]
		data_extracted_features = feature.rms(y=data, S=S, frame_length=frame_length, hop_length=hop_length, center=center, pad_mode=pad_mode)
		extracted_features.loc[index, "data"] = data_extracted_features

	save_filename = "{}_rms_features.pkl".format(os.path.splitext(os.path.basename(processed_file_path))[0].split("_")[0])
	save_file_path = os.path.join(save_path, save_filename)
	pkl.dump(extracted_features, open(save_file_path, "wb" ) )
	print("- RMS features extraction on {} Saved in {}".format(processed_file_path, save_file_path))



def mfcc(processed_data_path, save_path, n_processes,  sr=22050, S=None, n_mfcc=20, dct_type=2, norm='ortho', lifter=0):
	print("Extracting features with MFCC ...")
	ensure_dir(save_path)
	processed_data_files = sorted([f.path for f in os.scandir(processed_data_path)])
	print(processed_data_files)
	pool=multiprocessing.Pool(processes=n_processes)
	pool.starmap(_mfcc, [[processed_file_path, save_path, sr, S,  n_mfcc, dct_type, norm, lifter] for processed_file_path in processed_data_files], chunksize=1)



def _mfcc(processed_file_path, save_path, sr, S, n_mfcc, dct_type, norm, lifter):
	processed_data = pkl.load(open(processed_file_path, "rb" ))
	extracted_features = processed_data.copy(deep=True)

	for index, row in processed_data.iterrows():
		data = row["data"]
		data_extracted_features = feature.mfcc(y=data, sr=sr, n_mfcc=n_mfcc, dct_type=dct_type, norm=norm, lifter=lifter)
		extracted_features.loc[index, "data"] = data_extracted_features
	save_filename = "{}_mfcc_features.pkl".format(os.path.splitext(os.path.basename(processed_file_path))[0].split("_")[0])
	save_file_path = os.path.join(save_path, save_filename)
	pkl.dump(extracted_features, open(save_file_path, "wb" ) )
	print("- MFCC features extraction on {} Saved in {}".format(processed_file_path, save_file_path))


 



def extract_features(processed_data_path, save_path, n_processes, algorithm):
	""" Extract features from files at processed_data_path and save the extracted features found at save_path

	Args:
		processed_data_path (str): Path to the directory containing the processed data
		save_path (str): Path to the directory where to save the extracted features
		n_processed (int): Number of processed to run at the same time to exctract features faster
		algorithm (dict): Dictionary containing a key "name" (corresponding to the name of a function that will be call to build a model) and a key "args" containing the hyperparameters of the model to be built.
	"""
	print(processed_data_path)
	print(algorithm)

	globals()[algorithm["name"]](processed_data_path, save_path, n_processes, **algorithm["args"])





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--config_file", default="configs/config.yaml", type=str, help = "Path to the configuration file")

    args = parser.parse_args()
    features_extraction_cfg = yaml.safe_load(open(args.config_file))["features_extraction"]
    	
    print(features_extraction_cfg)
    extract_features(**features_extraction_cfg)