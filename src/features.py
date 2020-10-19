import os
import pandas as pd
import pickle as pkl
import argparse
import yaml
import librosa

import multiprocessing
import time 

'''
extracting Mel-frequency cepstral coefficients (MFCCs) from preprocessed audio files.
MFCC can't be extracted from spectrograms. Takes an audio series as input
'''
def mfcc(audio_input_data_path, extracted_features_data_path, target_sr):
    print('extracting mfcc feature from preprocessed audifiles for {}'.format(audio_input_data_path))    
    user_df = pd.DataFrame(columns=['data', "user_id", "record_num", "label"])
    audio_input_data_path = pkl.load(open(audio_input_data_path, "rb" ))
    wav_user_id = 0

    for row in audio_input_data_path.iterrows():
        mfcc_features = librosa.feature.mfcc(row[1]['data'], sr=target_sr)
        
        wav_user_id = row[1]['user_id']
        new_row = {"data": mfcc_features, "user_id": wav_user_id, "record_num": row[1]['record_num'], "label": row[1]['label']}
        user_df = user_df.append(new_row, ignore_index=True)

    pkl.dump( user_df, open("{}.pkl".format(os.path.join(extracted_features_data_path, 'mfcc/', str(wav_user_id))), "wb" ) )

def spectrogram_descriptors(spectrograms_data_files, extracted_features_data_path, target_sr):
    print('extracting spectrogram descriptors for {}'.format(spectrograms_data_files))    
    user_df = pd.DataFrame(columns=['centroid', "bandwith", "flatness", "rolloff", "user_id", "record_num", "label"])
    audio_input_data_path = pkl.load(open(spectrograms_data_files, "rb" ))
    wav_user_id = 0

    for row in audio_input_data_path.iterrows():
        '''
        We only need the magnitude of the spectrogram to extract the features. 
        Seperating the spectrogram in two components. 
        '''
        magnitude, phase = librosa.magphase(row[1]['data'])
        centroid = librosa.feature.spectral_centroid(S=magnitude)
        bandwith = librosa.feature.spectral_bandwidth(S=magnitude)
        flatness = librosa.feature.spectral_flatness(S=magnitude)
        rolloff = librosa.feature.spectral_rolloff(S=magnitude, sr=target_sr)

        wav_user_id = row[1]['user_id']
        new_row = {"centroid": centroid[0], "bandwith": bandwith[0], "flatness": flatness[0], "rolloff": rolloff[0], "user_id": wav_user_id, "record_num": row[1]['record_num'], "label": row[1]['label']}
        user_df = user_df.append(new_row, ignore_index=True)

    pkl.dump( user_df, open("{}.pkl".format(os.path.join(extracted_features_data_path, 'spectrogramDescriptors/', str(wav_user_id))), "wb" ) )

def create_features(audio_input_data_path, representation_data_path, extracted_features_data_path, target_sr):
    #If a directory for features doesn't exist, create on
    if not os.path.exists(extracted_features_data_path):
        os.makedirs(extracted_features_data_path)
    
    processed_data_files = sorted([f.path for f in os.scandir(audio_input_data_path)])
    spectrograms_data_files = sorted([f.path for f in os.scandir(representation_data_path)])
    pool=multiprocessing.Pool(processes=10)

    #generating mfcc features
    if not os.path.exists(extracted_features_data_path+'mfcc/'):
        os.makedirs(extracted_features_data_path+'mfcc/')
    t1 = time.time()
    pool.starmap(mfcc, [[file, extracted_features_data_path, target_sr] for file in processed_data_files], chunksize=1)
    t2 = time.time()
    mfcc_time = t2-t1

    #generating spectrogram descriptors
    if not os.path.exists(extracted_features_data_path+'spectrogramDescriptors/'):
        os.makedirs(extracted_features_data_path+'spectrogramDescriptors/')

    t1 = time.time()
    pool.starmap(spectrogram_descriptors, [[file, extracted_features_data_path, target_sr] for file in spectrograms_data_files], chunksize=1)
    t2 = time.time()
    spectrogram_descriptors_time = t2-t1

    return mfcc_time, spectrogram_descriptors_time

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--extract_features_cfg", default="configs/config.yaml", type=str, help = "Path to the feature extraction configuration file")

    args = parser.parse_args()
    extract_features_cfg = yaml.safe_load(open(args.extract_features_cfg))["extract_features"]
    
    mfcc_time, spectrogram_descriptors_time = create_features(**extract_features_cfg)

    print("Time elapsed to extract mfcc features: {} seconds ".format(mfcc_time))
    print("Time elapsed to extract spectrogram descriptor features: {} seconds ".format(spectrogram_descriptors_time))

    #Generating mfcc : ~190.67s using Parallel computing with n_processed = 10
    #Extracting spectrogram descriptors : ~52.98s using Parallel computing with n_processed = 10