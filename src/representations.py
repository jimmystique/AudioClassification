import os
import pandas as pd
import pickle as pkl
import argparse
import yaml
import librosa

import multiprocessing
import time 
import datetime
import socket

def create_spectrograms(preprocessed_audio_file, representation_data_path):
    print('creating spectograms for {}'.format(preprocessed_audio_file))    
    user_df = pd.DataFrame(columns=['data', "user_id", "record_num", "label"])
    preprocessed_audio_data = pkl.load(open(preprocessed_audio_file, "rb" ))
    wav_user_id = 0

    for row in preprocessed_audio_data.iterrows():
        audio_data_stft_format = librosa.stft(row[1]['data'])

        #For audio data, spectrograms should be in db magnitude. This also puts the data in log scale
        db_spectrogram = librosa.amplitude_to_db(abs(audio_data_stft_format))
        
        wav_user_id = row[1]['user_id']
        new_row = {"data": db_spectrogram, "user_id": wav_user_id, "record_num": row[1]['record_num'], "label": row[1]['label']}
        user_df = user_df.append(new_row, ignore_index=True)

    pkl.dump( user_df, open("{}.pkl".format(os.path.join(representation_data_path, str(wav_user_id))), "wb" ) )

def create_representations(input_data_path, representation_data_path):
    #If a directory for spectrograms doesn't exist, create on
    if not os.path.exists(representation_data_path):
        os.makedirs(representation_data_path)

    processed_data_files = sorted([f.path for f in os.scandir(input_data_path)])
    pool=multiprocessing.Pool(processes=10)
    
    #generating spectrograms
    t1 = time.time()
    pool.starmap(create_spectrograms, [[file, representation_data_path] for file in processed_data_files], chunksize=1)
    t2 = time.time()
    with open("logs/logs.csv", "a") as myfile:
        myfile.write("{:%Y-%m-%d %H:%M:%S},{},{},{},{:.2f}\n".format(datetime.datetime.now(),"generate spectrograms",socket.gethostname(),pool._processes,t2-t1))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--create_respresentations_cfg", default="configs/config.yaml", type=str, help = "Path to the representation creation configuration file")

    args = parser.parse_args()
    create_respresentations_cfg = yaml.safe_load(open(args.create_respresentations_cfg))["create_respresentations"]

    create_representations(**create_respresentations_cfg)