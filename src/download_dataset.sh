#!/bin/sh

DATASET_CONFIG_FILE_PATH=configs/dataset_cfg.yaml

save_raw_data_pth=$(grep 'save_raw_data_path' $DATASET_CONFIG_FILE_PATH); 
save_raw_data_pth=${save_raw_data_pth//*save_raw_data_path: /}; 
save_raw_data_pth=${save_raw_data_pth//[[:blank:]]/};

svn checkout https://github.com/soerenab/AudioMNIST/trunk/data toto/raw/