import pandas as pd 
import os 

labels = pd.read_pickle("labels.pkl")

for dir_name in labels['img_folder']:
    if not os.path.exists(f'deepdart_data_ring/{dir_name}'):
        os.mkdir(f'deepdart_data_ring/{dir_name}')