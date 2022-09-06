import pandas as pd 
import os 

labels = pd.read_pickle("./dataset/labels.pkl")

for dir_name in labels['img_folder']:
    if not os.path.exists(f'T:/deepdart_data/{dir_name}'):
        os.mkdir(f'T:/deepdart_data/{dir_name}')