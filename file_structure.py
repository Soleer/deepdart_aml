import pandas as pd 
import os 

labels = pd.read_pickle("/home/jacob-relle/Dokumente/AML Projekt/labels_pkl/labels.pkl")

for dir_name in labels['img_folder']:
    if not os.path.exists(f'/home/jacob-relle/Dokumente/AML Projekt/cropped_labels/{dir_name}'):
        os.mkdir(f'/home/jacob-relle/Dokumente/AML Projekt/cropped_labels/{dir_name}')