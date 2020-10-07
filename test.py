import pandas as pd
import numpy as np
import os
from data_processing.maru_voice import MaruDataset

ori_path = "/Users/gimgibeom/Documents/maru_data/"
# rawData = pd.read_csv(os.path.join(ori_path, "file_list.txt"))
# print(np.transpose(rawData).values[0])

mcv = MaruDataset(ori_path, val_dataset_size=200)
clean_train_filenames, clean_val_filenames = mcv.get_train_val_filenames()
mcv.get_test_filenames()
#print(clean_train_filenames)