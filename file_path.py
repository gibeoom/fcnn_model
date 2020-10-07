import os
import numpy as np

ori_path = "/Users/gimgibeom/Documents/maru_data/"
male_path = ori_path + "male"
female_path = ori_path + "female"
paths = [male_path, female_path]

path_str = ''
dir_list = os.listdir(male_path)
for dir in dir_list:
    file_list = os.listdir(male_path + '/' + dir)
    for file in file_list:
        path_str += male_path + '/' + dir + '/' + file + '\n'

path_list = path_str.split()
np.random.shuffle(path_list)
train_paths = path_list[:-(int(len(path_list)*0.1))]
test_paths = path_list[-(int(len(path_list)*0.1)):]

f = open(ori_path + 'male_files_train.txt', 'w')
f.write('\n'.join(train_paths))
print('male train set size : ', len(train_paths))
f.close()

f = open(ori_path + 'male_files_test.txt', 'w')
f.write('\n'.join(test_paths))
print('male test set size : ', len(test_paths))
f.close()

path_str = ''
dir_list = os.listdir(female_path)
for dir in dir_list:
    file_list = os.listdir(female_path + '/' + dir)
    for file in file_list:
        path_str += female_path + '/' + dir + '/' + file + '\n'

path_list = path_str.split()
np.random.shuffle(path_list)
train_paths = path_list[:-(int(len(path_list)*0.1))]
test_paths = path_list[-(int(len(path_list)*0.1)):]

f = open(ori_path + 'female_files_train.txt', 'w')
f.write('\n'.join(train_paths))
print('female train set size : ', len(train_paths))
f.close()

f = open(ori_path + 'female_files_test.txt', 'w')
f.write('\n'.join(test_paths))
print('female test set size : ', len(test_paths))
f.close()