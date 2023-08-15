import numpy, os

def load_training_files_list():
    with open('MissingDataOpenData/data_splits/training.txt', 'r') as f:
        files = ['%s.jpg'%(file.replace('\n', '')) for file in f.readlines()]
    return files