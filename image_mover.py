#move all training images to a folder

#read data splits text file
import shutil


IMAGE_PATH = "MissingDataOpenData/originals"
with open('MissingDataOpenData/data_splits/training.txt') as f:
    content = f.readlines()
    content = [x.strip() for x in content]
    print(content)
    for filename in content:
        shutil.copy(IMAGE_PATH + "/" + filename + '.jpg', "MissingDataOpenData/training")