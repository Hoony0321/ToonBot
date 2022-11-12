from static import DATA_PATH
import csv
import numpy as np
import tensorflow as tf

LABEL_FILE_PATH = DATA_PATH + "/labels.csv"
WEBTOON_FILE_PATH = DATA_PATH + "/webtoons.csv"
BRIEF_IDX = 5

def getCsvReader(filename):
    f = open(filename, 'r', encoding='utf-8')
    next(f) # Header 건너뛰기
    return csv.reader(f)

# Label 데이터 가져오기
def loadLabelData():
    csvReader = getCsvReader(LABEL_FILE_PATH)
    data = np.array([int(row[1]) for row in csvReader]).reshape(-1,1)
    return data

#Webtoon 데이터 가져오기
def loadWebtoonData():
    csvReader = getCsvReader(WEBTOON_FILE_PATH)
    data = np.array([row[BRIEF_IDX] for row in csvReader]).reshape(-1,1) #소개글 칼럼만 뽑기
    return data

def loadDataset():
    labels = loadLabelData()
    webtoons = loadWebtoonData()
    dataset = tf.data.Dataset.from_tensor_slices((webtoons, labels))
    return dataset


def get_dataset_partitions_tf(ds, ds_size, train_split=0.8, val_split=0.2, shuffle=True, shuffle_size=10000):
    if shuffle:
        ds = ds.shuffle(shuffle_size, seed=12)

    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)

    train_ds = ds.take(train_size)
    val_ds = ds.skip(train_size).take(val_size)

    return train_ds, val_ds

dataset = loadDataset()
trainSet, validationSet = get_dataset_partitions_tf(dataset, len(dataset), shuffle=True)
print(len(trainSet))