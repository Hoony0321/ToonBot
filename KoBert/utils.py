from enum import Enum

from static import DATA_PATH
from sklearn.preprocessing import  OrdinalEncoder, OneHotEncoder
import csv
import numpy as np
import tensorflow as tf
import pandas as pd
import urllib.request
import cv2
import matplotlib.pyplot as plt

LABEL_FILE_PATH = DATA_PATH + "/labels.csv"
WEBTOON_FILE_PATH = DATA_PATH + "/webtoons.csv"
GENRE_LIST = ['무협/사극', '액션', '개그', '옴니버스', '스릴러', '일상', '드라마', '에피소드', '로맨스', '스토리', '감성', '판타지', '스포츠']

ordinal_encoder = OrdinalEncoder()
onehot_encoder = OneHotEncoder()

class WebtoonColumnIdx(Enum):
    ID = 0
    RANK = 1
    TITLE = 2
    GENRE  = 3
    AGE = 4
    BRIFE = 5


def getCsvReader(filename):
    f = open(filename, 'r', encoding='utf-8')
    next(f) # Header 건너뛰기
    return csv.reader(f)

#Label 데이터 가져오기
def load_label_data():
    csvReader = getCsvReader(LABEL_FILE_PATH)
    data = np.array([int(row[1]) for row in csvReader]).reshape(-1,1)
    return data

#Webtoon 소개글 가져오기
def load_brief_data():
    csvReader = getCsvReader(WEBTOON_FILE_PATH)
    data = np.array([row[WebtoonColumnIdx.BRIFE.value] for row in csvReader]).reshape(-1,1) #소개글 칼럼만 뽑기
    return data

def load_genre_data():
    raw_data = pd.read_csv(WEBTOON_FILE_PATH)

    #장르 OneHot Encoding
    raw_data_genre = raw_data["장르"].to_numpy()

    _genres = []
    for genre in raw_data_genre:
        genre = genre.replace(' ','').split(',')
        _genres.append(genre)

    _genres = np.array(_genres).reshape(-1,2)

    genre_data_encoded = ordinal_encoder.fit_transform(_genres[:, 0].reshape(-1,1))
    genre1_data_onehot = onehot_encoder.fit_transform(genre_data_encoded)

    genre_data_encoded = ordinal_encoder.fit_transform(_genres[:, 1].reshape(-1, 1))
    genre2_data_onehot = onehot_encoder.fit_transform(genre_data_encoded)
    data = np.c_[genre1_data_onehot.toarray(), genre2_data_onehot.toarray()]
    data.reshape(-1,13)
    return data



def loadDataset():
    labels = load_label_data()
    webtoons = load_brief_data()
    dataset = tf.data.Dataset.from_tensor_slices((webtoons, labels)).batch(32)
    return dataset


def get_dataset_partitions_tf(ds, ds_size, train_split=0.8, val_split=0.2, shuffle=True, shuffle_size=10000):
    if shuffle:
        ds = ds.shuffle(shuffle_size, seed=12)

    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)

    train_ds = ds.take(train_size)
    val_ds = ds.skip(train_size).take(val_size)

    return train_ds, val_ds

def save_webtoon_images():
    raw_data = pd.read_csv(WEBTOON_FILE_PATH)

    image_url_data = raw_data[["ID", "이미지 위치"]]
    image_dir_path = "/Users/daehungo/Desktop/ToonBot/datasets/images/"

    # 다운받을 이미지 url
    for idx,row in image_url_data.iterrows():
        urllib.request.urlretrieve(row["이미지 위치"], image_dir_path + str(row["ID"]) + ".jpg")

def save_resize_image_data():
    raw_data = pd.read_csv(WEBTOON_FILE_PATH)
    image_dir_path = "/Users/daehungo/Desktop/ToonBot/datasets/images/"
    image_dir_path2 = "/Users/daehungo/Desktop/ToonBot/datasets/resizes/"

    raw_data_id = raw_data["ID"]

    for idx, id in raw_data_id.items():
        path = image_dir_path + str(id) + ".jpg"
        image_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)  # RGB로 변환
        image_resize = cv2.resize(image_rgb, (64, 64))
        cv2.imwrite(image_dir_path2 + str(id) + ".jpg", image_resize)

def load_image_data():
    raw_data = pd.read_csv(WEBTOON_FILE_PATH)
    image_dir_path = "/Users/daehungo/Desktop/ToonBot/datasets/resizes/"

    raw_data_id = raw_data["ID"]
    data = []

    for idx, id in raw_data_id.items():
        path = image_dir_path + str(id) + ".jpg"
        image_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)  # RGB로 변환
        # image_flatten = image_rgb.flatten()
        data.append(image_rgb)

    return np.array(data)
# save_webtoon_images()