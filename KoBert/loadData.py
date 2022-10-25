from static import DATA_PATH
import csv
import numpy as np

LABEL_FILE_PATH = DATA_PATH + "/labels.csv"
WEBTOON_FILE_PATH = DATA_PATH + "/webtoons.csv"
BRIEF_IDX = 5

def getCsvReader(filename):
    f = open(filename, 'r', encoding='utf-8')
    next(f) # Header 건너뛰기
    return csv.reader(f)


# Label Data Set 구성
def getLabelDataSet():
    csvReader = getCsvReader(LABEL_FILE_PATH)
    data = np.array([int(row[1]) for row in csvReader]).reshape(-1,1)
    return data

# Train Data Set 구성
def getTrainDataSet():
    csvReader = getCsvReader(WEBTOON_FILE_PATH)
    data = np.array([row[BRIEF_IDX] for row in csvReader]).reshape(-1,1)
    return data
