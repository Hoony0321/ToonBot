from static import DATA_PATH
import csv

# 파일 경로 위치
readFile = DATA_PATH + "/webtoons.csv"
writeFile = DATA_PATH + "/labels.csv"

# csv 파일 변수
rf = open(readFile, 'r', encoding='utf-8')
csvReader = csv.reader(rf)
wf = open(writeFile, 'w', encoding='utf-8', newline='')
csvWriter = csv.writer(wf)

# webtoon 정보 불러오기

rankIdx = 1 #랭크 인덱스 위치
dayIdx = -1 #요일 인덱스 위치
topCriteria = 35 #상위 웹툰 기준
label = -1 #라벨링 변수

for data in csvReader:
    if(data[rankIdx] <= topCriteria): label = 1
    else: label = 0

    wf.write






rf.close()
wf.close()
