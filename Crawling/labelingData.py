from static import DATA_PATH
import csv

# 파일 경로 위치
readFile = DATA_PATH + "/webtoons.csv"
writeFile = DATA_PATH + "/labels.csv"

# csv 파일 변수
rf = open(readFile, 'r', encoding='utf-8')
next(rf)
csvReader = csv.reader(rf)
wf = open(writeFile, 'w', encoding='utf-8', newline='')
csvWriter = csv.writer(wf)

columns = ["ID", "label"]
rankIdx = 1 #랭크 인덱스 위치
dayIdx = -1 #요일 인덱스 위치
topCriteria = 30 #상위 웹툰 기준
label = -1 #라벨링 변수

# 웹툰 순위에 따른 라벨링( 1 : 흥행 성공 웹툰, 0 : 흥행 실패 웹툰 )
csvWriter.writerow(columns)
for data in csvReader:
    if(int(data[rankIdx]) <= topCriteria): label = 1
    else: label = 0
    csvWriter.writerow([data[0], label])

rf.close()
wf.close()
