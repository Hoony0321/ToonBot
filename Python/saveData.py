from crawling import crawlingWebtoon
from Webtoon import Webtoon
import csv

#STATIC 변수 선언
fileName = "webtoons.csv"
weekly = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]

webtoonList = crawlingWebtoon() #웹툰 데이터 가져오기
columns = ["ID", "순위", "제목", "장르", "제한 나이", "소개글", "작가", "이미지 위치", "요일"]


file = open(fileName,'w', encoding='utf-8-sig', newline='')
writer = csv.writer(file)

writer.writerow(columns)

for idx in range(len(weekly)):
    for webtoon in webtoonList[idx]:
        writer.writerow([webtoon.titleId, webtoon.rank, webtoon.title, webtoon.genre,
                         webtoon.age, webtoon.brief, webtoon.authors, webtoon.img, weekly[idx]])


