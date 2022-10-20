import requests
import re
from Webtoon import Webtoon
from bs4 import BeautifulSoup


# 특정 URL의 HTML 데이터 받아오기
def getHtmlSource(url):
    request = requests.get(url)

    if (request.status_code != 200):
        raise Exception("성공적인 응답을 얻지 못했습니다.")

    htmlSource = request.content
    return htmlSource

def getTitleIdFromHtml(html):
    href = html.get("href")
    reCompile = re.compile("titleId=\d+")
    titleId = reCompile.search(href).group(0)[8:]

    return titleId



def crawlingWebtoon():
    # STATIC 변수 선언
    URL = "https://comic.naver.com/webtoon/weekday?order=ViewCount"
    Detail_URL = "https://comic.naver.com/webtoon/list?titleId="
    weekly = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
    webtoonList = [[] for _ in range(len(weekly))]

    soup = BeautifulSoup(getHtmlSource(URL), 'html.parser')

    for idx, day in enumerate(weekly):
        htmlSources = soup.select("#content > div.list_area.daily_all > .col > .col_inner > {} + ul > li > a".format("." + day)) #웹툰 리스트 정보
        for rank, html in enumerate(htmlSources):
            titleId = getTitleIdFromHtml(html) #웹툰ID 가져오기
            detailSoup = BeautifulSoup(getHtmlSource(Detail_URL + titleId), 'html.parser')  #웹툰 디테일 페이지 정보 가져오기
            title = detailSoup.select_one("#content > div.comicinfo > div.detail > h2 > span.title").text.strip() #웹툰 제목
            authors = detailSoup.select_one("#content > div.comicinfo > div.detail > h2 > span.wrt_nm").text.strip() #웹툰 작가
            brief = detailSoup.select_one("#content > div.comicinfo > div.detail > p:nth-child(2)").text.strip() #웹툰 설명
            genre = detailSoup.select_one("#content > div.comicinfo > div.detail > p.detail_info > span.genre").text.strip() #웹툰 장르
            age = detailSoup.select_one("#content > div.comicinfo > div.detail > p.detail_info > span.age").text[:-5] #웹툰 이용 가능 나이
            if(age==""): age = 0 #전체이용가

            img = detailSoup.select_one("#content > div.comicinfo > div.thumb > a > img")["src"]
            item = Webtoon(titleId, title, genre, brief, age, authors, img, rank+1) # 웹툰 객체 생성
            webtoonList[idx].append(item) # 웹툰 객체 리스트에 추가

    return webtoonList
