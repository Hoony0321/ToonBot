class Webtoon():

    def __init__(self, titleId, title, genre, brief, age, authors, img, rank):
        self.titleId = titleId
        self.title = title
        self.genre = genre
        self.brief = brief
        self.age = age
        self.authors = authors
        self.img = img
        self.rank = rank

    def __str__(self):
        str = "titleID : {}\n" \
              "title : {}\n" \
              "genre : {}\n" \
              "brief : {}\n" \
              "age : {}\n" \
              "authors : {}\n" \
              "img : {}\n" \
              "rank : {}"

        return str.format(self.titleId, self.title, self.genre, self.brief, self.age, self.authors, self.img, 목self.rank)