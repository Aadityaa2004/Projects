class Book:
    def __init__(self,title=None,author=None,price=None,ref=None):
        self.title=title
        self.author=author
        self.price=price
        self.ref=ref

    def __str__(self):
        info = "Title: "+self.title+"; Author: "+self.author
        info =info+"; (Ref: "+self.ref+"; Price: $"+str(self.price)+")"
        return info


def main():
    book1=Book("A Brief History of Time","Stephen Hawking",10.17,"GV5N32M9")
    print(book1)


if __name__=="__main__":
    main()
