""" This solution module works fine but does not have all the proper comments (docstring are missing as well)
This an example of what you should not do regarding comments.
"""


from Book import Book


class Inventory:

    def __init__(self):
        self.items=[]
        
    def initialize(self):
        self.items.append(Book("A Brief History of Time","S. Hawking",10.17,"GV5N32M9"))
        self.items.append(Book("The Alchemist","P. Coelho",6.99,"TR3FL0EW"))
        self.items.append(Book("Thus Spoke Zarathustra","F. Nietzsche",7.81,"F2O9PIE9"))
        self.items.append(Book("Jonathan Livingston Seagull","R. Bach",6.97,"R399CED1"))
        self.items.append(Book("The Time Machine","H. G. Wells",5.95,"6Y9OPL87"))
        self.items.append(Book("Introduction to Programming in Python","R. Sedgewick",69.99,"5T3RRO90"))
        self.items.append(Book("Atoms of Silence","H. Reeves",28.02,"3W2TB162"))
        self.items.append(Book("2001: A Space Odyssey","A. C. Clarke",8.99,"TU2RL012"))
        self.items.append(Book("20,000 Leagues under the Sea","J. Verne",5.99,"JI2PL986"))
        self.items.append(Book("Les Miserables","V. Hugo",9.98,"VC5CE249"))

        
    def __str__(self):
        dis=""
        for i in range(len(self.items)):
            dis=dis+str(i+1)+"-"+str(self.items[i])+"\n"
        return dis

        
    def size(self):
        return len(self.items)
        
    def display(self):  # obsolete
        for i in range(len(self.items)):
            print(i+1,"-",self.items[i])

    def info(self):
        maxprice=0.0
        totprice=0.0
        for item in self.items:
            totprice=totprice+item.price
            if item.price>maxprice:
                maxprice=item.price
                
        print("#Items: "+str(self.size()))      
        print("Total price $"+str(totprice))
        print("Most expensive item at $"+str(maxprice))
         

    def search(self,keyword):
        found=False
        for i in range(len(self.items)):
            if keyword in self.items[i].title:
                print(i+1,"-",self.items[i])
                found=True
        if not found:
            print("No book found!")

          
    def add(self,mybook=None):
        if mybook is not None:
            self.items.append(mybook)
        else:
            new_item=Book()
            new_item.title=input("Enter Title: ")
            new_item.author=input("Enter Author Name: ")
            new_item.price=float(input("Enter Price: "))        
            new_item.ref=input("Enter Reference: ")
            self.items.append(new_item)

    def adjust_price(self,inc):
        for item in self.items:
            item.price=item.price*(1+inc)
 
