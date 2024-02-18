# Modules
from Inventory import Inventory

# Functions
def display_menu():
    print("-------------------------------------------------------")
    print("1-List Inventory; 2-Info Inventory; 3-Search Inventory;")
    print("4-Add Item; 5-Remove Item; 6-Inflation; 7-Shop; 8-Check-out")


'''------------------------------------------------------
                Main program starts here 
---------------------------------------------------------'''


print("Welcome to BestMedia")
print("====================")

# initialize the inventory
store=Inventory()
store.initialize()

# initialize the shopping cart
cart=Inventory()


# closed loop, keep executing until user press "Enter"
while True:
    display_menu() #menu that contains the options
    command=input("\nEnter Command: ")

    if command=="": #exit program
        print("Goodbye!")
        break    

    elif command=="1": #display all the list items
        #store.display()
        print(store)

    elif command=="2": #display information about the list items
        store.info()

    elif command=="3": #search inventory
        keyword=input("Enter a title keyword: ")
        store.search(keyword)      

    elif command=="4": #add a particular item to the inventory
        store.add()

    elif command=="5": #delete a particular item
        id=input("which item do you want to delete: ")
        del(store.items[int(id)-1])

    elif command=="6": #accounting for inflation
        num=input("Enter Inflation %: ")
        store.adjust_price(float(num)/100)
        
    elif command=="7": #shopping cart
        id=input("which item do you want to buy? ")
        print("\""+store.items[int(id)-1].title+"\" added to shopping cart!")
        cart.add(store.items[int(id)-1])

    elif command=="8": #check-out
        if cart.size()==0:
            print("Your cart is Empty!")
            continue
        print("Current shopping cart: ")
        #cart.display()
        print(cart)
        cart.info()
        promo=input("Enter your promotion code if any: ")
        update=False
        if promo=="Voyage":
            cart.adjust_price(-0.05)
            update=True
        elif promo=="Parfait":
            cart.adjust_price(-0.10)
            update=True
        if update:
            print("Updated shopping cart:")
            #cart.display()
            print(cart)
            cart.info()

        card=input("Enter you credit card number: ")
        print("Purchase done!...Enjoy you new books")
        cart=Inventory()
