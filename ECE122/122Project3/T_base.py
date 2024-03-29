# Hrishit Srivastava
# Aadityaa rengaraj Sethuraman
from tkinter import *
import numpy as np
from p import Pixel
import time, random
from g import Grid




class Tetrominoes:
    ## to complete  
    def __init__(self, canv, nrow, ncol, scale, nbpatterns=2, pattern=None):
        '''Under the __init__ fuction we define 5 main attributes which are nrow, ncol, scale, nbpatterns and pattern.
        we also create an empty array where we register a certain set of values to implement the shape pattern and colour. '''
        self.canv = canv
        self.nrow = nrow
        self.ncol = ncol
        self.scale = scale
        self.nbpattern = nbpatterns
        self.currentindex=0
        self.love= pattern
        self.pixel_list= []#list to store the bigpixel attribute so that it can be used in delete function
        
       
        if pattern is not None :#check if pattern is getting passed or not
        
          self.patterns=np.array(pattern[0])
          self.name="Custom"
          self.nbpattern=len(self.patterns)
          
          self.h=self.patterns.shape[0]
          self.w=self.patterns.shape[1]
          
          
        else:
            self.patterns = np.array([[2, 2, 2],
                                      [2, 0, 2],
                                      [2, 2, 2]]) #storing the initial pattern
            self.name = "Basic"
            
            
            self.h=self.patterns.shape[0]
            self.w=self.patterns.shape[1]
       

    def activate(self, i=1, j=None):
        ''''takes the pattern, i and j coo-ordinates. If j is none then we assign a random j value 
        or else it passes a new i and j co-ordinates which is nothing but the selfrow and selfcol to pixrel for 
        printing it in the desired format.'''
        self.i=i
        if j is None:
            ncol = 30  # Assuming a grid with 30 columns
            j = np.random.randint(0, ncol - self.w + 1)#randomly giving a value to j
        
        for liz in range (self.patterns.shape[0]):
           for row in range(self.h): 
                for col in range(self.w):
                    # Calculate the pixel coordinates based on scale and indices
                    x1 = j + col
                    y1 = i + row
                    if self.patterns.ndim==2:#check dimensions of array

                     element= self.patterns[row,col]
                     if element !=  0:
                      big_pixel= Pixel(self.canv, y1, x1, self.nrow, self.ncol, self.scale, element)#printing the pattern by sending it to pixel class
                      self.pixel_list.append(big_pixel)
                    else:
 
                     element= self.patterns[liz][row,col]
                     if element !=  0:
                        big_pixel= Pixel(self.canv, y1, x1, self.nrow, self.ncol, self.scale, element )
                        self.pixel_list.append(big_pixel)
        self.j=j
                     
    def get_pattern(self):
        '''This function is used to get the pattern every time the code runs. It access the array and returns the shape
        according to the number inside the self.currentindex'''
        return self.patterns[self.currentindex] 
    
    def rotate(self):
     '''This function is used to rotate the shape in all four different sides and this is used under test-function 3.
     This function is later called to tetris to enable the player to turn the pieces and allign them correctly before the shape falls down.'''
     self.delete()
     if self.love is not None:
      self.currentindex = (self.currentindex + 1) % self.nbpattern#update the current index
      if self.currentindex >= len(self.love):
        self.currentindex = 0
      self.patterns = np.array(self.love[self.currentindex])
      self.h = self.patterns.shape[0]
      self.w = self.patterns.shape[1]
      self.activate(self.i, self.j)#send the coordinates to activate function for printing process
     else:
       self.patterns = np.array([[2, 2, 2],
                                 [2, 0, 2],
                                 [2, 2, 2]])
        
       self.h=self.patterns.shape[0]
       self.w=self.patterns.shape[1]
       self.activate(self.i, self.j)
     

    def delete(self):
        '''this function is used to delete all the patterns '''
        for p in  self.pixel_list:
         p.delete()


    def left(self):
        '''This function ensures that the pattern moves according to the key
        input given by the user. Here the object moves left.'''
        self.delete()
        self.j -=1
        self.activate(self.i,self.j)

    def right(self):
        '''This function ensures that the pattern moves according to the key
        input given by the user. Here the object moves right.'''
        self.delete()
        self.j +=1
        self.activate(self.i,self.j)

    def down(self):
        '''This function ensures that the pattern moves according to the key
        input given by the user. Here the object moves down.'''
        self.delete()
        self.i+=1
        self.activate(self.i,self.j)

    def up(self):
        '''This function ensures that the pattern moves according to the key
        input given by the user. Here the object moves up.'''
        self.delete()
        self.i-=1
        self.activate(self.i,self.j)

    @staticmethod
    def random_select(canv,nrow,ncol,scale):
        t1=TShape(canv,nrow,ncol,scale)
        t2=TripodA(canv,nrow,ncol,scale)
        t3=TripodB(canv,nrow,ncol,scale)
        t4=SnakeA(canv,nrow,ncol,scale)
        t5=SnakeB(canv,nrow,ncol,scale)
        t6=Cube(canv,nrow,ncol,scale)
        t7=Pencil(canv,nrow,ncol,scale)        
        return random.choice([t1,t2,t3,t4,t5,t6,t7,t7]) #a bit more change to obtain a pencil shape


#########################################################
############# All Child Classes #########################
#########################################################
class TShape(Tetrominoes):
    '''We are initiating this class to register Tshape in an array format.'''
    def __init__(self, canv, nrow, ncol, scale):
        '''We define the __init__ function and enter manually the value into 
        the array'''
        pattern = [np.array([[0, 3, 0], [0, 3, 0], [3, 3, 3]]),
                  np.array([[3, 0, 0], [3, 3, 3], [3, 0, 0]]),
                  np.array([[3, 3, 3], [0, 3, 0], [0, 3, 0]]),
                  np.array([[0, 0, 3], [3, 3, 3], [0, 0, 3]]),
        ]
        
        super().__init__(canv, nrow, ncol, scale, 5, pattern)
        self.name="TShape"

class TripodA(Tetrominoes):
    '''We are initiating this class to register TripodA in an array format.'''    
    def __init__(self, canv, nrow, ncol, scale):
        '''We define the __init__ function and enter manually the value into 
        the array'''
        pattern = [ np.array([[0, 4, 0], [0, 4, 0], [4, 0, 4]]),
                   np.array([[4, 0, 0], [0, 4, 4], [4, 0, 0]]),
            np.array([[4, 0, 4], [0, 4, 0], [0, 4, 0]]),
            np.array([[0, 0, 4], [4, 4, 0], [0, 0, 4]])]
        
        
        super().__init__(canv, nrow, ncol, scale, 5, pattern)
        self.name= "TripodA"


class TripodB(Tetrominoes):
    '''We are initiating this class to register TripodB in an array format.'''    
    def __init__(self, canv, nrow, ncol, scale):
        '''We define the __init__ function and enter manually the value into 
        the array'''
        pattern =  [np.array([[0, 5, 0], [5, 0, 5], [5, 0, 5]]),
                   np.array([[5, 5, 0], [0, 0, 5], [5, 5, 0]]),
            np.array([[5, 0, 5], [5, 0, 5], [0, 5, 0]]),
            np.array([[0, 5, 5], [5, 0, 0], [0, 5, 5]]),
            
        ]
        
        super().__init__(canv, nrow, ncol, scale, 5, pattern)
        self.name= "TripodB"

class SnakeA(Tetrominoes):
    '''We are initiating this class to register SnakeA in an array format.'''
    def __init__(self, canv, nrow, ncol, scale):
        '''We define the __init__ function and enter manually the value into 
        the array'''
        pattern = [np.array([[6, 6, 0], [0, 6, 0], [0, 6, 6]]),
        
            np.array([[0, 0, 6], [6, 6, 6], [6, 0, 0]]),
            np.array([[6, 6, 0], [0, 6, 0], [0, 6, 6]]),
        
            np.array([[0, 0, 6], [6, 6, 6], [6, 0, 0]])
        ]
        
        super().__init__(canv, nrow, ncol, scale, 5, pattern)
        self.name= "SnakeA"

class SnakeB(Tetrominoes):
    '''We are initiating this class to register SnakeB in an array format.'''
    def __init__(self, canv, nrow, ncol, scale):
        '''We define the __init__ function and enter manually the value into 
        the array'''
        pattern =  [np.array([[0, 7, 7], [0, 7, 0], [7, 7, 0]]),
                    
            np.array([[7, 0, 0], [7, 7, 7], [0, 0, 7]]),
            np.array([[0, 7, 7], [0, 7, 0], [7, 7, 0]]),
                    
            np.array([[7, 0, 0], [7, 7, 7], [0, 0, 7]])
        ]
            
      
        super().__init__(canv, nrow, ncol, scale, 5, pattern)
        self.name= "SnakeB"

class Cube(Tetrominoes):
    '''We are initiating this class to register Cube in an array format.'''
    def __init__(self, canv, nrow, ncol, scale):
        '''We define the __init__ function and enter manually the value into 
        the array'''
        pattern =  [np.array([[8, 8, 8], [8, 8, 8], [8, 8, 8]]),
        np.array([[0, 8, 0], [8, 8, 8], [0, 8, 0]]),
        np.array([[8, 0, 8], [0, 8, 0], [8, 0, 8]]),
        np.array([[8, 8, 8], [8, 8, 8], [8, 8, 8]])]
    
        
        super().__init__(canv, nrow, ncol, scale, 5, pattern)
        self.name= "Cube"

class Pencil(Tetrominoes):
    '''We are initiating this class to register Pencil in an array format.'''
    def __init__(self, canv, nrow, ncol, scale):
        '''We define the __init__ function and enter manually the value into 
        the array'''
        pattern =  [np.array([[0, 9, 0], [0, 9, 0], [0, 9, 0]]),
        np.array([[0, 0, 0], [9, 9, 9], [0, 0, 0]]),
         np.array([[0, 9, 0], [0, 9, 0], [0, 9, 0]]),
        np.array([[0, 0, 0], [9, 9, 9], [0, 0, 0]]) ]
            
       
        super().__init__(canv, nrow, ncol, scale, 5, pattern)
        self.name= "Pencil"





#########################################################
############# Testing Functions #########################
#########################################################
def delete_all(canvas):
    canvas.delete("all")
    print("Delete All")


def test1(canvas,nrow,ncol,scale):
    print("Generate a Tetromino (basic shape)- different options")
    
    tetro1=Tetrominoes(canvas,nrow,ncol,scale) # instantiate
    print("Tetro1",tetro1.name)
    print("  number of patterns:",tetro1.nbpattern)
    print("  current pattern:\n",tetro1.get_pattern()) # retrieve current pattern
    print("  height/width:",tetro1.h,tetro1.w)
    tetro1.activate(nrow//2,ncol//2)        # activate and put in the middle
    print("  i/j coords:  ",tetro1.i,tetro1.j)

    pattern=np.array([[0,3,0],[3,3,3],[0,3,0],[3,0,3],[3,0,3]]) # matrix motif
    tetro2=Tetrominoes(canvas,nrow,ncol,scale,3,[pattern]) # instantiate (list of patterns-- 1 item here)
    print("\nTetro2",tetro2.name)
    print("  number of patterns:",tetro2.nbpattern)
    print("  current pattern:\n",tetro2.get_pattern()) # retrieve current pattern
    print("  height/width:",tetro2.h,tetro2.w)
    tetro2.activate()        # activate and place at random at the top
    print("  i/j coords:  ",tetro2.i,tetro2.j)

    
    
def test2(root,canvas,nrow,ncol,scale):
    print("Generate a 'square' Tetromino (with double shape) and rotate")
    
    print("My Tetro")
    pattern1=np.array([[4,0,0],[0,4,0],[0,0,4]]) # matrix motif
    pattern2=np.array([[0,0,4],[0,4,0],[4,0,0]]) # matrix motif
    tetro=Tetrominoes(canvas,nrow,ncol,scale,4,[pattern1,pattern2]) # instantiate (list of patterns-- 2 items here)
    print("  number of patterns:",tetro.nbpattern)
    print("  height/width:",tetro.h,tetro.w)
    tetro.activate(nrow//2,ncol//2)        # activate and place in the middle
    print("  i/j coords:  ",tetro.i,tetro.j)

    for k in range(10): # make 10 rotations
        tetro.rotate() # rotate (change pattern)
        print("  current pattern:\n",tetro.get_pattern()) # retrieve current pattern
        root.update()
        time.sleep(0.5)
    tetro.delete() # delete tetro (delete every pixels)


def rotate_all(tetros): #auxiliary routine
    for t in tetros:
        t.rotate()
    
       
def test3(root,canvas,nrow,ncol,scale):
    print("Dancing Tetrominoes")

    t0=Tetrominoes(canvas,nrow,ncol,scale)
    t1=TShape(canvas,nrow,ncol,scale)
    t2=TripodA(canvas,nrow,ncol,scale)
    t3=TripodB(canvas,nrow,ncol,scale)
    t4=SnakeA(canvas,nrow,ncol,scale)
    t5=SnakeB(canvas,nrow,ncol,scale)
    t6=Cube(canvas,nrow,ncol,scale)
    t7=Pencil(canvas,nrow,ncol,scale)
    tetros=[t0,t1,t2,t3,t4,t5,t6,t7]

    for t in tetros:
        print(t.name)

    # place the tetrominos
    for i in range(4):
        for j in range(2):
            k=i*2+j
            tetros[k].activate(5+i*10,8+j*10)
            
    ####### Tkinter binding for this test
    root.bind("<space>",lambda e:rotate_all(tetros))     

    
      
def test4(root,canvas,nrow,ncol,scale):
    print("Moving Tetromino")
    tetro=Tetrominoes.random_select(canvas,nrow,ncol,scale) # choose at random
    print(tetro.name)
        
    ####### Tkinter binding for this test
    root.bind("<space>",lambda e:tetro.rotate())
    root.bind("<Up>",lambda e:tetro.up())
    root.bind("<Down>",lambda e:tetro.down())
    root.bind("<Left>",lambda e:tetro.left())
    root.bind("<Right>",lambda e:tetro.right())

    tetro.activate()

    

#########################################################
############# Main code #################################
#########################################################

def main():
    
        ##### create a window, canvas 
        root = Tk() # instantiate a tkinter window
        nrow=45
        ncol=30
        scale=20
        canvas = Canvas(root,width=ncol*scale,height=nrow*scale,bg="black") # create a canvas width*height
        canvas.pack()

        ### general binding events to choose a testing function
        root.bind("1",lambda e:test1(canvas,nrow,ncol,scale))
        root.bind("2",lambda e:test2(root,canvas,nrow,ncol,scale))
        root.bind("3",lambda e:test3(root,canvas,nrow,ncol,scale))
        root.bind("4",lambda e:test4(root,canvas,nrow,ncol,scale))
        root.bind("<d>",lambda e:delete_all(canvas))

        
        root.mainloop() # wait until the window is closed        

if __name__=="__main__":
    main()

