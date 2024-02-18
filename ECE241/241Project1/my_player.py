from song import Song
from binary_search_tree import BinarySearchTree
from stack import Stack


class MyPlayer:
    def __init__(self):

        '''
        The __init__ function initialises MyPlayer with attributes for managing a song library and play history:
        songList, is_sorted, yearMemory, playHistory. 
        '''

        self.songList = [] # List to store songs in the library
        self.is_sorted = False # Indicator for sorted library
        self.yearMemory = {}  # Dictionary to store songs organized by year
        self.playHistory = Stack()  # Stack to store the play history
        
    def loadLibrary(self, filename):

        '''
        The loadLibrary function loads a music library from a file and populate the song list.
        Under this function, we open up the file required, extract data, and convert it to a 
        Song object format. 
        '''

        file = open(filename, 'r') # opens the file only under the option of "read only"
        for line in file:
            song_data = line.strip().split('|') # extract all the data from the file
            if song_data:
                self.artist_name, self.song_title, self.song_id, self.duration, self.year = song_data # add it to list 
                self.duration = float(self.duration) 
                self.year = int(self.year)

                song_data = Song(self.artist_name, self.song_title, self.song_id, self.duration, self.year) # Create a Song object 
                self.songList.append(song_data) # Add the created object to the songList
            
    def quickSort(self):

        '''
        The quickSort function sorts the song library using the quicksort algorithm. Here, I created two more functions
        namely: quick_sort and partition which performs the quicksort. The I write an algorithm the runs through the 
        songList to match the year with its respective title. This method sorts the whole songList according to year
        in ascending order.
        '''

        def quick_sort(arr, left, right): # Implementing quick_sort algorithm
            if left < right:
                pivot_index = partition(arr, left, right)
                quick_sort(arr, left, pivot_index - 1) # quick sorting the left side of the unsortered list from the new pivot
                quick_sort(arr, pivot_index + 1, right) # quick sorting the right side of the unsortered list from the new pivot

        def partition(arr, left, right): # Implementing partition algorithm
            pivot = arr[right] # setting the pivot point 
            i = left - 1
            for j in range(left, right):
                if arr[j] < pivot:
                    i += 1
                    arr[i], arr[j] = arr[j], arr[i] 
            arr[i + 1], arr[right] = arr[right], arr[i + 1]
            return i + 1

        year_list = [] # creating a new list 
        for line in self.songList:
            year_list.append(line.year) # adding all the years to this array

        quick_sort(year_list, 0, len(year_list) - 1) # performing quicksort to the new year_list created 

        sorted_song_list = [] # creating a new list for the sorted songs

        for year in year_list:
            for song in self.songList:
                if song.year == year:
                    sorted_song_list.append(song) # adds the song to the sorted list in the increasing order of year

        complete = [] # creating a new list to give an output of the completed songList
        repeated = set()  # Create a set to keep track of repeated songs

        for i in sorted_song_list: 
            if i not in repeated: # this is done to cancel out the repitions in the sorted list
                complete.append(i)
                repeated.add(i)

        self.songList = complete # setting the complete list equal to self.songList to return the new sorted list.
        self.is_sorted = True # assigning and confirming that the list has been sorted

    def playSong(self, title):

        '''
        The playSong function plays a song with the given title and adds it to the play history. Here I first implemented 
        a search algorithm to find the song played and then used the stack algorithm which waas given to push the song played
        into it. So that last-in will be first-out, thus giving us the history and also the last played song. 
        '''

        found = False
        while found is not True: # performing a search algorithm
            for search in self.songList:
                if title in search.song_title:
                    Song.play(search) # retruns the song that is currently played 
                    self.playHistory.push(search) # inserts the song played to a stack (history)
                    found = True
        
        if found == False:
            pass
        
    def getLastPlayed(self):

        '''
        The getlastPlayed function retrieves the last played song from the play history. It retruns the last-in value and 
        also returns None as the output if the stack is empty. 
        '''

        if not self.playHistory.isEmpty(): # checks if the history is empty 
            return self.playHistory.peek() # if the history not empty, then it peeks into the stack by returning the most recent output
        else:
            return None # if stack is empty, returns None
        
    def hashfunction(self, song):

        '''
        The hashfunction computes a hash value for a song based on its release year.
        '''

        return song.year # returns the song year from the hashtable
  
    def buildYearMemory(self):

        '''
        The buildYearmemory function builds a data structure to organize songs by year and title. This function is implemented 
        by first calling the hashfuction for the year and then inserting it into the binary search tree. 
        '''

        for song in self.songList:
            year = self.hashfunction(song)  # Getting the year of the song
            if year not in self.yearMemory:
                self.yearMemory[year] = BinarySearchTree() # creating a Binary Search Tree  
                self.yearMemory[year].put(song.song_title, song) # inserting the title and object into the tree according to the year
            else:
                self.yearMemory[year].put(song.song_title, song) # inserting the title and object into the tree according to the year
            
    def getYearMemory(self, year, title):

        '''
        The getYearMemory function retrieves a song from the year-based memory structure by year and title. This is done by first 
        creating a dictionary and then using the key:value pair of the dictionary to retrive the songyear from the binary search 
        tree. We also change the BinarySearchTree so that we can accomodate the step counter to it. 
        '''

        steps = 0  # counter for steps
        the_song = None # the song to be retrieved but set to None initially

        if year not in self.yearMemory: # checks if the year is not in the yearmemory
            return {"steps": None, "song": None} # returns None 
        else:
            self.yearMemory[year].get(title) # calls get function from BinarySearchtree to search for the title of song
            if self.yearMemory[year].get(title) is not None:
                steps = self.yearMemory[year].get(title)['steps'] # to retrieve the number of steps
                the_song = self.yearMemory[year].get(title)['song'] # to retrieve the song
                return {"steps": steps, "song": the_song} # returns the song and steps
            else:
                return {"steps": None, "song": None} # returns the song and steps

    def getSong(self, year, title):

        '''
        The getSong function retrieves a song directly from the song list using its year and title. This is done by 
        performing a simple linear search to acess the song and the number of steps it took to complete the search.
        '''

        steps = 0  # counter for steps
        the_song = None  # The song that is going to be retrived but here set to None initially

        for song in self.songList: # Implememting Liner Search: 
            if song.year == year and song.song_title == title:
                the_song = song # retrieved song 
                steps+=1
                break
            else:
                steps+=1
            
        return {"steps": steps, "song": the_song} # returns the song and steps

