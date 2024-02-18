class Song:
    def __init__(self, artist_name: str, song_title: str, song_id: str, duration: float, year: int):

        '''
        The __init__ function initialises a song object with its attributes :
        artist_name, song_title, song_id, duration and year. 
        '''

        self.artist_name = artist_name # Store the artist's name
        self.song_title = song_title # Store the title of the song
        self.song_id = song_id # Store the unique song id
        self.duration = duration # Store the duration of the song in seconds
        self.year = year # Store the year when the song was released
        

    def __str__(self):

        '''
        The __str__ function returns a formatted string representing the song.
        '''
        
        return f"{self.song_title} by {self.artist_name} (ID: {self.song_id}) released in {self.year}" # the return statement of __str__ function

    def play(self):

        '''
        The play function prints an output to idicate the song that is being played along with the duration of the song 
        '''

        print ("%s is playing, with a duration of %s second(s)" %(self.song_title,self.duration)) # the print statement for the song that is playing
