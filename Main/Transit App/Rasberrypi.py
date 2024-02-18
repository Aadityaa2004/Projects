import RPi.GPIO as GPIO
import time

# GPIO 7 SEG PIN SETUP
GPIO.setmode(GPIO.BOARD)
GPIO.cleanup()

# SEGMENT Definition
A = 3
B = 5
C = 7
D = 8
E = 10
F = 11
G = 12


# DIGIT Definition
D1 = 13
D2 = 15
D3 = 16
D4 = 18
D5 = 19
D6 = 21

# Setup
SEGMENTS = [A, B, C, D, E, F, G]
DIGITS = [D1, D2, D3, D4, D5, D6]

for segment in SEGMENTS:
    GPIO.setup(segment, GPIO.OUT)
    GPIO.output(segment, 0)

for digit in DIGITS:
    GPIO.setup(digit, GPIO.OUT)
    GPIO.output(digit, 1)
   
   

NUMBERS = {
    0: (A, B, C, D, E, F),
    1: (B, C),
    2: (A, B, D, E, G),
    3: (A, B, C, D, G),
    4: (B, C, F, G),
    5: (A, C, D, F, G),
    6: (A, C, D, E, F, G),
    7: (A, B, C),
    8: (A, B, C, D, E, F, G),
    9: (A, B, C, D, F, G),
    'S': (A, C, D, F, G),
    'N': (A, B, C, E, F)
}


# 7 SEGMENT DISPLAY
def show(number): # Number can be 999999 to 0
    for digit in range(6):
        for segment in SEGMENTS:
            GPIO.output(segment, 0)
        for seg in NUMBERS[number[digit]]:
            GPIO.output(seg, 1)
        GPIO.output(DIGITS[digit], 0)
        time.sleep(0.001)
        GPIO.output(DIGITS[digit], 1)
    return

# Example
while True:
    NUMBER = [8,8,'N','S',8,8]
    show(NUMBER)
  #  time.sleep(0.00001)