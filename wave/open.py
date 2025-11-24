import RPi.GPIO as GPIO
import time 

GPIO.setmode(GPIO.BCM)

def is_open(pin):
    return GPIO.input(pin)