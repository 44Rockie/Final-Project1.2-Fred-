# Final-Project1.2-Fred-
Final Project 'Fred'
developed by Grady Morris

# 3.1.21 This is my program for the final project. The program went through several different iterations and
# took about 12 hours to complete. There is a total of 14 installed and imported libraries that are being
# used by the program to perform the different functions that the program performs. I have nicknamed the
# program 'Fred.' Fred uses speech recognition and pyttsx3 to greet the user and also to receive the users
# inputs.  There is a friendly and simple user interface that prompts the user with four options. Option one
# is a function that scrapes twitter for tweets on a subject that user is prompted to speak to Fred as the
# input.  Part of the twitter sentiment code is from the previous project related to twitter sentiment analysis
# but I have also added an additional feature using the matplotlib library to return the results of the analysis in
# pie chart. The second option uses facial recognition to take six pictures of the user in order to train Fred
# to recognize the user. Option three tests Fred to determine if he recognizes the user. Option Four allows
# the user to say to Fred a subject for a wiki search and returns the results from the search to user as
# printed results and spoken to the user. All of the functions return the user to the main menu. The last
# option is the quit option which allows the user to exit the program gracefully. I believe I could use this
# program in the future and further expand upon it adding additional features and libraries to turn into
# a very robust and proficient virtual assistant.

The program was developed using python 3.7 and pycharm.
The following imports are necessary:
import pandas as pd
import datetime as dt
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import snscrape.modules.twitter as sntwitter
import nltk
import matplotlib.pyplot as plt
import pickle
import cv2
import face_recognition
import numpy as np
import speech_recognition as sr
import pyttsx3
import wikipedia
