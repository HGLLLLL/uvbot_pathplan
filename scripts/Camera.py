import numpy as np
import threading
import cv2
import datetime
import os
from time import sleep
from copy import deepcopy

class Monitor(threading.Thread):
    """
    Class runs the camera in another thread.
    The main thread will take a frame using get_frame() when needed.
    """
    def __init__(self, camera_id):
        super(Monitor, self).__init__()
        self.camera_id = camera_id
        self.lock = threading.Lock()
        self.exit = False
        self.frame = np.zeros((0, 0, 3), dtype=np.uint8)

    def get_frame(self):
        """
        The main will call this function to get the frame
        """
        return deepcopy(self.frame)
         
    def Exit(self):
        """
        The main will call this function to terminate the camera
        """
        self.exit = True

    def run(self): 
        """
        Execute after thread.start()
        """
        cap = cv2.VideoCapture(self.camera_id)

        while not self.exit :    
            ret, frame = cap.read()
            if not ret :
                break

            self.lock.acquire()
            self.frame = frame
            self.lock.release()
        cap.release()

class photoTake():
    """
    Class to take pictures in a row
    """
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.photo_amount = 10
        self.check_folder()

        now = datetime.datetime.now()
        self.file_name = str(now.year) + "-" + str(now.month) + "-" + str(now.day) + " " + str(now.hour) + ":" + str(now.minute)
        
        # init the dictionary to store the file name and the amount 
        #self.dic = {'0' :0}
        self.dic = {self.file_name: 0}

    def check_folder(self):
        """        
        Check if the image folder exist.
        If not, create one.
        """
        cwd = os.getcwd() # current working directory
        self.path = cwd + '/image/'
        isExist = os.path.exists(self.path) 
        if not isExist:
            os.mkdir(self.path)

    def get_series_name(self):
        self.file_name = input('Name of the series: ')
        if self.file_name not in self.dic:
            self.dic[self.file_name] = 0

    def __call__(self):
        """
        Determine the file name first. 
        Then press a bottom to take several photos at a time and then save it to a certain folder.
        Press r to input the name of the image.
        Press t to take photos
        """
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            cv2.imshow('frame', frame)

            if not ret or cv2.waitKey(1) & 0xFF == ord('q'):
                break

            if cv2.waitKey(1) & 0xFF == ord('r'):
                self.get_series_name()
            elif cv2.waitKey(1) & 0xFF == ord('t'):
                for i in range(self.photo_amount):
                    sleep(0.2)
                    cv2.imwrite(self.path + self.file_name + str(i + self.dic[self.file_name]),frame)
                self.dic[self.file_name] = self.dic[self.file_name] + self.photo_amount

                
class Recorder:
    """
    Class to record the video and store to the video folder
    """
    
    def __init__(self):
        """
        Setup the video format and where to store
        """
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.get_file_name()
        self.check_folder()
        self.writer = cv2.VideoWriter(self.path + self.file_name + '.mp4', fourcc, 30.0, (640,480))
    
    def get_file_name(self):
        """
        Generate the file name using the date and current time
        """
        now = datetime.datetime.now()
        self.file_name = str(now.year) + "-" + str(now.month) + "-" + str(now.day) + " " + str(now.hour) + ":" + str(now.minute) + ":" +str(now.second)
        
    def check_folder(self):
        """        
        Check if the video folder exist.
        If not, create one.
        """
        cwd = os.getcwd() # current working directory
        self.path = cwd + '/video/'
        isExist = os.path.exists(self.path) 
        if not isExist:
            os.mkdir(self.path)

    def __call__(self, frame):
        self.writer.write(frame)
