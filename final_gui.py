# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 12:49:48 2020

@author: Adminz,gagan.
"""
import numpy as np
import argparse
import cv2 as cv
import subprocess
import time
import os
from yolo_utils import infer_image, show_image
import matplotlib.image as mpimg
from matplotlib import pyplot as plt
from matplotlib.pyplot import imshow
import tkinter as tk
from yolo import live_feed,vid_det_YOLO,vid_det_tinyYOLO

global person, bike, car, building, searched
LARGE_FONT= ("Verdana", 12)
val=0

class ObjDet(tk.Tk):
    def __init__(self, *args, **kwargs):
        
        tk.Tk.__init__(self, *args, **kwargs)
        container = tk.Frame(self)
        self.title("Object Detection")
        self.configure(bg='black')
        container.pack()

        self.frames = {}
        self.geometry("800x900")

        for F in (StartPage, Viddet_YOLO, livefeed, Viddet_tinyYOLO):
            frame = F(container, self)
            self.frames[F] = frame
            frame.grid(row=900, column=800)
        self.show_frame(StartPage)

    def show_frame(self, cont):

        Frame = self.frames[cont]
        Frame.tkraise()

from tkinter.filedialog import askopenfilename

class StartPage(tk.Frame): #initial page of the GUI
    def __init__(self, parent, controller):
        tk.Frame.__init__(self,parent)
        
        #<enter location of image file that would be displayed on initial page. It should be .gif format>
        self.image1 = tk.PhotoImage(file='drone5.gif')
        
        panel1 = tk.Label(self, image=self.image1)
        panel1.place(x=35, y=200)
        
        panel1.image = self.image1
        
        label = tk.Label(self, text="Deployment of light weight object detection model on drone", font=LARGE_FONT)
        label2 = tk.Label(self, text="to distinguish different objects", font=LARGE_FONT)
        label.pack(pady=20,padx=20,side="top", fill="both", expand = False)
        label2.pack()
        
        tk.Label(self, text="Objects like person, car, motorcycle and building can be detected from a drone footage", font="Verdana 10").pack(pady=15,padx=5)

        button = tk.Button(self, text="Detection on Video using YOLO",command=lambda: controller.show_frame(Viddet_YOLO))
        button.pack(pady=530,padx=110,side="right")
        
        button1 = tk.Button(self, text="Detection on Video using tinyYOLO",command=lambda: controller.show_frame(Viddet_tinyYOLO)).place(x=252, y=670)

        button2 = tk.Button(self, text="Detection on Live Feed",command=lambda: controller.show_frame(livefeed))
        button2.pack(pady=530,padx=110,side="left")
        
class Viddet_YOLO(tk.Frame): #object detection based on video
    def __init__(self, parent, controller):
        val=1
        tk.Frame.__init__(self, parent)
        
        label = tk.Label(self, text="Object detection using YOLOv3 on an input video", font=LARGE_FONT)
        label.pack(pady=50,padx=50,side="top", fill="both", expand = True)
        
        tk.Label(self, text="Enter location of .mp4 file").place(x=180, y=80)
        searched = tk.StringVar()
        
        tk.Button(self, text ="Browse", command=lambda:DisplayDir(searched)).place(x=300, y=110)
        
        search = tk.Entry(self, textvariable = searched).place(x=330, y=80)
        
        tk.Label(self, text = "Select the objects to be detected", font="Times 14").place(x=210, y=150)
        
        person = tk.IntVar()
        bike = tk.IntVar()
        car = tk.IntVar()
        building = tk.IntVar()
        tk.Checkbutton(self, text="Person", variable=person).place(x=310, y=180)
        tk.Checkbutton(self, text="Two-wheeler", variable=bike).place(x=310, y=210)
        tk.Checkbutton(self, text="Car", variable=car).place(x=310, y=240)
        tk.Checkbutton(self, text="Building", variable=building).place(x=310, y=270)
        
        tk.Button(self, text ="Detect", command = lambda: check_class(person,car,bike,building,val,searched)).place(x=300, y=310)

        button = tk.Button(self, text="Home",
                            command=lambda: controller.show_frame(StartPage))
        button.pack(pady=530,padx=130,side="left")
        
        button1 = tk.Button(self, text="Detection on Live Feed",
                            command=lambda: controller.show_frame(livefeed)).place(x=250, y=654)

        button2 = tk.Button(self, text="Detection on Video using tinyYOLO",command=lambda: controller.show_frame(Viddet_tinyYOLO))
        button2.pack(pady=530,padx=130,side="right")

class livefeed(tk.Frame):
    def __init__(self, parent, controller):
        val=2
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Object detection using YOLOv3 on live feed", font=LARGE_FONT)
        label.pack(pady=50,padx=50)
        
        tk.Label(self, text = "Select the objects to be detected", font="Times 14").place(x=210, y=80)
        
        person = tk.IntVar()
        bike = tk.IntVar()
        car = tk.IntVar()
        building = tk.IntVar()
        tk.Checkbutton(self, text="Person", variable=person).place(x=310, y=110)
        tk.Checkbutton(self, text="Two-wheeler", variable=bike).place(x=310, y=140)
        tk.Checkbutton(self, text="Car", variable=car).place(x=310, y=170)
        tk.Checkbutton(self, text="Building", variable=building).place(x=310, y=200)
        
        tk.Button(self, text ="Detect", command = lambda: check_class(person,car,bike,building,val)).place(x=300, y=240)

        button = tk.Button(self, text="Home",
                            command=lambda: controller.show_frame(StartPage))
        button.pack(pady=530,padx=130,side="left")
        
        button1 = tk.Button(self, text="Detection on Video using tinyYOLO",command=lambda: controller.show_frame(Viddet_tinyYOLO)).place(x=220, y=654)

        button2 = tk.Button(self, text="Detection on Video using YOLO",
                            command=lambda: controller.show_frame(Viddet_YOLO))
        button2.pack(pady=530,padx=130,side="right")
        
class Viddet_tinyYOLO(tk.Frame):
    def __init__(self, parent, controller):
        val=3
        tk.Frame.__init__(self, parent)
        
        label = tk.Label(self, text="Object detection using tinyYOLOv3 on an input video", font=LARGE_FONT)
        label.pack(pady=50,padx=50,side="top", fill="both", expand = True)
        
        tk.Label(self, text="Enter location of .mp4 file").place(x=180, y=80)
        searched = tk.StringVar()
        search = tk.Entry(self, textvariable = searched).place(x=330, y=80)
        
        tk.Button(self, text ="Browse", command=lambda:DisplayDir(searched)).place(x=300, y=110)
        
        tk.Label(self, text = "Select the objects to be detected", font="Times 14").place(x=210, y=150)
        
        person = tk.IntVar()
        bike = tk.IntVar()
        car = tk.IntVar()
        building = tk.IntVar()
        tk.Checkbutton(self, text="Person", variable=person).place(x=310, y=180)
        tk.Checkbutton(self, text="Two-wheeler", variable=bike).place(x=310, y=210)
        tk.Checkbutton(self, text="Car", variable=car).place(x=310, y=240)
        tk.Checkbutton(self, text="Building", variable=building).place(x=310, y=270)
        
        tk.Button(self, text ="Detect", command = lambda: check_class(person,car,bike,building,val,searched)).place(x=300, y=310)

        button = tk.Button(self, text="Home",
                            command=lambda: controller.show_frame(StartPage))
        button.pack(pady=530,padx=130,side="left")
        
        button1 = tk.Button(self, text="Detection on Video using YOLO",command=lambda: controller.show_frame(Viddet_YOLO)).place(x=250, y=654)

        button2 = tk.Button(self, text="Detection on Live Feed",
                            command=lambda: controller.show_frame(livefeed))
        button2.pack(pady=530,padx=130,side="right")


def DisplayDir(Var):
    feedback = askopenfilename()
    Var.set(feedback)

def check_class(person,car,bike,building,val,searched='live feed'):
    object_det = []
    if(person.get() == 1):
        object_det.append('person')
    if(car.get() == 1):
        object_det.append('car')
    if(bike.get() == 1):
        object_det.append('bike')
    if(building.get() == 1):
        object_det.append('building')
    if(searched!='live feed'):
        text = searched.get()
    if(val==1):
        vid_det_YOLO(text,object_det)
    elif(val==2):
        live_feed(object_det)
    elif(val==3):
        vid_det_tinyYOLO(text,object_det)
    #give 'text' and array 'object_det' as input to code file
    #<enter function name here>#
        
    


app = ObjDet()
app.mainloop()