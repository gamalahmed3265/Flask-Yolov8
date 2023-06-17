import os
from detetions.tracker import TrackerMaster
import numpy as np
import cv2 as cv
import pandas as pd


    
# area1=[(312,388),(289,390),(474,469),(497,462)]

# area2=[(279,392),(250,397),(423,477),(454,469)]

font=cv.FONT_HERSHEY_COMPLEX

def RGB(event, x, y, flags, param):
    if event == cv.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)

class ConterMaster:
    def __init__(self):
        self.tracker=TrackerMaster()
            
    def conter(self,results,classList):
        a=results.boxes.data
        #print("a",a)
        px=pd.DataFrame(a).astype("float")
        #print(px)
        list=[]
                
        for index,row in px.iterrows():
            #print(row)
            x1=int(row[0])
            y1=int(row[1])
            x2=int(row[2])
            y2=int(row[3])
            d=int(row[5])
            print(d)
            
            c=classList[d]
            
            if 'person' in c:
            
                list.append([x1,y1,x2,y2])  
            
        bbox_id = self.tracker.update(list)
        
        for bbox in bbox_id:
            x3,y3,x4,y4,id = bbox
            print(id," id")
        # print(bbox_id)
        # return id