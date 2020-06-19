import os
import pickle
import torch
import json
import numpy as np
import cv2 as cv
import shutil
import random
import tqdm
from data_tools.results_postprocess import re_nms, intersect, filter_boxinbox
color = [(0,0,255), (255,0,0), (0,255,0), (255,0,255),(0, 255,255),(255,255,0)]
def images_count():
    path = '/home/admin/jupyter/tianchi_data/train_1/labels_train'
    empty = neg = pos = 0
    for file in os.listdir(path):
        if file.find('empty')!=-1:
            empty+=1
        elif file.find('segment')!=-1:
            neg+=1
        else:
            pos+=1
    print('pos:', pos, 'neg:', neg, 'empty:', empty)

def delete_label():
    path = '/home/admin/jupyter/tianchi_data/train/labels_train/'
    for file in os.listdir(path):
        if file.find('empty')==-1 and file.find('segment')==-1:
            f = open(path+file, 'r')
            lines = f.readlines()
            f.close()
            tmp = []
            for line in lines:
                if line.find('NILM')==-1:
                    tmp.append(line)
            f = open(path+file, 'w')
            for s in tmp:
                f.write(s)
            f.close()
def data_merge():
    label_path= '/home/admin/jupyter/tianchi_data/train_1/labels_train/'
    img_path = '/home/admin/jupyter/tianchi_data/train_1/images_train/'
    txt_path = '/home/admin/jupyter/tianchi_data/train_1/VOC2007/ImageSets/Main/test.txt'
    pos = []
    neg = []
    empty = []
    f = open(txt_path , 'r')
    for line in f.readlines():
        if line.find('empty')!=-1:
            empty.append(line.split('\n')[0])
        elif line.find('segment')!=-1:
            neg.append(line.split('\n')[0])
        else:
            pos.append(line.split('\n')[0])
    random.shuffle(pos)
    em = neg+empty
    name = []
    for i in range(len(em)):
        name.append('image_'+str(i))
        img_1 = cv.imread(img_path + em[i] + '.jpg')
        img_2 = cv.imread(img_path + pos[i] + '.jpg')
        img = np.concatenate((img_2, img_1), 1)
        cv.imwrite(img_path+'image_'+str(i)+'.jpg', img)
        f = open(label_path+ pos[i] + '.txt', 'r')
        lines = f.readlines()
        f.close()
        f = open(label_path+ 'image_'+str(i)+ '.txt', 'w')
        for line in lines:
            f.write(line)
    for i in range(int((len(pos)-len(em))/2)):
        name.append('image_'+str(len(em)+i))
        img_1 = cv.imread(img_path + pos[len(em)+2*i] + '.jpg')
        img_2 = cv.imread(img_path + pos[len(em)+2*i+1] + '.jpg')
        img = np.concatenate((img_2, img_1), 1)
        cv.imwrite(img_path+'image_'+str(len(em)+i)+'.jpg', img)
        f = open(label_path+ pos[len(em)+2*i] + '.txt', 'r')
        lines = f.readlines()
        f.close()
        f = open(label_path+ pos[len(em)+2*i+1] + '.txt', 'r')
        lines += f.readlines()
        f.close()
        f = open(label_path+ 'image_'+str(len(em)+i)+ '.txt', 'w')
        for line in lines:
            f.write(line)
    f = open(txt_path, 'w')
    for s in name:
        f.write(s+'\n')
    f.close()


def nms(dets, thresh):
    
        
    if dets.shape[0] == 0:
        return dets.reshape(dets.shape[0],5)
   
    #print(dets.shape)
    if len(dets.shape)==1:
        dets= dets.reshape(1,dets.shape[0])
    if dets.shape[1]!=5:
        dets = dets.reshape(-1,5)
#     x1 = dets[:, 0]
#     y1 = dets[:, 1]
#     x2 = dets[:, 2]
#     y2 = dets[:, 3]
#     scores = dets[:, 4]
#     results=[]


#     areas = (x2 - x1 + 1) * (y2 - y1 + 1)
#     order = scores.argsort()[::-1]

#     keep = []
#     while order.size > 0:
#         i = order[0]
#         keep.append(i)
#         xx1 = np.maximum(x1[i], x1[order[1:]])
#         yy1 = np.maximum(y1[i], y1[order[1:]])
#         xx2 = np.minimum(x2[i], x2[order[1:]])
#         yy2 = np.minimum(y2[i], y2[order[1:]])

#         w = np.maximum(0.0, xx2 - xx1 + 1)
#         h = np.maximum(0.0, yy2 - yy1 + 1)
#         inter = w * h
#         ovr = inter / (areas[i] + areas[order[1:]] - inter)
#         inds = np.where(ovr <= thresh)[0]
#         order = order[inds + 1]
    return dets   

if __name__ == '__main__':
    #images_count()
    delete_label()
    #data_merge()
#     path = '/home/admin/jupyter/pkl_file/submission1101.pkl'
#     pkl_file = open(path, 'rb')
#     data = pickle.load(pkl_file)
#     print(len(data))
#     for i in range(len(data)):
#         print(i)
#         tmp = []
#         for s in data[i]:
#             tmp.append(nms(s,0.4))
#         data[i]=tmp
#     with open('/home/admin/jupyter/pkl_file/submission.pkl', 'wb') as f:
#         pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    
            
        
        
        
   

        
    
    

  
  
                                                  
    
      
 

   
           
    