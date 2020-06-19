import cv2
import numpy
import pickle
import numpy as np
import xml.etree.ElementTree as ET
import os

pkl_file_path="/home/hust4/github_flies/mmdetection-master/result/result_cascade_hrnet.pkl"
xml_dict="/mnt/C/tianchi/train/VOC2007-/Annotations/"
img_dict="/mnt/C/tianchi/train/VOC2007-/JPEGImages/"
img_save_dict="/mnt/C/tianchi/box_predict_result/"
test_txt="/mnt/C/tianchi/train/VOC2007-/ImageSets/Main/train.txt"

pkl_file=open(pkl_file_path,'rb')
data=pickle.load(pkl_file)

def get(root, name):
    vars = root.findall(name)
    return vars

def get_and_check(root, name, length):
    vars = root.findall(name)
    if len(vars) == 0:
        raise NotImplementedError('Can not find %s in %s.'%(name, root.tag))
    if length > 0 and len(vars) != length:
        raise NotImplementedError('The size of %s is supposed to be %d, but is %d.'%(name, length, len(vars)))
    if length == 1:
        vars = vars[0]
    return vars

for i in range(len(data)):
    bs=np.squeeze(np.asarray(data[i]),axis=0)

f_txt=open(test_txt,'r')
test_list=f_txt.readlines()
print(len(test_list))
z=0
for image_name in test_list:
    image_name=image_name.rstrip('\n')
    print(image_name)
    image_path=img_dict+image_name+'.jpg'
    img=cv2.imread(image_path,1)
    #draw the predict green
    bs = np.squeeze(np.asarray(data[z]), axis=0)
    if bs.shape[0] != 0:
        for j in range(bs.shape[0]):
            if bs[j][4]<0.25:
                continue
            xmin,ymin,xmax,ymax=bs[j][0],bs[j][1],bs[j][2],bs[j][3]
            cv2.rectangle(img,(xmin,ymin),(xmax,ymax),(0,255,0),3)
    z=z+1
    #draw the ground_truth red
    xml_path=xml_dict+image_name+'.xml'
    tree = ET.parse(xml_path)
    root = tree.getroot()
    for obj in get(root, 'object'):
        bndbox = get_and_check(obj, 'bndbox', 1)
        xmin = int(get_and_check(bndbox, 'xmin', 1).text) - 1
        ymin = int(get_and_check(bndbox, 'ymin', 1).text) - 1
        xmax = int(get_and_check(bndbox, 'xmax', 1).text)
        ymax = int(get_and_check(bndbox, 'ymax', 1).text)
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 3)
    cv2.imwrite(os.path.join(img_save_dict,image_name+".jpg"),img)






