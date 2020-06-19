import pickle
import numpy as np
import os
import random
import cv2
from tqdm import tqdm
def pseudo_labels():
    pkl_file=open("/home/admin/jupyter/mmdetection-master/result/res_multi_1024_16944.pkl",'rb')
    txt_flie="/home/admin/jupyter/tianchi_data/test/VOC2007/ImageSets/Main/test.txt"
    txt_save="/home/admin/jupyter/tianchi_data/train/pseudo/pseudo.txt"
    label_dir="/home/admin/jupyter/tianchi_data/train/pseudo/pseudo_label/"
    save_img_dir="/home/admin/jupyter/tianchi_data/train/pseudo/pseudo_image/"
    img_dir="/home/admin/jupyter/tianchi_data/train/images_train/"
    data=pickle.load(pkl_file)
    score_list=[]
    z=0
    num_neg=0
    num_pos=0
    num_pic=0
    f=open(txt_flie,'r')
    file_list=f.readlines()
    f.close()
    f1=open(txt_save,'w')
    name_list=[]
    count = 0
    print(len(data))
    for i in tqdm(range(len(data))):
        for j in range(len(data[i])):
            if (data[i][j][:,4]>0.8).sum()>0:
                count += 1
                continue
    print(count)
#         rnd=False
#         rn=random.randint(1,12)
#         #print(rn)
#         if ((rn==2)):
#             rnd=True
#         ind=False
#         bs=data[i][0]  #pos
#         cs=data[i][1]   #neg
#         a_list=[]
#         pic_name=file_list[i].rstrip('\n')
#         label_path=os.path.join(label_dir,pic_name+'.txt')
#         img_path=os.path.join(img_dir,pic_name+'.jpg')
#         save_img_path=os.path.join(save_img_dir,pic_name+'.jpg')
#         box_list=[]
#         if cs.shape[0]!=0:
#             for pp in range(cs.shape[0]):
#                 if cs[pp][4]>0.7:
#                     ind=True
#             for j in range(bs.shape[0]):
#                 if (ind)and(rnd):
#                     if bs[j][4]>0.5:
#                         num_pos = num_pos + 1
#                         xmin = bs[j][0]
#                         ymin = bs[j][1]
#                         xmax = bs[j][2]
#                         ymax = bs[j][3]
#                         center_x = round((xmin + xmax) / 2, 1)
#                         center_y = round((ymin + ymax) / 2, 1)
#                         w = round(xmax - xmin, 1)
#                         h = round(ymax - ymin, 1)
#                         bbox_mess = ' '.join([str(center_x), str(center_y), str(w), str(h)]) + ' pos' + '\n'
#                         box_list.append(bbox_mess)
#             for l in range(cs.shape[0]):
#                 if (ind)and(rnd):
#                     if cs[l][4]>0.5:
#                         num_neg = num_neg + 1
#                         xmin = cs[l][0]
#                         ymin = cs[l][1]
#                         xmax = cs[l][2]
#                         ymax = cs[l][3]
#                         center_x = round((xmin + xmax) / 2, 1)
#                         center_y = round((ymin + ymax) / 2, 1)
#                         w = round(xmax - xmin, 1)
#                         h = round(ymax - ymin, 1)
#                         bbox_mess = ' '.join([str(center_x), str(center_y), str(w), str(h)]) + ' neg' + '\n'
#                         box_list.append(bbox_mess)
#         if (ind)and(rnd):
#             num_pic=num_pic+1
#             txt_save_file=open(label_path,'w')
#             txt_save_file.writelines(box_list)
#             txt_save_file.close()
#             ig=cv2.imread(img_path)
#             cv2.imwrite(save_img_path,ig)
#             name_list.append(file_list[i])
#             #print(pic_name)

#     f1.writelines(name_list)
#     print("the total num of picture is %d"%num_pic)
#     print("the total num of pos boxes is %d"%num_pos)
#     print("the total num of neg boxes is %d"%num_neg)

if __name__=='__main__':
    pseudo_labels()