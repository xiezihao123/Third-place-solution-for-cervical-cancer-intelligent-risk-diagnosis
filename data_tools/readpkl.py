import pickle
import numpy as np
import os
import random
import cv2
import shutil
from tqdm import tqdm
pkl_file=open("/home/admin/jupyter/mmdetection-master/result/empty_dets.pkl",'rb')
txt_flie="/home/admin/jupyter/tianchi_data/train/VOC2007_neg/ImageSets/Main/test.txt"
txt_save="/home/admin/jupyter/tianchi_data/train/neg_select.txt"
label_dir="/home/admin/jupyter/tianchi_data/train/neg_select_labels/"
save_img_dir="/home/admin/jupyter/tianchi_data/train/neg_select_images/"
img_dir="/home/admin/jupyter/tianchi_data/train/neg_segment_images/"

def get_image_number():
    data=pickle.load(pkl_file)
    print(len(data))
    count = 0
    for i in range(len(data)):
        if (data[i][0][:,4]>0.5).sum()>0 or (data[i][1][:,4]>0.5).sum()>0 or (data[i][2][:,4]>0.5).sum()>0 or (data[i][3][:,4]>0.5).sum()>0 or (data[i][4][:,4]>0.5).sum()>0 or (data[i][5][:,4]>0.5).sum()>0:
            count += 1
    print(count)
def generate_test():
    path = '/home/admin/jupyter/tianchi_data/train/VOC2007/ImageSets/Main/test.txt'
    test_roi = []
    f = open(path , 'r')
    lines = f.readlines()
    f.close()
    for line in lines:
        test_roi.append(line.split('_')[0])
    test_roi = list(set(test_roi))
    
    print(len(test_roi))
    data=pickle.load(pkl_file)
    print(len(data))
    f=open(txt_flie,'r')
    file_list=f.readlines()
    f.close()
    test = []
    for i in tqdm(range(len(data))):
        bs = data[i][0] #prediction
        for k in range(1, len(data[i])):
            bs = np.concatenate((bs, data[i][k]), 0)
        if (bs[:,4]>0.1).sum()>0 and file_list[i].split('_')[0] in test_roi:
            test.append(file_list[i].split('\n')[0])
    random.shuffle(test)
    tmp = []
    for i in range(3000):
        img_name = test[i]+'.jpg'
        save_name = test[i].split('_')[0]+'_'+'empty'+test[i].split('_')[1]+'.jpg'
        lb_name = test[i]+'.txt'
        lb_save_name = test[i].split('_')[0]+'_'+'empty'+test[i].split('_')[1]+'.txt'
        shutil.copy('/home/admin/jupyter/tianchi_data/train/ROI_empty_images/'+img_name, '/home/admin/jupyter/tianchi_data/train/images_train/' + save_name)
        shutil.copy('/home/admin/jupyter/tianchi_data/train/ROI_empty_labels/'+lb_name, '/home/admin/jupyter/tianchi_data/train/labels_train/' + lb_save_name)
        tmp.append(lb_save_name[:-4])
    f = open('/home/admin/jupyter/tianchi_data/train/VOC2007/ImageSets/Main/test.txt', 'a')
    for s in tmp:
        f.write(s+'\n')
    f.close()
def generate_train():
    data=pickle.load(pkl_file)
    print(len(data))
    score_list=[]
    z=0
    num=0
    num_pic=0
    f=open(txt_flie,'r')
    file_list=f.readlines()
    f.close()
    f1=open(txt_save,'w')
    name_list=[]
    for i in tqdm(range(len(data))):
        rnd=False
        rn=random.randint(1,6)
        if ((rn==2)):
            rnd=True
        ind=False
        bs = data[i][0] #prediction
        for k in range(1, len(data[i])):
            bs = np.concatenate((bs, data[i][k]), 0)
        a_list=[]
        pic_name=file_list[i].rstrip('\n')
        label_path=os.path.join(label_dir,pic_name+'.txt')
        img_path=os.path.join(img_dir,pic_name+'.jpg')
        save_img_path=os.path.join(save_img_dir,pic_name+'.jpg')
        box_list=[]
        if (bs.shape[0]!=0)and(rnd):
            for j in range(bs.shape[0]):
                if bs[j][4]>0.5:
                    ind=True
            for j in range(bs.shape[0]):
                if (ind)and(rnd):
                    if bs[j][4]>0.5:
                        num = num + 1
                        xmin = bs[j][0]
                        ymin = bs[j][1]
                        xmax = bs[j][2]
                        ymax = bs[j][3]
                        center_x = round((xmin + xmax) / 2, 1)
                        center_y = round((ymin + ymax) / 2, 1)
                        w = round(xmax - xmin, 1)
                        h = round(ymax - ymin, 1)
                        bbox_mess = ' '.join([str(center_x), str(center_y), str(w), str(h)]) + ' NILM' + '\n'
                        box_list.append(bbox_mess)
            if (ind)and(rnd):
                num_pic=num_pic+1
                txt_save_file=open(label_path,'w')
                txt_save_file.writelines(box_list)
                txt_save_file.close()
                # ig=cv2.imread(img_path)
                # cv2.imwrite(save_img_path,ig)
                shutil.copy(img_path, save_img_path)
                name_list.append(file_list[i])

    f1.writelines(name_list)
    print("the total num of picture is %d"%num_pic)
    print("the total num of empty boxes is %d"%num)

def filter():
    import os
    path = '/home/admin/jupyter/tianchi_data/train/VOC2007/ImageSets/Main/test.txt'
    test_roi = []
    f = open(path , 'r')
    lines = f.readlines()
    f.close()
    for line in lines:
        test_roi.append(line.split('_')[0])
    test_roi = list(set(test_roi))
    print(len(test_roi))
    tmp = []
    txt_save="/home/admin/jupyter/tianchi_data/train/empyt_select.txt"
    f = open(txt_save, 'r')
    lines = f.readlines()
    f.close()
    for line in lines:
        if line.split('_')[0] in test_roi:
            os.remove("/home/admin/jupyter/tianchi_data/train/empty_select_images/"+line.split('\n')[0]+'.jpg')
            os.remove("/home/admin/jupyter/tianchi_data/train/empty_select_labels/"+line.split('\n')[0]+'.txt')
            continue
        else:
            tmp.append(line)
    f = open(txt_save, 'w')
    for s in tmp:
        f.write(s)
    f.close()
    
def merge():
    path = '/home/admin/jupyter/tianchi_data/train/empty_select_labels/'
    for file in os.listdir(path):
        shutil.copy(path + file , '/home/admin/jupyter/tianchi_data/train/labels_train/'+file)
    path = '/home/admin/jupyter/tianchi_data/train/empty_select_images/'
    for file in os.listdir(path):
        shutil.copy(path + file , '/home/admin/jupyter/tianchi_data/train/images_train/'+file)
    f = open('/home/admin/jupyter/tianchi_data/train/empyt_select.txt', 'r')
    lines = f.readlines()
    f.close()
    f = open('/home/admin/jupyter/tianchi_data/train/VOC2007/ImageSets/Main/train.txt', 'a')
    for line in lines:
        f.write(line)
    f.close()
def file_rename():
    path = "/home/admin/jupyter/tianchi_data/train/empyt_select.txt"
    img_path = "/home/admin/jupyter/tianchi_data/train/empty_select_images/"
    lbt_path = "/home/admin/jupyter/tianchi_data/train/empty_select_labels/"
    f = open(path , 'r')
    lines = f.readlines()
    f.close()
    tmp = []
    for line in lines:
        new_name = line.split('_')[0]+'_'+'empty' + line.split('_')[1]
        tmp.append(new_name)
        os.rename(lbt_path+line.split('\n')[0]+'.txt',lbt_path+new_name.split('\n')[0]+'.txt' )
        os.rename(img_path+line.split('\n')[0]+'.jpg',img_path+new_name.split('\n')[0]+'.jpg' )
    f =open(path , 'w')
    for s in tmp:
        f.write(s)
    f.close()
if __name__=='__main__':
    #get_image_number()
    #generate_train()
    #filter()
    #merge()
    data=pickle.load(pkl_file)
    print(len(data))
    print(len(data[0]))
    
            
            

   
   
        
    

