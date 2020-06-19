import os
import numpy as np
import pickle
import cv2
#import results_postprocess.intersect as iou
from results_postprocess import intersect as iou
from results_postprocess import re_nms_neg as nms
import torch
pkl_path="/home/admin/jupyter/mmdetection-master/result/trainset_dets.pkl"
txt_flie="/home/admin/jupyter/tianchi_data/train/VOC2007/ImageSets/Main/train.txt"
ground_truth_dir="/home/admin/jupyter/tianchi_data/train/backup/labels_train/"
save_label_dir="/home/admin/jupyter/tianchi_data/train/labels_train/"
pkl_file=open(pkl_path,'rb')
data=pickle.load(pkl_file)
name_txt=open(txt_flie,'r')
name_list=[line for line in name_txt.readlines() if line.find('empty')==-1 and line.find('segment')==-1]
print(len(name_list))
assert len(data)==len(name_list), 'error'
lall=0
neg_num_box=0
pos_num_box=0

for name in name_list:
    name=name.rstrip('\n')
    print(name)
    bs = data[lall][0] #prediction
    for i in range(1, len(data[lall])):
        bs = np.concatenate((bs, data[lall][i]), 0)
    print(lall)
    boxes_prediction=[]
    boxes_gt=[]
    neg_boxes=[]
    if bs.shape[0]!=0:
        for b in range(bs.shape[0]):
            if(bs[b][4]>0.5):
                box=bs[b][0:4]
                boxes_prediction.append(box.tolist())
    ground_truth_path=os.path.join(ground_truth_dir,name+'.txt')
    #print("boxes_prediction:")
    #print(boxes_prediction)
    f1=open(ground_truth_path,'r')
    gt_box_lists=f1.readlines()
    f1.close()
    save_label_path=os.path.join(save_label_dir,name+'.txt')

    for gt_box in gt_box_lists:
        gt_box_centerpoint=list(map(float,(gt_box.split(' '))[0:4]))
        gt_box_point=[gt_box_centerpoint[0]-(gt_box_centerpoint[2]/2),gt_box_centerpoint[1]-(gt_box_centerpoint[3]/2),gt_box_centerpoint[0]+(gt_box_centerpoint[2]/2),gt_box_centerpoint[1]+(gt_box_centerpoint[3]/2)]
        boxes_gt.append(gt_box_point)
    box_prediction_tensor=torch.tensor(boxes_prediction)
    box_gt_tensor=torch.tensor(boxes_gt)
    # print(box_prediction.size())
    # print(box_gt.size(0))
    negbox_id_list = []
    if box_gt_tensor.size(0)*box_prediction_tensor.size(0)!=0:
        box_c=iou(box_prediction_tensor,box_gt_tensor)
        box_c_list=box_c.numpy()
        #print(box_c_list)
        m=(np.max(box_c_list,axis=1))
        for i in range(len(m)):
            if(m[i]<0.1):
                negbox_id_list.append(i)
        for j in negbox_id_list:
            neg_boxes.append(boxes_prediction[j])
    #print("neg_boxes:")
    #print(neg_boxes)
    if ((box_gt_tensor.size(0)==0)and(box_prediction_tensor.size(0))!=0):
        for l in boxes_prediction:
            neg_boxes.append(l)

    result_neg_boxes=nms(neg_boxes)
    result = []
    f_pos=open(ground_truth_path,'r')
    f_pos_list=f_pos.readlines()
    f_pos.close()
    for yayay in f_pos_list:
        result.append(yayay)
        pos_num_box=pos_num_box+1
    for lll in result_neg_boxes:
        neg_num_box=neg_num_box+1
        xmin=lll[0]
        ymin=lll[1]
        xmax=lll[2]
        ymax=lll[3]
        center_x=round((xmin+xmax)/2,1)
        center_y=round((ymin+ymax)/2,1)
        w=round(xmax-xmin,1)
        h=round(ymax-ymin,1)
        bbox_mess=' '.join([str(center_x),str(center_y),str(w),str(h)])+' NILM'+'\n'
        #print(type(bbox_mess))
        result.append(bbox_mess)
    #print("result:")
    #print(result)
    f2=open(save_label_path,'w')
    f2.writelines(result)
    f2.close()
    lall=lall+1
print("the total num of neg box is %d "%neg_num_box)
print("the total num of pos box is %d "%pos_num_box)



