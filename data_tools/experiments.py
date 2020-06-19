import os
import pickle
import torch
import json
import numpy as np
import cv2 as cv
from results_postprocess import filter_boxinbox
color = {'ASC-H':(0,0,255), 'ASC-US':(255,0,0), 'HSIL':(0,255,0), 'LSIL':(255,0,255), 'Candida':(0, 255,255), 'Trichomonas':(255,255,0)}
def json_vis():
    #path = './tianchi_json/1001.json'
    path = '/home/admin/jupyter/Data/101.json'
    f = open(path)
    content = f.read()
    label_dict = json.loads(content)
    print(label_dict)

def recursive_glob():
    paths = './tianchi_json/'
    paths1=[]
    filenames = []
    for base, dirs, files in os.walk(paths):
        for file in files:
            if file.endswith('.json') and file[0].isdigit() and len(file)<=9:
                paths1.append(os.path.join(base, file))
                filenames.append(file.split('.')[0])
    print(len(filenames))


def object_sts():
    path ='/home/admin/jupyter/tianchi_data/train/ROI_labels'
    files = os.listdir(path)
    print(len(files))
    sts = []
    for file in files:
        file_path = os.path.join(path, file)
        f = open(file_path, 'r')
        lines = f.readlines()
        f.close()
        for line in lines:
            line = line.split()
            if line[4] == 'ASC-H' :
                area = np.sqrt(float(line[2])*float(line[3]))
                #area = float(line[2])/float(line[3])
                sts.append(area)
    print(len(sts))
    min_ = min(sts)
    max_ = max(sts)
    print(min_, max_)
    num = 10
    count = [0 for i in range(num)]
    for a in sts:
        for i in range(num):
            if a>min_+i*(max_-min_)/num and a < min_+(i+1)*(max_-min_)/num:
                count[i]+=1
    for i in range(num):
        print(min_+i*(max_-min_)/num, '~', min_+(i+1)*(max_-min_)/num, ':', count[i], '\n')



def vis():
  root='/home/admin/jupyter/'
  img_path  = root +'tianchi_data/train_1/images_train'
  labeldir_path = root +'tianchi_data/train_1/labels_train'
  f = open(root +'tianchi_data/train_1/VOC2007/ImageSets/Main/train.txt', 'r')
  image_names = [line.split('\n')[0] for line in f.readlines()]
  for name in image_names:
    print(name)
    image_path = os.path.join(img_path, name+'.jpg')
    img = cv.imread(image_path)
    label_path = os.path.join(labeldir_path, name+'.txt')
    f = open(label_path, 'r')
    for line in f.readlines():
      print(line)
      coord = line.split()
      x1 = int(float(coord[0])-float(coord[2])/2)
      y1 = int(float(coord[1])-float(coord[3])/2)
      x2 = int(float(coord[0])+float(coord[2])/2)
      y2 = int(float(coord[1])+float(coord[3])/2)
      img = cv.rectangle(img,(x1, y1 ),(x2,y2), (255,0,0), 4)
    cv.imwrite('/home/admin/jupyter/mmdetection-master/vis_val/'+name+'.jpg',img)
    
def vis_by_pkl():
  root='/home/admin/jupyter/'
  img_list = root +'tianchi_data/test/VOC2007_roi/ImageSets/Main/test.txt'
  img_path  = root +'tianchi_data/test/VOC2007_roi/JPEGImages'
  labeldir_path = root +'mmdetection-master/result/res_roi.pkl'
  pkl_file = open(labeldir_path, 'rb')
  data = pickle.load(pkl_file)
  print(len(data))
  f = open(img_list, 'r')
  image_names = [line.split('\n')[0] for line in f.readlines()]
  for i in range(len(image_names)):
    name = image_names[i]
    image_path = os.path.join(img_path, name+'.jpg')
    img = cv.imread(image_path)
    label_path = os.path.join(labeldir_path, name+'.txt')
    dt = data[i]
    for k in range(6):
        dt[k] = np.concatenate((dt[k], np.array([k for i in range(dt[k].shape[0])]).reshape(dt[k].shape[0],1)),1)
    tmp = dt[0]
    for k in range(1,6):
        tmp = np.concatenate((tmp, dt[k]),0)
    tmp = tmp[tmp[:,4]>0.1]
    for j in range(tmp.shape[0]):
      x1 = int(float(tmp[j][0]))
      y1 = int(float(tmp[j][1]))
      x2 = int(float(tmp[j][2]))
      y2 = int(float(tmp[j][3]))
      img = cv.rectangle(img,(x1, y1 ),(x2,y2), color[int(float(tmp[j][5]))], 3)
    cv.imwrite('/home/admin/jupyter/mmdetection-master/vis_roi/'+name+'.jpg',img)

    
def vis_by_json():
    ca = [1192, 142,1469, 1621, 2612, 47, 4809, 4811, 4970, 4983, 5523, 5748, 6676, 7390, 7689, 7948, 7980, 9462]
    path = '/home/admin/jupyter/mmdetection-master/tianchi_json/'
    for file in os.listdir(path):
        print(file[:file.rfind('roi')])
#         if int(file[:file.rfind('roi')]) not in ca:
#             continue
        f = open(path+file, 'r')
        data = json.load(f)
        img_path = '/home/admin/jupyter/tianchi_data/test/ROI_images/'+file[:-5]+'.jpg'
        img = cv.imread(img_path)
        for s in data:
            if s['p']>0.1:
                x1 = int(s['x'])
                y1 = int(s['y'])
                x2 = int(s['x']+s['w'])
                y2 = int(s['y']+s['h'])
                img = cv.rectangle(img,(x1, y1 ),(x2,y2), color[s['class']], 4)
                if s['class'] == 'ASC-H' or s['class'] == 'Candida' or s['class'] == 'Trichomonas':
                    img = cv.putText(img, s['class'] + str(round(s['p'],2)), (x1, y1 ), cv.FONT_HERSHEY_COMPLEX, 2, color[s['class']], 3)
                    #img = cv.putText(img, str(round(s['p'],2)), (x2,y2), cv.FONT_HERSHEY_COMPLEX, 2, color[s['class']], 3)
                elif s['class'] == 'ASC-US':
                    img = cv.putText(img, s['class'] + str(round(s['p'],2)), (x2, y1 ), cv.FONT_HERSHEY_COMPLEX, 2, color[s['class']], 3)
                elif s['class'] == 'HSIL':
                    img = cv.putText(img, s['class'] + str(round(s['p'],2)), (x1, y2 ), cv.FONT_HERSHEY_COMPLEX, 2, color[s['class']], 3)
                else:
                    img = cv.putText(img, s['class'] + str(round(s['p'],2)), (x2, y2 ), cv.FONT_HERSHEY_COMPLEX, 2, color[s['class']], 3)
        cv.imwrite('/home/admin/jupyter/mmdetection-master/vis_test_roi/'+file[:-5]+'.jpg',img)

        
def vis_roi_gt():
  path = '/home/admin/jupyter/tianchi_data/train/ROI_images/'
  name = []
  for file in os.listdir(path):
      f = open('/home/admin/jupyter/tianchi_data/train/ROI_labels/'+file[:-4]+'.txt', 'r')
      lines = f.readlines()
      f.close()
      tmp = []
      for line in lines:
          line = line.split('\n')[0].split()
          print(line)
          if line[4]!='Candida':
              continue
          else:
              tmp.append([int(float(line[0])), int(float(line[1])), int(line[2]), int(line[3])])
      if not tmp:
          continue
      img = cv.imread(path + file)
      for s in tmp:
          img = cv.rectangle(img, (int(s[0]-s[2]/2), int(s[1]-s[3]/2)), (int(s[0]+s[2]/2), int(s[1]+s[3]/2)), color['Candida'], 4)
      cv.imwrite('/home/admin/jupyter/mmdetection-master/vis_ca_gt/'+file[:-4]+'.jpg',img)
    
import shutil
def generate_backup():
    src_path = '/home/admin/jupyter/tianchi_data/train/labels_train/'
    dst_path = '/home/admin/jupyter/tianchi_data/train/backup/labels_train/'
    if not os.path.exists(dst_path):
        os.mkdir(dst_path)
    files = os.listdir(src_path)
    for file in files:
        shutil.copy(src_path+file, dst_path + file)
def label_vis():
  src_path = '/home/admin/jupyter/tianchi_data/train/labels_train/'
  files = os.listdir(src_path)
  for file in files:
    f= open(src_path +file)
    lines = f.readlines()
    f.close()
    if not lines:
      continue
    box = []
    for line in lines:
      line = line.split()
      box.append([float(line[0])-float(line[2])/2, float(line[1])-float(line[3])/2 ,float(line[0])+float(line[2])/2, float(line[1])+float(line[3])/2])
    iou = intersect(torch.from_numpy(np.array(box)),torch.from_numpy(np.array(box)))
    if ((iou>0.5).sum()-iou.shape[0])>0:
      print(file, lines)      
    
def train_test_val(): 
  path= '/home/admin/jupyter/tianchi_data_512/train/VOC2007_512/ImageSets/Main/test.txt'
  path1 = '/home/admin/jupyter/tianchi_data_512/train/VOC2007_512/ImageSets/Main/train.txt'
  f = open(path , 'r')
  lines = f.readlines()
  f.close()
  test = []
  for line in lines:
    test.append(line.split('_')[0])
  f = open(path1, 'r')
  lines = f.readlines()
  f.close()
  train = []
  for line in lines:
    train.append(line.split('_')[0])
  print('train:', len(train), 'test:', len(test))
  print('train:', len(set(train)), 'test:', len(set(test)))
  for s in test:
    if s in train:
        print('error')
    
def diff_cat_nms():
  path = '/home/admin/jupyter/mmdetection-master/result/res_multi_aug_8603.pkl'
  pkl_file = open(path, 'rb')
  data = pickle.load(pkl_file)
  for i in range(len(data)):
    for j in range(4):
      box_a = data[i][j]
      box_b = np.array([])
      flag = 0
      for k in range(4):
        if k!=j:
          if flag == 0:
            box_b = data[i][k]
            flag = 1
          else:
            box_b = np.concatenate((box_b, data[i][k]), 0)
      data[i][j]= np.concatenate((compare(box_a[box_a[:,4]>0.1], box_b, ratio=1, thres=0.9), box_a[box_a[:,4]<=0.1]),0)
  save_path = path[:-4]+'backup.pkl'
  with open(save_path, 'wb') as f:
    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    
def compare(a, b, ratio=1, thres=0.1):
  boxes_a = torch.from_numpy(a[:,:4])
  boxes_b = torch.from_numpy(b[:,:4])
  iou = intersect(boxes_a, boxes_b).data.numpy()
  index = []
  for i in range(iou.shape[0]):
    #if ((iou[i]>thres)*((a[i, 4]+0.3)<b[:,4])).sum()>0:
    if (iou[i]>thres).sum()>0:
      a[i,4] = a[i,4]+0.2
      index.append(i)
      #continue
    else:
      index.append(i)
  return a[index]

def intersect(box_a, box_b):
  """ We resize both tensors to [A,B,2] without new malloc:
  [A,2] -> [A,1,2] -> [A,B,2]
  [B,2] -> [1,B,2] -> [A,B,2]
  Then we compute the area of intersect between box_a and box_b.
  Args:
    box_a: (tensor) bounding boxes, Shape: [A,4].
    box_b: (tensor) bounding boxes, Shape: [B,4].
  Return:
    (tensor) intersection area, Shape: [A,B].
  """
  A = box_a.size(0)
  B = box_b.size(0)
  max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                     box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
  min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                     box_b[:, :2].unsqueeze(0).expand(A, B, 2))
  inter = torch.clamp((max_xy - min_xy), min=0)
  w_a = (box_a[:, 2]-box_a[:, 0]).unsqueeze(1).expand(A, B)
  h_a = (box_a[:, 3] - box_a[:, 1]).unsqueeze(1).expand(A, B)
  w_b = (box_b[:, 2]-box_b[:, 0]).unsqueeze(0).expand(A, B)
  h_b = (box_b[:, 3] - box_b[:, 1]).unsqueeze(0).expand(A, B)
  union = w_a* h_a + w_b* h_b - (inter[:, :, 0] * inter[:, :, 1])
  return (inter[:, :, 0] * inter[:, :, 1])/union

def filter():
  path = '/home/admin/jupyter/mmdetection-master/result/res_multi_aug_8603.pkl'
  pkl_file = open(path, 'rb')
  data = pickle.load(pkl_file)
  for i in range(len(data)):
    for j in range(len(data[i])):
      data[i][j] = np.concatenate((data[i][j][data[i][j][:,4]<=0.1], filter_boxinbox(data[i][j][data[i][j][:,4]>0.1])), 0)
  save_path = path[:-4]+'backup.pkl'
  with open(save_path, 'wb') as f:
    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)                                        


if __name__ == '__main__':
    #train_test_val()
  # object_sts()
  #recursive_glob()
  # generate_backup()
  #vis()
  #vis_by_json()
  #filter_boxinbox('/home/admin/jupyter/mmdetection-master/result/res.pkl')
  #vis_by_pkl()
  vis_roi_gt()
  #diff_cat_nms()
  #filter()
#   path = '/home/admin/jupyter/9379.json'
#   f = open(path, 'r')
#   data = json.load(f)
#   tmp = []
#   for s in data:
#     if s['class']=='roi':
#       continue
#     else:
#       tmp.append(s)
#   print(len(data))
#   print(len(tmp))
#   save_dir = '/home/admin/jupyter/9379_1.json'
#   f =open(save_dir, 'w')
#   json.dump(tmp ,f)
#   f.close()
      
 
  

  
                                                  
    
      
 

   
           
    