import pickle
import json
import numpy as np
import torch
import os
CLASSES = ['ASC-H','ASC-US','HSIL','LSIL' ,'Candida', 'Trichomonas']
def results_postprocess():
  path = './tianchi_json/'
  files = os.listdir(path)
  for file in files:
    if os.path.isdir(path+file):
      os.rmdir(path+file)
    else:
      os.remove(path+file)
  pkl_file = open('/home/admin/jupyter/mmdetection-master/result/res_multi_aug.pkl', 'rb')
  data = pickle.load(pkl_file)
  print(len(data))
  test_list_path = '/home/admin/jupyter/tianchi_data/test/VOC2007/ImageSets/Main/test.txt'
  result = dict()
  f = open(test_list_path, 'r')
  lines = f.readlines()
  print(len(lines))
  assert len(data)==len(lines)
  for i in range(len(lines)):
    if data[i][0].shape[0]==0 and data[i][1].shape[0]==0 and data[i][2].shape[0]==0 and data[i][3].shape[0]==0 and data[i][4].shape[0]==0 and data[i][5].shape[0]==0:
      continue
    lines[i] = lines[i].split('\n')[0]
    name = lines[i][:lines[i].rfind('seg')]
    suffix = lines[i][lines[i].rfind('seg')+4:]
    #print('name:', name, 'suffix:', suffix)
    if name not in result.keys():
      result[name] = dict()
      dt_tmp = coord_transformer(data[i], suffix, name)
      result[name][suffix] = dt_tmp
    else:
      dt_tmp = coord_transformer(data[i], suffix, name)
      result[name][suffix] = dt_tmp
  duplicate_removal(result)

def coord_transformer(data, suffix, name):
  index = suffix.split('_')
  i = int(index[0])
  j = int(index[1])
  for k in range(6):
    data[k] = data[k] + np.array([j*768, i*768, j*768, i*768, 0])
#   f = open('/home/admin/jupyter/tianchi_data/test/ROI_coord/'+name+'.txt', 'r')
#   lines = f.readlines()
#   f.close()
#   coord = lines[0].split()
#   for k in range(6):
#     data[k] += np.array([float(coord[0]), float(coord[1]), float(coord[0]), float(coord[1]), 0])
  return data[:6]
  
def duplicate_removal(result):
  nms_thres = [0.4,0.4,0.2,0.2,0.2,0.4]
  for key in result.keys():
    roi_dets = [np.array([]).reshape(0,5),np.array([]).reshape(0,5),np.array([]).reshape(0,5),np.array([]).reshape(0,5),np.array([]).reshape(0,5),np.array([]).reshape(0,5)]
    #each roi
    for k in result[key].keys():
        for i in range(6):
            roi_dets[i] = np.concatenate((roi_dets[i], result[key][k][i]),0)
    for i in range(6):
        roi_dets[i] = nms(roi_dets[i], nms_thres[i])
    res = []
    for i in range(6):
      for m in range(roi_dets[i].shape[0]):
        w = roi_dets[i][m][2]-roi_dets[i][m][0] +1
        h = roi_dets[i][m][3]-roi_dets[i][m][1] +1
        res.append({'x': roi_dets[i][m][0], 'y':roi_dets[i][m][1],'w':w, 'h':h, 'p':roi_dets[i][m][4], 'class':CLASSES[i]})
    save_dir = './json_vis/'+key+'.json'
    f =open(save_dir, 'w')
    json.dump(res ,f)
    f.close()
def nms(dets, thresh):
    if dets.shape[0] == 0:
        return dets.reshape(dets.shape[0],5)
    if len(dets.shape)==1:
        dets= dets.reshape(1,dets.shape[0])
    if dets.shape[1]!=5:
        dets = dets.reshape(-1,5)
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]
    results=[]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return dets[keep]
# def compare(a, b, ratio=1, thres=0.1):
#   boxes_a = torch.from_numpy(a[:,:4])
#   boxes_b = torch.from_numpy(b[:,:4])
#   iou = intersect(boxes_a, boxes_b).data.numpy()
#   index = []
#   for i in range(iou.shape[0]):

#     if ((iou[i]>thres)*(a[i, 4]<b[:,4])).sum()>0:
#       #print(a[i,4])
#       # print((iou[i]>0.3))
#       # print((a[i, 4]<b[:,4]))
#       # print(((iou[i]>0.3)*(a[i, 4]<b[:,4])))
#       # print(((iou[i]>0.3)*(a[i, 4]<b[:,4])).sum())
#       continue
#     else:
#       index.append(i)
#   return a[index]

def compare_area(a, b, ratio=1, thres=0.8):
  boxes_a = torch.from_numpy(a[:,:4])
  boxes_b = torch.from_numpy(b[:,:4])
  ios, iou = ios_cp(boxes_a, boxes_b)
  ios = ios.data.numpy()
  iou = iou.data.numpy()
  index = []
  for i in range(iou.shape[0]):

    if ((ios[i]>thres)*(iou[i]<0.3)).sum()>0:
      #print(a[i,4])
      # print((iou[i]>0.3))
      # print((a[i, 4]<b[:,4]))
      # print(((iou[i]>0.3)*(a[i, 4]<b[:,4])))
      # print(((iou[i]>0.3)*(a[i, 4]<b[:,4])).sum())
      continue
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

def ios_cp(box_a, box_b):
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
  area_self = w_a* h_a 
  union = w_a* h_a + w_b* h_b - (inter[:, :, 0] * inter[:, :, 1])
  return (inter[:, :, 0] * inter[:, :, 1])/area_self, (inter[:, :, 0] * inter[:, :, 1])/union

def filter_boxinbox(pkl_path):
  pkl_file = open(pkl_path, 'rb')
  data = pickle.load(pkl_file)
  for i in range(len(data)):
    for j in range(len(data[i])):
      box_a = data[i][j]
      box_b = np.array([])
      flag = 0
      for k in range(len(data[i])):
        if k!=j:
          if flag == 0:
            box_b = data[i][k]
            flag = 1
          else:
            box_b = np.concatenate((box_b, data[i][k]), 0)
      data[i][j]=  np.concatenate((compare_area(box_a[box_a[:,4]>0.1], box_b[box_b[:,4]>0.1], ratio=1, thres=0.8), box_a[box_a[:,4]<=0.1]),0)
  save_path = pkl_path[:-4]+'backup.pkl'
  with open(save_path, 'wb') as f:
    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
  
# def re_nms(pkl_path):
#   pkl_file = open(pkl_path, 'rb')
#   data = pickle.load(pkl_file)
#   for i in range(len(data)):
#     for j in range(len(data[i])):
#       box_a = data[i][j]
#       box_b = np.array([])
#       flag = 0
#       for k in range(len(data[i])):
#         if k!=j:
#           if flag == 0:
#             box_b = data[i][k]
#             flag = 1
#           else:
#             box_b = np.concatenate((box_b, data[i][k]), 0)
#       data[i][j]= compare(box_a, box_b, ratio=1, thres=0.8)
#   save_path = pkl_path[:-4]+'backup.pkl'
#   with open(save_path, 'wb') as f:
#     pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

# def re_nms_neg(data):
#   # for i in range(len(data)):
#   keep = []
#   det = np.array(data)
#   if len(data)==0:
#     return []
#   x1 = det[:, 0]
#   y1 = det[:, 1]
#   x2 = det[:, 2]
#   y2 = det[:, 3]
#   areas = (x2 - x1 + 1) * (y2 - y1 + 1)
#   order = areas.argsort()[::-1]
#   while order.shape[0]>0:
#     max_index = order[0]
#     keep.append(max_index)
#     xx1 = np.maximum(x1[max_index], x1[order[1:]])
#     yy1 = np.maximum(y1[max_index], y1[order[1:]])
#     xx2 = np.minimum(x2[max_index], x2[order[1:]])
#     yy2 = np.minimum(y2[max_index], y2[order[1:]])

#     w = np.maximum(0.0, xx2 - xx1 + 1)
#     h = np.maximum(0.0, yy2 - yy1 + 1)
#     inter = w * h
#     ovr = inter / areas[order[1:]]
#     inds = np.where(ovr <= 0.5)[0]
#     order = order[inds + 1]
#   return det[keep].tolist()

# def delete_neg():
#   path = '/mnt/C/tianchi/train/VOC2007-/labels_train'
#   files = os.listdir(path)
#   sample_list = []
#   for file in files:
#     name = file[:-4]
#     if name.find('sample')!=-1:
#       sample_list.append(name)
#   print(len(sample_list))
#   for name in sample_list:
#     file_path = os.path.join(path, name+'.txt')
#     os.remove(file_path)
#     f = open(file_path, 'w')
#     f.close()

# def nms_pos_neg():
#   pkl_file = open('/home/hust3/mmdetection-master/result/epoch27.5.pkl', 'rb')
#   data1 = pickle.load(pkl_file)
#   pkl_file = open('/home/hust3/mmdetection-master/result/hard_neg_1108.pkl', 'rb')
#   data2 = pickle.load(pkl_file)
#   assert len(data1)==len(data2)
#   print('num_images:', len(data1))
#   result = []
#   for i in range(len(data1)):
#       if data1[i][0].shape[0]==0:
#           result.append(data1[i])
#           continue
#       else:
#           result.append([compare(data1[i][0], data2[i][1],1,0.5),])
#   with open('/home/hust3/mmdetection-master/result/combine.pkl', 'wb') as f:
#       pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)


def pkl_merge():
  pkl_file = open('/home/hust3/mmdetection-master/result/test0.pkl', 'rb')
  data1 = pickle.load(pkl_file)
  print(len(data1))
  pkl_file = open('/home/hust3/mmdetection-master/result/test1.pkl', 'rb')
  data2 = pickle.load(pkl_file)
  print(len(data2))

  data = data1 + data2
  print(len(data))
  with open('/home/hust3/mmdetection-master/result/test.pkl', 'wb') as f:
    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
def json_merge():
  path = './tianchi_json/'
  files = os.listdir(path)
  kfbnames = [s[:s.rfind('roi')] for s in files if s.find('.json')!=-1]
  kfbnames = list(set(kfbnames))
  for name in kfbnames:
    label_dict = []
    for file in files:
      if file[:file.rfind('roi')]==name:
        filepath = path+file
        print(filepath)
        f = open(filepath, encoding='utf-8')
        content = f.read()
        label_dict += json.loads(content)
        f.close()
        os.remove(path+file)
    save_dir = './tianchi_json/'+name+'.json'
    f =open(save_dir, 'w')
    json.dump(label_dict ,f)
    f.close()
  json_files = '/home/admin/jupyter/Data/test/'
  raw_json =  [s[:-5] for s in os.listdir(json_files) if s.find('.json')!=-1]
  tmp = []
  print(len(raw_json))
  for s in raw_json:
    if s not in kfbnames:
      save_dir = './tianchi_json/'+s+'.json'
      f =open(save_dir, 'w')
      json.dump(tmp,f)
      f.close()
  files = os.listdir(path)
  for file in files:
    if file.find('json')==-1:
      
      if os.path.isdir(path+file):
        os.rmdir(path+file)
      else:
        os.remove(path+file)

def recursive_glob():
    paths = '/home/admin/jupyter/mmdetection-master/tianchi_json/'
    paths1=[]
    filenames = []
    for base, dirs, files in os.walk(paths):
        for file in files:
            if file.endswith('.json') and file[0].isdigit() and len(file)<=9:
                paths1.append(os.path.join(base, file))
                filenames.append(file.split('.')[0])
    print(len(filenames))

if __name__=='__main__':
  #delete_neg()
  results_postprocess()
  #json_merge()
  #recursive_glob()
 
 
    




