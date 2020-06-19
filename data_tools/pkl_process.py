import os
import pickle
import torch
import json
import numpy as np
import cv2 as cv
color = [(0,0,255), (255,0,0), (0,255,0), (255,0,255),(0, 255,255),(255,255,0)]
CLASSES = ['ASC-H','ASC-US','HSIL','LSIL' ,'Candida', 'Trichomonas']

def filter_from_pkl():
    pkl_file = open('/home/admin/jupyter/mmdetection-master/result/res_512_12115.pkl', 'rb')
    data = pickle.load(pkl_file)
    for i in range(len(data)):
        for j in range(len(data[i])):
            print(data[i][j].shape)
            tmp = data[i][j]
            data[i][j] = tmp[(((tmp[:,3]-tmp[:,1])*(tmp[:,2]-tmp[:,0]))<102*102)]
            print(data[i][j].shape)
    save_path = '/home/admin/jupyter/mmdetection-master/result/res_512_12115_filter.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        
def file_num_sts():
    m  = dict()
    m['ASC-H'] = []
    m['ASC-US'] = []
    m['HSIL'] = []
    m['LSIL'] = []
    m['Candida'] = []
    m['Trichomonas'] = []
    m['NILM'] = []
    path ='/home/admin/jupyter/tianchi_data/train/VOC2007/ImageSets/Main/train.txt'
    f = open(path, 'r')
    files = f.readlines()
    f.close()
    print(len(files))
    sts = []
    tmp = []
    for file in files:
        tmp.append(file)
        file_path = os.path.join('/home/admin/jupyter/tianchi_data/train/labels_train', file.split('\n')[0]+'.txt')
        f = open(file_path, 'r')
        lines = f.readlines()
        f.close()
        for line in lines:
            line = line.split()
            for key in m.keys():
                if key==line[4]:
                    m[key].append(file)
    print('list:',len(tmp))
    for key in m.keys():
        m[key]=list(set(m[key]))
        print(key, ':', len(m[key]))
#     ca = m['Candida']+ m['HSIL'][:1000]+m['ASC-H'][:1000]+ m['LSIL'][:1000]+m['Trichomonas'][:1000]+m['NILM'][:1000]+m['ASC-US'][:1000]
#     print('ca:',len(ca))
# #     for i in range(1):
# #         tmp = tmp + m['HSIL'] + m['LSIL'] + m['Trichomonas']
# #     for i in range(3):
# #         tmp = tmp + m['Candida']
# #     print('new_list:', len(tmp))
#     f = open('/home/admin/jupyter/tianchi_data/train/VOC2007/ImageSets/Main/train_mini.txt', 'w')
#     for s in ca:
#         f.write(s)
#     f.close()
        
def results_postprocess_2048():
  path = './tianchi_json_sub/'
  files = os.listdir(path)
  for file in files:
    if os.path.isdir(path+file):
      os.rmdir(path+file)
    else:
      os.remove(path+file)
  pkl_file = open('/home/admin/jupyter/mmdetection-master/result/res_2048.pkl', 'rb')
  data = pickle.load(pkl_file)
  print(len(data))
  test_list_path = '/home/admin/jupyter/tianchi_data_2048/test/VOC2007_2048/ImageSets/Main/test.txt'
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
      dt_tmp = coord_transformer_2048(data[i], suffix, name)
      result[name][suffix] = dt_tmp
    else:
      dt_tmp = coord_transformer_2048(data[i], suffix, name)
      result[name][suffix] = dt_tmp
  return result

def results_postprocess_1024(path):
#   path = './tianchi_json_1024/'
#   files = os.listdir(path)
#   for file in files:
#     if os.path.isdir(path+file):
#       os.rmdir(path+file)
#     else:
#       os.remove(path+file)
  pkl_file = open(path, 'rb')
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
      dt_tmp = coord_transformer_1024(data[i], suffix, name)
      result[name][suffix] = dt_tmp
    else:
      dt_tmp = coord_transformer_1024(data[i], suffix, name)
      result[name][suffix] = dt_tmp
  return result
def coord_transformer_2048(data, suffix, name):
  index = suffix.split('_')
  i = int(index[0])
  j = int(index[1])
  for k in range(6):
    data[k] = data[k] + np.array([j*1024, i*1024, j*1024, i*1024, 0])
  f = open('/home/admin/jupyter/tianchi_data/test/ROI_coord/'+name+'.txt', 'r')
  lines = f.readlines()
  f.close()
  coord = lines[0].split()
  for k in range(6):
    data[k] += np.array([float(coord[0]), float(coord[1]), float(coord[0]), float(coord[1]), 0])
  return data[:6]
  
def coord_transformer_1024(data, suffix, name):
  index = suffix.split('_')
  i = int(index[0])
  j = int(index[1])
  for k in range(6):
    data[k] = data[k] + np.array([j*768, i*768, j*768, i*768, 0])
  f = open('/home/admin/jupyter/tianchi_data/test/ROI_coord/'+name+'.txt', 'r')
  lines = f.readlines()
  f.close()
  coord = lines[0].split()
  for k in range(6):
    data[k] += np.array([float(coord[0]), float(coord[1]), float(coord[0]), float(coord[1]), 0])
  return data[:6]

def duplicate_removal(result1,result2,ind):
  assert len(result1.keys())==len(result2.keys())
  print('num_roi:', len(result1.keys()))
  nms_thres = [0.2,0.2,0.2,0.2,0.2,0.2]
  for key in result1.keys():
    roi_dets = [np.array([]).reshape(0,5),np.array([]).reshape(0,5),np.array([]).reshape(0,5),np.array([]).reshape(0,5),np.array([]).reshape(0,5),np.array([]).reshape(0,5)]
    #each roi
    for k in result1[key].keys():
        for i in range(6):
            tmp = result1[key][k][i]
            tmp = tmp[((tmp[:,3]-tmp[:,1])*(tmp[:,2]-tmp[:,0]))>(290*290)]
            #if i in ind:
            roi_dets[i] = np.concatenate((roi_dets[i], tmp),0)
    for k in result2[key].keys():
        for i in range(6):
            #if i not in ind:
            roi_dets[i] = np.concatenate((roi_dets[i], result2[key][k][i]),0)
    for i in range(6):
        roi_dets[i] = nms(roi_dets[i], nms_thres[i])
    res = []
    for i in range(6):
      for m in range(roi_dets[i].shape[0]):
        w = roi_dets[i][m][2]-roi_dets[i][m][0] +1
        h = roi_dets[i][m][3]-roi_dets[i][m][1] +1
        res.append({'x': roi_dets[i][m][0], 'y':roi_dets[i][m][1],'w':w, 'h':h, 'p':roi_dets[i][m][4], 'class':CLASSES[i]})
    save_dir = './tianchi_json_sub/'+key+'.json'
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

def json_merge():
  path = './tianchi_json_sub/'
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
    save_dir = './tianchi_json_sub/'+name+'.json'
    f =open(save_dir, 'w')
    json.dump(label_dict ,f)
    f.close()
  json_files = '/home/admin/jupyter/Data/test/'
  raw_json =  [s[:-5] for s in os.listdir(json_files) if s.find('.json')!=-1]
  tmp = []
  print(len(raw_json))
  for s in raw_json:
    if s not in kfbnames:
      save_dir = './tianchi_json_sub/'+s+'.json'
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
    paths = '/home/admin/jupyter/mmdetection-master/tianchi_json_sub/'
    paths1=[]
    filenames = []
    for base, dirs, files in os.walk(paths):
        for file in files:
            if file.endswith('.json') and file[0].isdigit() and len(file)<=9:
                paths1.append(os.path.join(base, file))
                filenames.append(file.split('.')[0])
    print(len(filenames))
    
def json_analysis():
    path =  '/home/admin/jupyter/mmdetection-master/tianchi_json/'
    files = os.listdir(path)
    tmp = []
    for file in files:
        m  = dict()
        m['ASC-H'] = 0
        m['ASC-US'] = 0
        m['HSIL'] = 0
        m['LSIL'] = 0
        m['Candida'] = 0
        m['Trichomonas'] = 0
        m['NILM'] = []
        f = open(path+file, 'r')
        data = json.load(f)
        for s in data:
            if s['p']<0.25:
                continue
            m[s['class']]= m[s['class']]+ s['p']
        print(file, 'AAHL:',m['ASC-H']+m['ASC-US']+m['HSIL']+m['LSIL'],'Candida:',m['Candida'],'Trichomonas:',m['Trichomonas'])
        #print(file, 'ASC-H:',m['ASC-H'],'ASC-US:',m['ASC-US'],'HSIL:',m['HSIL'],'LSIL:',m['LSIL'],'Candida:',m['Candida'],'Trichomonas:',m['Trichomonas'])
        tmp.append([file, m['ASC-H']+m['ASC-US']+m['HSIL']+m['LSIL'],m['Candida'],m['Trichomonas']])
    th = []
    aa = []
    ca = []
    for s in tmp:
        if s[3]>2*(s[1]+s[2]):
            th.append(s)
            ##print(s)
        if s[1]>2*(s[3]+s[2]):
            aa.append(s)
            #print(s)
        if s[2]>2*(s[3]+s[1]):
            ca.append(s)
            print(s)
    print(len(th))
    print(len(aa))
    print(len(ca))
if __name__ == '__main__':
    #filter_from_pkl()
#     results_postprocess()
#     json_merge()
#     recursive_glob()
#     result1 = results_postprocess_512()
#     result2 = results_postprocess_1024()
#     duplicate_removal(result1,result2)
#     json_merge()
#     recursive_glob()
#     pkl_file = open('/home/admin/jupyter/mmdetection-master/result/res_multi_ohem_512_67854.pkl', 'rb')
#     #pkl_file = open('/home/admin/jupyter/mmdetection-master/result/res_multi_ohem_1024_16944_.pkl', 'rb')
#     data = pickle.load(pkl_file)
    
#     count = 0 
#     for i in range(len(data)):
#         for j in range(6):
#             tmp = data[i][j]
#             count += ((tmp[:,3]-tmp[:,1])*(tmp[:,2]-tmp[:,0])>200*200).sum()
#     print(count)#   


#     path = './tianchi_json_sub/'
#     files = os.listdir(path)
#     for file in files:
#         if os.path.isdir(path+file):
#             os.rmdir(path+file)
#         else:
#             os.remove(path+file)

#     pth1 = '/home/admin/jupyter/mmdetection-master/result/res_multi_1024_16944.pkl'
#     pth2 = '/home/admin/jupyter/mmdetection-master/result/res_multi_ba_1024_16944.pkl'
#     res1 = results_postprocess_2048()
#     res2 = results_postprocess_1024(pth1)
#     duplicate_removal(res1,res2,[0,1,5])
#     json_merge()
#     recursive_glob()
    json_analysis()
                                                  
    
      
 

   
           
    