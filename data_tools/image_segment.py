import kfbReader
import cv2 as cv
import os
import numpy as np

def read_name():
    root='/home/admin/jupyter/'
    train_image_path= root +'tianchi_data/train/images_train'
    image_names = os.listdir(train_image_path)
    f = open(root+'tianchi_data/train/VOC2007/ImageSets/Main/test_add.txt','w') 
    for image_name in image_names:
#         if image_name[:4]=='8484' or image_name[:8]=='2393roi2':
        if image_name[:8]=='2393roi1':
            print(image_name[:-4])
            f.write(image_name[:-4]+'\n')
    f.close()
    
    
def image_segment():
#   path = '/home/admin/jupyter/update/update_train'
#   files = [s[:-5]  for s in os.listdir(path) if s.find('.json')!=-1]
#   print(files)
  root='/home/admin/jupyter/'
  image_path = root +'tianchi_data/train/ROI_images'
  label_path = root +'tianchi_data/train/ROI_labels'
  train_image_path= root +'tianchi_data/train/images_train'
  train_label_path= root +'tianchi_data/train/labels_train'
  empty_image_path = root +'tianchi_data/train/ROI_empty_images'
  BLACK = [0, 0, 0]
  image_names = os.listdir(image_path)
  # def record_pos_labels(window,)
  lost_object_count = 0
  for image_name in image_names:
    ##image
#     if image_name!='2393roi2.jpg':
# #         print('continue')
# #         print(image_name)
#         continue
    print(image_name)
    img = cv.imread(os.path.join(image_path, image_name))
    #print(img.shape)
    H, W = (int(img.shape[0] / 512) + 1)*512, (int(img.shape[1] / 512) + 1)*512
    bottom = H - img.shape[0]
    right = W - img.shape[1]
    constant = cv.copyMakeBorder(img, 0, bottom, 0, right, cv.BORDER_CONSTANT, value=BLACK)
    idx_window = empty_count = 1
    #sliding window
    m, n = int(H/512), int(W/512)
    image_label_path = label_path + '/' + image_name[:-4] + '.txt'
    f = open(image_label_path,'r')
    lines = f.readlines()
    f.close()
    num_pos = len(lines)
    flag = np.zeros(num_pos)
    for i in range(m-1):
      for j in range(n-1):
        window = constant[i*512:(i+2)*512, j*512:(j+2)*512]
        ##pos
        tmp = []
        for idx_p in range(num_pos):
          pos = lines[idx_p].split(" ")
          c_x, c_y, w, h = float(pos[0]), float(pos[1]), int(pos[2]), int(pos[3])
          X0, X1, Y0, Y1 = j*512, (j+2)*512, i*512, (i+2)*512
          xmin = max(c_x - w / 2, X0)
          ymin = max(c_y - h / 2, Y0)
          xmax = min(c_x + w / 2, X1)
          ymax = min(c_y + h / 2, Y1)
          w_ = np.maximum(xmax-xmin+1, 0)
          h_ = np.maximum(ymax-ymin+1, 0)

          #if ((w_*h_)/(w*h) < 0.1 and w*h<1000000) or ((w_*h_)/(w*h) == 0.0 and w*h>1000000):
          if (w_*h_)/(w*h) < 0.5:
            continue
          else:
            if w_/h_>5 or h_/w_>5:
              print('error', w_, h_, w, h)
              print((w_*h_)/(w*h))
            #cv.imwrite(train_image_path + '/' + image_name[:-4] + '_' + str(idx_window) + '.jpg', window)
            #save_label (xmin+xmax)/2 (ymin+ymax)/2 w_ h_ pos
            #print(idx_window)
            flag[idx_p]=1
            tmp.append(str((xmin+xmax)/2-X0)+' '+str((ymin+ymax)/2-Y0)+' '+str(w_)+' '+str(h_)+ ' '+ pos[4].split('\n')[0] +'\n')

        if tmp:
          txtName = train_label_path + '/' + image_name[:-4] + '_' + str(idx_window) + '.txt'
          imageName = train_image_path + '/' + image_name[:-4] + '_' + str(idx_window) + '.jpg'
          f = open(txtName, 'w')
          for i_tmp in range(len(tmp)):
            f.write(tmp[i_tmp])
          f.close()
#           print('---')
          cv.imwrite(imageName, window)
          idx_window+=1
#         if not tmp:
#           imageName = empty_image_path + '/' + image_name[:-4] + '_' + str(empty_count) + '.jpg'
#           cv.imwrite(imageName, window)
#           empty_count+=1
            
    lost_object_count += flag.shape[0] - flag.sum()
    print(lost_object_count)
    # print(constant.shape)
#     break

def test_image_segment():
    test_image_path='/home/admin/jupyter/tianchi_data/test/ROI_images'
    BLACK = [0, 0, 0]
    image_names = os.listdir(test_image_path)
    print(image_names)
    name_list = []
    for image_name in image_names:
    ##image
        print(image_name)
        img_path = os.path.join(test_image_path, image_name)
        img = cv.imread(img_path)
        height = img.shape[0]
        width = img.shape[1]
        print(height, width)
        H, W = (int((height-256) / 768) + 1)*768+256, (int((width -256)/ 768) + 1)*768+256
        bottom = H - height
        right = W - width
        m, n = int((H-256)/768), int((W-256)/768)
        print(m, n)
        count = 0
        for i in range(m):
            for j in range(n):
                count+=1
                print(count)
                roi_height = 1024
                roi_width = 1024
                if i == m- 1:
                    roi_height = roi_height-bottom
                if j == n - 1:
                    roi_width = roi_width -right
                window = img[i*768:i*768+roi_height,j*768:j*768+roi_width]
                if i == m- 1:
                    window = cv.copyMakeBorder(window, 0, bottom, 0, 0, cv.BORDER_CONSTANT, value=BLACK)
                if j == n -1:
                    window = cv.copyMakeBorder(window, 0, 0, 0, right, cv.BORDER_CONSTANT, value=BLACK)
                assert window.shape[0]==1024 and window.shape[1]==1024, 'error'
                imageName = '/home/admin/jupyter/tianchi_data/test/test_seg_img/' + image_name[:-4] + 'seg_' + str(i)+'_'+str(j) + '.jpg'
                name_list.append(image_name[:-4] + 'seg_' + str(i)+'_'+str(j))
                cv.imwrite(imageName, window)
                label_path = '/home/admin/jupyter/tianchi_data/test/test_seg_label/' + image_name[:-4] + 'seg_' + str(i)+'_'+str(j) + '.txt'
                f = open(label_path, 'w')
                f.close()
    f = open('/home/admin/jupyter/tianchi_data/test/VOC2007/ImageSets/Main/test.txt', 'w')
    for s in name_list:
        f.write(s+'\n')
    f.close()


import random
def train_test_segment():
  path = '/mnt/B/tianchi/labels_train'
  files = os.listdir(path)
  tmp =[]
  for file in files:
    tmp.append(file[:-4])
  random.shuffle(tmp)
  f = open('/mnt/B/tianchi/train.txt', 'w')
  for i in range(int(len(tmp)*0.75)):
    f.write(tmp[i]+'\n')
  f.close()
  f = open('/mnt/B/tianchi/test.txt', 'w')
  for i in range(int(len(tmp) * 0.75),len(tmp)):
    f.write(tmp[i] + '\n')
  f.close()
def two_txt_combine():
  path1 = '/mnt/C/tianchi/train/VOC2007-/ImageSets/Main/test1.txt'
  path2 = '/mnt/C/tianchi/neg_samples/neg/neg_test.txt'
  tmp = []
  f = open(path1, 'r')
  for line in f.readlines():
    line = line.split()
    tmp.append(line[0])
  f.close()
  f = open(path2, 'r')
  for line in f.readlines():
    line = line.split()
    tmp.append(line[0])
  f.close()
  save_path = '/mnt/C/tianchi/train/VOC2007-/ImageSets/Main/test1.txt'
  f = open(save_path, 'w')
  for s in tmp:
    if s == tmp[-1]:
      f.write(s)
    else:
      f.write(s+'\n')
  f.close()

def annno_txt_combine():
  path1 = '/home/xiezihao/mmdetection-master/data/tianchi/VOC2007/ImageSets/Main/train.txt'
  path2 = '/home/xiezihao/mmdetection-master/data/tianchi/neg_samples/Annotations'
  tmp = []
  f = open(path1, 'r')
  for line in f.readlines():
    line = line.split()
    tmp.append(line[0])
  f.close()
  files = os.listdir(path2)
  for file in files:
    tmp.append(file[:-4])
  save_path = '/home/xiezihao/mmdetection-master/data/tianchi/VOC2007/ImageSets/Main/test1.txt'
  f = open(save_path, 'w')
  for s in tmp:
    if s == tmp[-1]:
      f.write(s)
    else:
      f.write(s+'\n')
  f.close()

def object_sts():
    path ='/home/admin/jupyter/tianchi_data/train/labels_train'
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
          
            if line[4]=='Candida':
                area = np.sqrt(float(line[2])*float(line[3]))
                #area = float(line[2])/float(line[3])
                sts.append(area) 
    print(len(sts))
    min_ = min(sts)
    max_ = max(sts)
    print(min_, max_)
    num = 40
    count = [0 for i in range(num)]
    for a in sts:
        for i in range(num):
            if a>min_+i*(max_-min_)/num and a < min_+(i+1)*(max_-min_)/num:
                count[i]+=1
    for i in range(num):
        print(min_+i*(max_-min_)/num, '~', min_+(i+1)*(max_-min_)/num, ':', count[i], '\n')
        
        

def object_num_sts():
    m  = dict()
    m['ASC-H'] = 0
    m['ASC-US'] = 0
    m['HSIL'] = 0
    m['LSIL'] = 0
    m['Candida'] = 0
    m['Trichomonas'] = 0
    m['NILM'] = 0
    path ='/home/admin/jupyter/tianchi_data/train/VOC2007/ImageSets/Main/HSIL.txt'
    f = open(path , 'r')
    files = f.readlines()
    print(len(files))
    sts = []
    for file in files:
        file_path = os.path.join('/home/admin/jupyter/tianchi_data/train/labels_train', file[:-1]+'.txt')
        f = open(file_path, 'r')
        lines = f.readlines()
        f.close()
        for line in lines:
            line = line.split()
            for key in m.keys():
                if key==line[4]:
                    m[key]+=1
    for key in m.keys():
        print(key, ':', m[key])

def object_ratio_sts():
    m  = dict()
    m['ASC-H'] = 0
    m['ASC-US'] = 0
    m['HSIL'] = 0
    m['LSIL'] = 0
    m['Candida'] = 0
    m['Trichomonas'] = 0
    m['NILM'] = 0
    path ='/home/admin/jupyter/tianchi_data/train/VOC2007/ImageSets/Main/train.txt'
    f = open(path , 'r')
    files = f.readlines()
    print(len(files))
    sts = []
    for file in files:
        file_path = os.path.join('/home/admin/jupyter/tianchi_data/train/labels_train', file[:-1]+'.txt')
        f = open(file_path, 'r')
        lines = f.readlines()
        f.close()
        for line in lines:
            line = line.split()
            for key in m.keys():
                if key==line[4]:
                    m[key]+=1/len(lines)
    for key in m.keys():
        print(key, ':', m[key])
        
def object_ex_sts():
    m  = dict()
    m['ASC-H'] = 0
    m['ASC-US'] = 0
    m['HSIL'] = 0
    m['LSIL'] = 0
    m['Candida'] = 0
    m['Trichomonas'] = 0
    m['NILM'] = 0
    path ='/home/admin/jupyter/tianchi_data/train/VOC2007/ImageSets/Main/train.txt'
    f = open(path , 'r')
    files = f.readlines()
    print(len(files))
    sts = []
    for file in files:
        file_path = os.path.join('/home/admin/jupyter/tianchi_data/train/labels_train', file[:-1]+'.txt')
        f = open(file_path, 'r')
        lines = f.readlines()
        f.close()
        tmp = []
        for line in lines:
            line = line.split()
            tmp.append(line[4])
        if 'HSIL' in tmp:
            sts.append(file)
    f = open('/home/admin/jupyter/tianchi_data/train/VOC2007/ImageSets/Main/HSIL.txt', 'w')
    for s in sts:
        f.write(s)
    f.close()
        
def object_cat_sts():
    m  = dict()
    m['1'] = 0
    m['2'] = 0
    m['3'] = 0
    m['4'] = 0
    m['5'] = 0
    m['6'] = 0
    m['7'] = 0
    path ='/home/admin/jupyter/tianchi_data/train/labels_train'
    files = os.listdir(path)
    print(len(files))
    sts = []
    for file in files:
        file_path = os.path.join(path, file)
        f = open(file_path, 'r')
        lines = f.readlines()
        f.close()
        tmp = []
        for line in lines:
            line = line.split()
            tmp.append(line[4])
        for key in m.keys():
            if key == str(len(set(tmp))):
                m[key]+=1
    for key in m.keys():
        print(key, ':', m[key])
            
def train_test_seg():
  path ='/home/admin/jupyter/tianchi_data/train/labels_train'
  lines = os.listdir(path)
  roi = []
  for line in lines:
    a = line.split('_')[0]
    roi.append(a)
  print(len(roi))
  roi_re = []
  for s in roi:
    if s not in roi_re:
      roi_re.append(s)
  print(len(roi_re))
  random.shuffle(roi_re)
  test_roi = roi_re[:360]
  print(len(test_roi))
  train = []
  test = []
  for line in lines:
    if line.split('_')[0] in test_roi:
      test.append(line)
    else:
      train.append(line)
  print(len(test))
  print(len(train))
  path_train ='/home/admin/jupyter/tianchi_data/train/VOC2007/ImageSets/Main/train.txt'
  path_test ='/home/admin/jupyter/tianchi_data/train/VOC2007/ImageSets/Main/test.txt'
  f = open(path_train,'w')
  for s in train:
    f.write(s[:-4]+'\n')
  f.close()
  f = open(path_test, 'w')
  for s in test:
    f.write(s[:-4]+'\n')
  f.close()          
            
if __name__ == '__main__':
  #annno_txt_combine()
  image_segment()
  #two_txt_combine()
  #object_sts()
  #train_test_seg()
  test_image_segment()
  #object_num_sts()
#   object_cat_sts()
#     object_ex_sts()
#     object_num_sts()
  
  

    
