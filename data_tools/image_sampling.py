import kfbReader
import cv2 as cv
import os
import random
import math

import shutil
def neg_sample():
  path = '/mnt/C/tianchi/neg_samples/test/test.txt'
  f = open(path, 'r')
  tmp = []
  for line in f.readlines():
    line = line.split()
    tmp.append(line[0][0:line[0].rfind('_sample')])
  print(tmp)
  neg_dir = ['neg_0', 'neg_1', 'neg_2', 'neg_3', 'neg_4', 'neg_5']
  data_path = '/mnt/C/tianchi/raw_data/'
  for dir in neg_dir:
    dir_path = data_path + dir
    image_names = os.listdir(dir_path)
    for image_name in image_names:
      print(image_name[:-4])
      if image_name[:-4] in tmp:
        continue
      image_path = os.path.join(dir_path, image_name)
      scale_kfb = 20
      read = kfbReader.reader()
      kfbReader.reader.ReadInfo(read, image_path, scale_kfb, True)
      kfbReader.reader.setReadScale(read, scale=20)
      height = read.getHeight()
      width = read.getWidth()
      for i in range(30):
        x_tmp = random.randint(0, width-1024)
        y_tmp = random.randint(0, height-1024)
        roi = read.ReadRoi(x_tmp, y_tmp, 1024, 1024, 20)
        img_save_name = image_name[:-4]+'_sample_'+str(i)+'.jpg'
        label_save_name = image_name[:-4] + '_sample_' + str(i) + '.txt'
        cv.imwrite('/mnt/C/tianchi/neg_samples/neg_sample_7500_images/'+img_save_name , roi)
        f = open('/mnt/C/tianchi/neg_samples/neg_sample_7500_labels/'+label_save_name, 'w')
        f.close()
def neg_image_segment():
    dir_path = '/home/admin/jupyter/Data/train'
    json_files = [s[:-5] for s in os.listdir(dir_path) if s.find('.json')!=-1]
    kfb_files = [s[:-4] for s in os.listdir(dir_path) if s.find('.kfb')!=-1]
    image_names =[]
    for file in  kfb_files:
        if file not in json_files:
            image_names.append(file+'.kfb')
    print(len(image_names))
    assert len(image_names)==250, 'error'
    for image_name in image_names:
        print(image_name[:-4])
        #if image_name[:-4] not in tmp:
         # continue
        image_path = os.path.join(dir_path, image_name)
        scale_kfb = 20
        read = kfbReader.reader()
        kfbReader.reader.ReadInfo(read, image_path, scale_kfb, True)
        kfbReader.reader.setReadScale(read, scale=20)
        height = read.getHeight()
        width = read.getWidth()
        print('height:', height)
        print('width:', width)
        n = int(height/1024.0)
        m = int(width/1024.0)
        for i in range(m):
            for j in range(n):
                x_tmp = i*1024
                y_tmp = j*1024
                roi = read.ReadRoi(x_tmp, y_tmp, 1024, 1024, 20)
                img_save_name = image_name[:-4]+'_segment_'+str(i*n+j)+'.jpg'
                label_save_name = image_name[:-4] + '_segment_' + str(i*n+j) + '.txt'
                cv.imwrite('/home/admin/jupyter/tianchi_data/train/neg_segment_images/'+img_save_name , roi)
                f = open('/home/admin/jupyter/tianchi_data/train/neg_segment_labels/'+label_save_name, 'w')
                f.close()
def build_test():
  neg_dir = ['neg_0', 'neg_1', 'neg_2', 'neg_3', 'neg_4', 'neg_5']
  data_path = '/mnt/D/xiezihao/tianchi/'
  image_names = []
  for dir in neg_dir:
    dir_path = data_path + dir
    image_names += os.listdir(dir_path)
  image_list = []
  for s in image_names:
    image_list.append(s[:-4])
  random.shuffle(image_list)
  test_name = image_list[0:50]
  f = open('/mnt/B/tianchi/neg_samples/test.txt', 'w')
  for s in test_name:
    for i in range(50):
      f.write(s+'_sample_'+str(i) + '\n')
      path = '/mnt/B/tianchi/neg_samples/labels/'+s+'_sample_'+str(i)+'.txt'
      if not os.path.exists(path):
        print('error')
  f.close()

def file_remove():
  path = '/mnt/B/tianchi/neg_samples/test.txt'
  f = open(path, 'r')
  tmp = []
  for line in f.readlines():
    line = line.split()
    tmp.append(line[0])
  image_path = '/mnt/B/tianchi/neg_samples/images'
  label_path = '/mnt/B/tianchi/neg_samples/labels'
  files = os.listdir(label_path)
  print(tmp)
  # for file in files:
  #   if file[:-4] in tmp:
  #     os.remove(os.path.join(label_path, file))
  for s in tmp:
    shutil.copy('/mnt/B/tianchi/neg_samples/images/'+s+'.jpg', '/mnt/B/tianchi/neg_samples/images_for_test/'+s+'.jpg')
    os.remove('/mnt/B/tianchi/neg_samples/images/'+s+'.jpg')
    shutil.copy('/mnt/B/tianchi/neg_samples/labels/' + s + '.txt',
                '/mnt/B/tianchi/neg_samples/labels_for_test/' + s + '.txt')
    os.remove('/mnt/B/tianchi/neg_samples/labels/' + s + '.txt')

  # anno_path = '/mnt/B/tianchi/neg_samples/Annotations'
  # files = os.listdir(anno_path)
  # for file in files:
  #   if file[:-4] not in tmp:
  #     print(file[:-4])
  #     os.remove(os.path.join(anno_path, file))

def delete_something():
  main_train = '/mnt/C/tianchi/train/VOC2007-/ImageSets/Main/train.txt'
  main_test = '/mnt/C/tianchi/train/VOC2007-/ImageSets/Main/test.txt'
  train_path = '/mnt/C/tianchi/images_train/'
  annotation_path = '/mnt/C/tianchi/Annotations/'
  train_list = [name[:-4] for name in os.listdir(train_path)]
  remain_list=[]
  f = open(main_train)
  lines = f.readlines()
  f.close()
  for line in lines:
    remain_list.append(line.split()[-1])
  f = open(main_test)
  lines = f.readlines()
  f.close()
  for line in lines:
    remain_list.append(line.split()[-1])
  print('train_list len:',len(train_list))
  for l in train_list:
    if l not in remain_list:
      os.remove(train_path+l+'.jpg')
      os.remove(annotation_path+l+'.xml')
  print('remian list len:', len(remain_list))
import shutil
def data_combine():
  src_dir = '/mnt/C/empty_image/JPEGImages'
  drc_dir = '/mnt/C/empty_image/image_select'
  list_path = '/mnt/C/empty_image/empty_select.txt'
  f = open(list_path, 'r')
  lines = f.readlines()
  for line in lines:
    name = line.split('\n')[0]
    shutil.copy(os.path.join(src_dir, name+'.jpg'), os.path.join(drc_dir, name+'.jpg'))

def train_txt_generate():
  path = '/home/admin/jupyter/tianchi_data/train/labels_train'
  files = os.listdir(path)
  train_name = []
  for file in files:
    train_name.append(file[:-4])
  print(len(train_name))
  f = open('/home/admin/jupyter/tianchi_data/train/VOC2007/ImageSets/Main/train.txt', 'w')
  for s in train_name:
    f.write(s + '\n')
  f.close()
def train_test_segment():
  test_txt = '/mnt/C/tianchi/train/VOC2007-/ImageSets/Main/test.txt'
  f = open(test_txt, 'r')
  lines = f.readlines() 
  f.close()
  test_name = []
  for line in lines:
    line = line.split('\n')[0]
    if line.find('sample')!=-1:
      continue
    test_name.append(line[:line.rfind('roi')+4])
  print(len(set(test_name)))
  neg_txt = '/mnt/C/empty_image/empty_select.txt'
  f = open(neg_txt, 'r')
  lines = f.readlines() 
  f.close()
  neg_test = []
  neg_train = []
  for line in lines:
    line = line.split('\n')[0]
    if line[:line.rfind('roi')+4] in test_name:
      neg_test.append(line)
    else:
      neg_train.append(line)
  f = open('/mnt/C/empty_image/empty_train.txt', 'w')
  for s in neg_train:
    f.write(s+'\n')
  f.close()
  f = open('/mnt/C/empty_image/empty_test.txt', 'w')
  for s in neg_test:
    f.write(s+'\n')
  f.close()
   
if __name__ == '__main__':
    neg_image_segment()
  #train_test_segment()
  #data_combine()
#   test_txt = '/mnt/C/tianchi/train/VOC2007-/ImageSets/Main/test.txt'
#   f = open(test_txt, 'r')
#   lines = f.readlines() 
#   f.close()
#   test_name = []
#   for line in lines:
#     line = line.split('\n')[0]
#     if line.find('sample')!=-1:
#       continue
#     test_name.append(line)
#   f = open('/mnt/C/tianchi/train/VOC2007-/ImageSets/Main/test1.txt', 'w')
#   for s in test_name:
#     f.write(s+'\n')
#   f.close()
  

  
  
    
