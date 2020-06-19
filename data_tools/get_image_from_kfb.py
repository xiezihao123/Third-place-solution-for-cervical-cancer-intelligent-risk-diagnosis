import kfbReader
import cv2 as cv
import json
import os
def test_process():
    path = '/home/admin/jupyter/Data/test'
    filename = [s[:-4]  for s in os.listdir(path) if s.find('.kfb')!=-1]
    print(filename)
    print(len(filename))
    for name in filename:
        filepath = os.path.join(path, name + '.json')
        image_path = os.path.join(path, name + '.kfb')
        f = open(filepath, encoding='utf-8')
        content = f.read()
        label_dict = json.loads(content)
        f.close()
        count = 0
        for dt in label_dict:
            if dt['class'] != 'roi':
                 print(dt['class'])
            if dt['class'] == 'roi':
                count+=1
                save_path = os.path.join('/home/admin/jupyter/tianchi_data/test', 'ROI_images/'+name+'roi'+str(count)+'.jpg')
                scale_kfb = 20
                read = kfbReader.reader()
                kfbReader.reader.ReadInfo(read, image_path, scale_kfb, True)
                kfbReader.reader.setReadScale(read, scale=20)
                roi = read.ReadRoi(dt['x'], dt['y'], dt['w'], dt['h'], 20)
                cv.imwrite(save_path, roi)
                f = open(os.path.join('/home/admin/jupyter/tianchi_data/test', 'ROI_coord/'+name+'roi'+str(count)+'.txt'),'w')
                f.write(str(dt['x'])+' '+str(dt['y'])+' '+str(dt['w'])+' '+str(dt['h'])+'\n')
                f.close()
def pos_process():
    path = '/home/admin/jupyter/Data/train'
    filename = [s[:-5]  for s in os.listdir(path) if s.find('.json')!=-1]
    for name in filename:
        print(name)
        filepath = os.path.join(path, name+'.json')
        image_path = os.path.join(path, name+'.kfb')
        f = open(filepath, encoding='utf-8')
        content = f.read()
        label_dict = json.loads(content)
        f.close()
        count = 0
        for dt in label_dict:
            if dt['class'] == 'roi':
                count+=1
                save_path = os.path.join( '/home/admin/jupyter/tianchi_data/train', 'ROI_images/'+name+'roi'+str(count)+'.jpg')
                scale_kfb = 20
                read = kfbReader.reader()
                kfbReader.reader.ReadInfo(read, image_path, scale_kfb, True)
                kfbReader.reader.setReadScale(read, scale=20)
                roi = read.ReadRoi(dt['x'], dt['y'], dt['w'], dt['h'], 20)
                cv.imwrite(save_path, roi)
                tmp = []
                for dt1 in label_dict:
                    if dt1['class'] == 'roi':
                        continue
                    if dt1['x']>=dt['x'] and dt1['y']>=dt['y'] and (dt1['x']+dt1['w'])<=(dt['x']+dt['w']) and (dt1['y']+dt1['h'])<=(dt['y']+dt['h']):
                        tmp.append(str(dt1['x']-dt['x'] + dt1['w']/2.0)+' '+str(dt1['y']-dt['y'] + dt1['h']/2.0)+' '+
                                str(dt1['w'])+' '+str(dt1['h'])+' '+dt1['class'])
                f = open(os.path.join('/home/admin/jupyter/tianchi_data/train', 'ROI_labels/'+name+'roi'+str(count)+'.txt'),'w')
                for s in tmp:
                    if s!=tmp[-1]:
                        f.write(s+'\n')
                    else:
                        f.write(s)
                f.close()
        

def neg_process():
  pass

def vis():
  root='/home/admin/jupyter/'
  img_path  = root +'tianchi_data/train/images_train'
  labeldir_path = root +'tianchi_data/train/labels_train'
  #list_path = '/mnt/B/tianchi/VOC2007/ImageSets/Main/test.txt'
  #f = open(list_path, 'r')
  #lines = f.readlines()
  image_names = os.listdir(labeldir_path)
  
  for name in image_names:
    image_path = os.path.join(img_path, name[:-4]+'.jpg')
    img = cv.imread(image_path)
    label_path = os.path.join(labeldir_path, name)
    f = open(label_path, 'r')
    for line in f.readlines():
      print(line)
      coord = line.split()
      x1 = int(float(coord[0])-float(coord[2])/2)
      y1 = int(float(coord[1])-float(coord[3])/2)
      x2 = int(float(coord[0])+float(coord[2])/2)
      y2 = int(float(coord[1])+float(coord[3])/2)
      img = cv.rectangle(img,(x1, y1 ),(x2,y2), (255,0,0), 4)
    cv.imwrite('/home/admin/jupyter/tianchi_data/train/vis/'+name[:-4]+'.jpg',img)
import random
def train_test_segment():
  path ='/mnt/B/tianchi/RoI/labels'
  files = os.listdir(path)
  sample_name = []
  for file in files:
    sample_name.append(file[:-5])

  random.shuffle(sample_name)
  #train_name = sample_name[0:450]
  test_name = sample_name[450:]
  print(test_name)
  train_path = '/mnt/B/tianchi/images_train'
  sample = []
  files = os.listdir(train_path)
  for file in files:
    sample.append(file[:-4])
  train = []
  test = []
  print(test_name)
  for s in sample:
    if s[0:s.rfind('roi')] in test_name:
      test.append(s)
    else:
      train.append(s)
  f = open('/mnt/B/tianchi/train.txt', 'w')
  for s in train:
    f.write(s+'\n')
  f.close()
  f = open('/mnt/B/tianchi/test.txt', 'w')
  for s in test:
    f.write(s + '\n')
  f.close()



def train_test_segment_by_roi():
  path = '/mnt/B/tianchi/RoI/labelTxt'
  files = os.listdir(path)
  sample_name = []
  for file in files:
    sample_name.append(file[:-4])

  random.shuffle(sample_name)
  # train_name = sample_name[0:450]
  test_name = sample_name[0:121]
  print(test_name)
  train_path = '/mnt/B/tianchi/images_train'
  sample = []
  files = os.listdir(train_path)
  for file in files:
    sample.append(file[:-4])
  train = []
  test = []
  print(test_name)
  for s in sample:
    if s[0:s.rfind('roi')+4] in test_name:
      test.append(s)
    else:
      train.append(s)
  f = open('/mnt/B/tianchi/train.txt', 'w')
  for s in train:
    f.write(s + '\n')
  f.close()
  f = open('/mnt/B/tianchi/test.txt', 'w')
  for s in test:
    f.write(s + '\n')
  f.close()
def object_count():
  train = '/home/admin/jupyter/tianchi_data/train/VOC2007/ImageSets/Main/train.txt'
  test = '/home/admin/jupyter/tianchi_data/train/VOC2007/ImageSets/Main/test.txt'
  #for train
  f = open(train, 'r')
  lines = f.readlines()
  f.close()
  train_name = []
  for s in lines:
    if s.find('empty')!=-1:
      continue
    train_name.append(s.split('\n')[0])
  small = 0
  medium = 0
  large = 0
  for name in train_name:
    path = '/home/admin/jupyter/tianchi_data/train/labels_train/'+name+'.txt'
    f = open(path, 'r')
    lines = f.readlines()
    f.close()
    for s in lines:
      s = s.split()
      area = float(s[2])*float(s[3])
      if area<32*32:
        small+=1
      elif area>32*32 and area<96*96:
        medium+=1
      else:
        large+=1
  print('small', small)
  print('medium', medium)
  print('large', large)

  #for test
  f = open(test, 'r')
  lines = f.readlines()
  f.close()
  train_name = []
  for s in lines:
    if s.find('empty')!=-1:
      continue
    train_name.append(s.split('\n')[0])
  small = 0
  medium = 0
  large = 0
  for name in train_name:
    path = '/home/admin/jupyter/tianchi_data/train/labels_train/' + name + '.txt'
    f = open(path, 'r')
    lines = f.readlines()
    f.close()
    for s in lines:
      s = s.split()
      area = float(s[2]) * float(s[3])
      if area < 32*32:
        small += 1
      elif area > 32*32 and area <96*96:
        medium += 1
      else:
        large += 1
  print('small', small)
  print('medium', medium)
  print('large', large)
if __name__ == "__main__":
 #train_test_segment_by_roi()
#   object_count()
  pos_process()
  #vis()
  test_process()





