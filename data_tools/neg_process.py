import numpy as np
import os
import cv2 as cv
import random
root='/home/xiezihao/mmdetection-master/data/tianchi/'
labels_path = '/home/xiezihao/mmdetection-master/data/tianchi/labels_train/'
images_path = '/home/xiezihao/mmdetection-master/data/tianchi/images_train/'
neg_root = '/home/xiezihao/mmdetection-master/data/tianchi/neg_samples/'
neg_labels_path = '/home/xiezihao/mmdetection-master/data/tianchi/neg_samples/labels/'
neg_images_path = '/home/xiezihao/mmdetection-master/data/tianchi/neg_samples/images/'

def compute_iou(posline_0,posline_1):
  list_0, list_1 = posline_0.split(' '), posline_1.split(' ')
  x0,y0,w0,h0 = float(list_0[0]), float(list_0[1]), int(list_0[2]), int(list_0[3])
  x1,y1,w1,h1 = float(list_1[0]), float(list_1[1]), int(list_1[2]), int(list_1[3])
  xmin = max(x0-w0/2, x1-w1/2)
  ymin = max(y0-h0/2, y1-h1/2)
  xmax = min(x0+w0/2, x1+w1/2)
  ymax = min(y0+h0/2, y1+h1/2)
  w_ = np.maximum(xmax - xmin + 1, 0)
  h_ = np.maximum(ymax - ymin + 1, 0)
  inters = w_*h_
  uni = w0*h0+w1*h1-inters
  return inters/uni

def compute_iou_seg(line0,line1):
  x0, y0, w0, h0 = float(line0[0]), float(line0[1]), float(line0[2]), float(line0[3])
  x1, y1, w1, h1 = float(line1[0]), float(line1[1]), float(line1[2]), float(line1[3])
  xmin = max(x0, x1)
  ymin = max(y0, y1)
  xmax = min(x0+w0, x1+w1)
  ymax = min(y0+h0, y1+h1)
  w_ = np.maximum(xmax - xmin + 1, 0)
  h_ = np.maximum(ymax - ymin + 1, 0)
  inters = w_ * h_
  uni = w0 * h0 + w1 * h1 - inters
  return min(inters/w0 * h0, inters/w1 * h1)



def get_iou_threshold():
  label_files = os.listdir(labels_path)
  # num_labels = len(label_files)
  iou = []
  num_labels = len(label_files)
  for i in range(num_labels):
    f = open(labels_path+label_files[i])
    lines = f.readlines()
    n = len(lines)
    f.close()
    label_iou = []
    if n==1:
      continue
    else:
      for j in range(1,n):
        tmp = compute_iou(lines[j-1],lines[j])
        label_iou.append(tmp)
    iou.extend(label_iou)
  max_iou = max(iou)
  print('max_iou:',max_iou)


def fill_neg():
  max_iou = 0.25
  neg_image_files = os.listdir(neg_images_path)
  label_files = os.listdir(labels_path)
  num_labels = len(label_files)
  for neg_image_file in neg_image_files:
    print('neg_image:', neg_image_file)
    neg_image_path = neg_images_path+neg_image_file
    neg_label_path = neg_labels_path+neg_image_file[:-4]+'.txt'
    neg_image = cv.imread(neg_image_path)
    print(neg_image.shape)
    H_neg, W_neg = neg_image.shape[0], neg_image.shape[1]
    idx_label = random.randint(0,num_labels-1)
    label_name = label_files[idx_label]
    label_path = labels_path+label_name
    image_path = images_path+label_name[:-4]+'.jpg'
    print('random image path:', image_path)
    image = cv.imread(image_path)
    f = open(label_path)
    lines = f.readlines()
    num_pos = len(lines)
    f.close()
    #get random pos(x,y) in neg_img
    lines_neg = [] #(leftx,lefty,w,h)

    for i in range(num_pos):

      line = lines[i].split(' ')
      x_train, y_train, w, h= float(line[0]), float(line[1]), float(line[2]), float(line[3])
      if w/h>5 or h/w>5:
        print(w / h)
        while 1:
          pass
    #without backpack, may drop in death cycle
      x = random.randint(0, int(W_neg - w ))
      y = random.randint(0, int(H_neg - h ))
      if i==0:
        lines_neg.append([x, y, w, h])
      if i>=1:
        count = 0
        while 1:
          count += 1
          if count > 200:
            break
          flag = 0
          for j in range(len(lines_neg)):
            if compute_iou_seg([x,y,w,h],lines_neg[j])>max_iou:
              flag = 1
              break
          if flag:
            x = random.randint(0, int(W_neg - w))
            y = random.randint(0, int(H_neg - h))
          else:
            break
        lines_neg.append([x,y,w,h])

    f = open(neg_label_path,'w')
    for i in range(len(lines_neg)):
      line = lines[i].split(' ')
      x_train, y_train, w, h = float(line[0]), float(line[1]), float(line[2]), float(line[3]) #center
      x_train_left, y_train_left = int(x_train-w/2), int(y_train-h/2)
      w,h = int(w),int(h)
      X_neg,Y_neg=int(lines_neg[i][0]), int(lines_neg[i][1]) #left
      neg_image[Y_neg:Y_neg+h, X_neg:X_neg+w] = image[y_train_left:y_train_left+h, x_train_left:x_train_left+w]
      f.write(str(X_neg+w/2) +' '+ str(Y_neg+h/2) +' '+ str(w) + ' ' + str(h) +' '+ 'pos'+'\n')
      print(str(X_neg+w/2) +' '+ str(Y_neg+h/2) +' '+ str(w) + ' ' + str(h) +' '+ 'pos'+'\n')
    f.close()
    cv.imwrite(neg_image_path,neg_image)
    print(neg_image_file)



if __name__ == '__main__':
  # get_iou_threshold() #max_iou: 0.19105235200377738
  fill_neg()