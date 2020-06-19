"""
Created on 11:00:05 8/23/2018

@author: LinZhao
"""
import os
import sys
import json
import argparse
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# def statistic_object_size_for_xml_annotation(xml_path, step_bin, num_bin):
#     count_size = [0 for i in range(num_bin + 1)]
#     xml_files = os.listdir(xml_path)
#     for xml_file in xml_f:                                                                                             iles:
#         dom_tree = ET.parse(os.path.join(xml_path, xml_file))
#         dom_root = dom_tree.getroot()
#
#         for obj in dom_root.iter('object'):
#             obj_box = obj.find('bndbox')
#             box_xmin = float(obj_box.find('xmin').text)  # type: int
#             box_ymin = float(obj_box.find('ymin').text) # type: int
#             box_xmax = float(obj_box.find('xmax').text)  # type: int
#             box_ymax = float(obj_box.find('ymax').text)  # type: int
#             box_w = box_xmax - box_xmin
#             box_h = box_ymax - box_ymin
#             area = box_w*box_h
#             ratio = area/(640*360)
#             idx = int(ratio*100)
#             if idx >= num_bin:
#                 count_size[num_bin] += 1
#             else:
#                 count_size[idx] += 1
#
#     return count_size


def statistic_size_for_json_annotation(json_path):
    list = json.load(open(json_path))
    num = len(list)
    roi, pos = [], []
    for i in range(num):
        if list[i]['class'] == 'roi':
            roi.append(list[i])
        elif list[i]['class'] == 'pos':
            pos.append(list[i])
        else:
            print('error')
            return
    # print('roi nums:',len(roi))
    # print('pos nums:',len(pos))
    num_pos,roi_area,roi_w_h_ratio,pos_area,pos_w_h_ratio, area_pos_in_roi = [],[],[],[],[],[]
    ##
    pos_w,pos_h=[],[]
    tmp = 0
    for i in range(len(roi)):
        r_x = roi[i]['x']
        r_y = roi[i]['y']
        r_w = roi[i]['w']
        r_h = roi[i]['h']
        for j in range(len(pos)):
            #number of pos in per roi
            p_x = pos[j]['x']
            p_y = pos[j]['y']
            p_w = pos[j]['w']
            p_h = pos[j]['h']
            if p_x>r_x and p_y>r_y and p_w<r_w and p_h<r_h:
                tmp +=1
            roi_area.append(r_w * r_h)
            roi_w_h_ratio.append(r_w / r_h)
            pos_area.append(p_w * p_h)
            pos_w_h_ratio.append(p_w / p_h)
            area_pos_in_roi.append((p_w * p_h)/(r_w * r_h))
            pos_w.append(p_w)
            pos_h.append(p_h)
        num_pos.append(tmp)
        # if tmp>50:
        #     print('\n','-------liqundian324--:',json_path)
        #     break
          #,[r_w,r_h,p_w,p_h]
        tmp=0
    ## print(len(num_pos))
    # print('num_pos per roi:',num_pos)
    # print('roi_area:', roi_area)
    # print('roi_w_h_ratio:', roi_w_h_ratio)
    # print('pos_area:', pos_area)
    # print('pos_w_h_ratio:', pos_w_h_ratio)
    return num_pos,roi_area,roi_w_h_ratio,pos_area,pos_w_h_ratio,area_pos_in_roi,pos_w,pos_h

def cnt_intervals(input_list,name,bins_=10):
    min_ = min(input_list)
    max_ = max(input_list)
    c={name:input_list}
    data=pd.DataFrame(c)
    data=data[name].value_counts(bins=bins_,sort=False)
    # pd.set_option('display.max_rows',None)
    print(data)
    print(name,':min=',min_,', max=',max_,'\n')

def draw_hist(input_list,name,colum_num=10):
    min_ = min(input_list)
    max_ = max(input_list)
    print(name,'min_:',min_)
    print(name,'max_:',max_)
    x = np.array(input_list)
    bins = np.arange(min_, max_+1, (max_-min_)/colum_num)
    plt.hist(input_list,max_)
    plt.hist(x,bins,color='fuchsia',alpha=0.5)
    plt.xlabel(name)
    plt.ylabel('count')
    plt.show()
    # count_size = [0 for i in range(num_bin + 1)]
    # min_ = 1000
    # max_ = 10
    # with open(json_path,"rb") as f:
    #     json_obj = json.load(f)
    #
    #     key = 'annotation'
    #     if key not in json_obj.keys():
    #         key = 'annotations'
    #
    #     for i  in range(len(json_obj[key])):
    #         bbox = json_obj[key][i]['bbox']
    #         box_w = bbox[2]
    #         box_h = bbox[3]
    #         c = int(max(box_w, box_h))
    #         # if c== 2604:
    #         #     print(json_obj[key][i]['image_id'])
    #         #     print(json_obj[key][i]['category_id'])
    #         #     print(json_obj[key][i]['bbox'])
    #         # if c== 1:
    #         #     print(json_obj[key][i]['image_id'])
    #         #     print(json_obj[key][i]['category_id'])
    #         #     print(json_obj[key][i]['bbox'])
    #         idx = int(max(box_h,box_w)) // step_bin
    #         if c<min_:
    #             min_ = c
    #         if c>max_:
    #             max_ = c
    #         if idx >= num_bin:
    #             count_size[num_bin] += 1
    #         else:
    #             count_size[idx] += 1
    # print(min_,max_)
    # return count_size


def argv_parse():
    parser = argparse.ArgumentParser('statistics object size for voc or coco')
    parser.add_argument('--anno_path', type=str, default='/mnt/C/object365/objects13_train_large.json', help='annotation path')
    parser.add_argument('--type', type=str, default='json', help='xml or json')
    parser.add_argument('--step', type=float, default=16, help='step')
    parser.add_argument('--num_bin', type=int, default=32, help='num of bins')
    parser.add_argument('--output_path', type=str, default='./data.txt', help='save file path')
    # if len(sys.argv) == 1:
    #     parser.print_help()
    #     sys.exit(1)
    return parser.parse_args()


if __name__ == '__main__':
    args = argv_parse()

    root =  '/home/admin/jupyter/Data/train'
    s = [s  for s in os.listdir(root) if s.find('.json')!=-1]
    num_pos_all, roi_area_all, roi_w_h_ratio_all, pos_area_all, pos_w_h_ratio_all,area_pos_in_roi_all=[],[],[],[],[],[]
    pos_w_all,pos_h_all=[],[]
    js_path = [os.path.join(root, f) for f in s]
    for json_path in js_path:
        num_pos,roi_area,roi_w_h_ratio,pos_area,pos_w_h_ratio,area_pos_in_roi,pos_w,pos_h = statistic_size_for_json_annotation(json_path)
        num_pos_all.extend(num_pos)
        roi_area_all.extend(roi_area)
        roi_w_h_ratio_all.extend(roi_w_h_ratio)
        pos_area_all.extend(pos_area)
        pos_w_h_ratio_all.extend(pos_w_h_ratio)
        area_pos_in_roi_all.extend(area_pos_in_roi)
        pos_w_all.extend(pos_w)
        pos_h_all.extend(pos_h)

    print('max_pos_w:',max(pos_w_all),'min_pos_w:',min(pos_w_all))
    print('max_pos_h:',max(pos_h_all),'min_pos_h:',min(pos_h_all))

    print('\n','num_pos_all:',num_pos_all, '\n','roi_area_all:',roi_area_all,'\n', 'roi_w_h_ratio_all:',roi_w_h_ratio_all)
    print('\n' ,'pos_area_all:',pos_area_all,'\n', 'pos_w_h_ratio_all:',pos_w_h_ratio_all,'\n','area_pos_in_roi_all:',area_pos_in_roi_all)

    ##add
    sqrt_roi = []
    sqrt_pos = []
    for n in roi_area_all:
        sqrt_roi.append(n ** 0.5)
    for n in pos_area_all:
        sqrt_pos.append(n ** 0.5)
    print('--------', pos_area_all, sqrt_pos)
    cnt_intervals(sqrt_roi, 'sqrt_roi', 10)
    cnt_intervals(sqrt_roi, 'sqrt_roi', 20)
    cnt_intervals(sqrt_pos, 'sqrt_pos', 10)
    cnt_intervals(sqrt_pos, 'sqrt_pos', 20)

    cnt_intervals(num_pos_all,'num_pos_all',10)
    cnt_intervals(num_pos_all,'num_pos_all',20)
    cnt_intervals(roi_area_all,'roi_area_all',10)
    cnt_intervals(roi_area_all,'roi_area_all',20)
    cnt_intervals(roi_w_h_ratio_all,'roi_w_h_ratio_all',10)
    cnt_intervals(roi_w_h_ratio_all,'roi_w_h_ratio_all',20)
    cnt_intervals(pos_area_all,'pos_area_all',10)
    cnt_intervals(pos_area_all,'pos_area_all',20)
    cnt_intervals(pos_w_h_ratio_all,'pos_w_h_ratio_all',10)
    cnt_intervals(pos_w_h_ratio_all,'pos_w_h_ratio_all',20)
    cnt_intervals(area_pos_in_roi_all, 'area_pos_in_roi_all', 10)
    cnt_intervals(area_pos_in_roi_all, 'area_pos_in_roi_all', 20)
    cnt_intervals(pos_w_all,'pos_w_all',10)
    cnt_intervals(pos_w_all,'pos_w_all',20)
    cnt_intervals(pos_h_all, 'pos_h_all', 10)
    cnt_intervals(pos_h_all, 'pos_h_all', 20)


    # if args.type == 'json':
    #     object_size = statistic_object_size_for_json_annotation(args.anno_path, args.step, args.num_bin)
    # else:
    #     object_size = statistic_object_size_for_xml_annotation(args.anno_path, args.step, args.num_bin)
    #
    # lines = ['{}~{}: {}\n'.format(i * args.step, (i + 1) * args.step, object_size[i]) for i in range(len(object_size))]
    # with open(args.output_path, 'w') as fd:
    #     fd.writelines(lines)
