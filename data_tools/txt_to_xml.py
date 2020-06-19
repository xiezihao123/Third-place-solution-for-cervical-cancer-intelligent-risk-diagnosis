from xml.dom.minidom import Document
import os
import os.path
from PIL import Image
import re

dataset = 'tianchi'
def txt_to_xml():
    ann_path = "/home/admin/jupyter/tianchi_data/test/roi_label/"
    img_path = "/home/admin/jupyter/tianchi_data/test/ROI_images/"
    xml_path = "/home/admin/jupyter/tianchi_data/test/VOC2007_roi/Annotations/"
    if not os.path.exists(xml_path):
        os.mkdir(xml_path)
    
    for files in os.walk(ann_path):
        temp = "/home/admin/jupyter/tianchi_data/temp/"
        if not os.path.exists(temp):
            os.mkdir(temp)
#         for file in files[2]:
        f = open('/home/admin/jupyter/tianchi_data/train/VOC2007/ImageSets/Main/test_add.txt','r')
        names = f.readlines()
        f.close()
        for name in names:
            print(name.split('\n')[0])
            name = name.split('\n')[0]
#             img_name = os.path.splitext(file)[0] + '.jpg'
            img_name = img_path + name + '.jpg'
#             print(file + "-->start!")
            if img_name.find('empty')!=-1 or img_name.find('segment')!=-1:
                continue
            print(img_name)
#             fileimgpath = img_path + img_name
#             im = Image.open(fileimgpath)
            im = Image.open(img_name)
            width = int(im.size[0])
            height = int(im.size[1])

            filelabel = open(ann_path + name+'.txt', "r")
            lines = filelabel.read().split('\n')
            obj = lines[:len(lines) - 1]

            filename = xml_path + name + '.xml'
            print(filename)
            writeXml(temp, img_name, width, height, obj, filename)
#         os.rmdir(temp)


def writeXml(tmp, imgname, w, h, objbud1, wxml):
    doc = Document()
    # owner
    annotation = doc.createElement('annotation')
    doc.appendChild(annotation)
    # owner
    folder = doc.createElement('folder')
    annotation.appendChild(folder)
    folder_txt = doc.createTextNode(dataset)
    folder.appendChild(folder_txt)

    filename = doc.createElement('filename')
    annotation.appendChild(filename)
    filename_txt = doc.createTextNode(imgname)
    filename.appendChild(filename_txt)
    # ones#
    source = doc.createElement('source')
    annotation.appendChild(source)

    database = doc.createElement('database')
    source.appendChild(database)
    database_txt = doc.createTextNode("The dota Database")
    database.appendChild(database_txt)

    annotation_new = doc.createElement('annotation')
    source.appendChild(annotation_new)
    annotation_new_txt = doc.createTextNode(dataset)
    annotation_new.appendChild(annotation_new_txt)

    image = doc.createElement('image')
    source.appendChild(image)
    image_txt = doc.createTextNode("flickr")
    image.appendChild(image_txt)
    # onee#
    # twos#
    size = doc.createElement('size')
    annotation.appendChild(size)

    width = doc.createElement('width')
    size.appendChild(width)
    width_txt = doc.createTextNode(str(w))
    width.appendChild(width_txt)

    height = doc.createElement('height')
    size.appendChild(height)
    height_txt = doc.createTextNode(str(h))
    height.appendChild(height_txt)

    depth = doc.createElement('depth')
    size.appendChild(depth)
    depth_txt = doc.createTextNode("3")
    depth.appendChild(depth_txt)
    # twoe#
    segmented = doc.createElement('segmented')
    annotation.appendChild(segmented)
    segmented_txt = doc.createTextNode("0")
    segmented.appendChild(segmented_txt)

    for i in range(0, len(objbud1)):
        # threes#
        objbud = objbud1[i].split()
        #if not objbud:
        #    continue
        name_ = objbud[-1]
        # name_ = objbud[-2]
        # x1 = str(int(min(float(objbud[0]), float(objbud[2]), float(objbud[4]), float(objbud[6]))))
        # x2 = str(int(max(float(objbud[0]), float(objbud[2]), float(objbud[4]), float(objbud[6]))))
        # y1 = str(int(min(float(objbud[1]), float(objbud[3]), float(objbud[5]), float(objbud[7]))))
        # y2 = str(int(max(float(objbud[1]), float(objbud[3]), float(objbud[5]), float(objbud[7]))))
        # difficult_flag = str(int(objbud[9]))
        x1 = str(int(float(objbud[0])-float(objbud[2])/2))
        x2 = str(int(float(objbud[0])+float(objbud[2])/2))
        y1 = str(int(float(objbud[1])-float(objbud[3])/2))
        y2 = str(int(float(objbud[1])+float(objbud[3])/2))
        difficult_flag = str(0)
        object_new = doc.createElement("object")
        annotation.appendChild(object_new)

        name = doc.createElement('name')
        object_new.appendChild(name)
        name_txt = doc.createTextNode(name_)
        name.appendChild(name_txt)

        pose = doc.createElement('pose')
        object_new.appendChild(pose)
        pose_txt = doc.createTextNode("Left")
        pose.appendChild(pose_txt)

        truncated = doc.createElement('truncated')
        object_new.appendChild(truncated)
        truncated_txt = doc.createTextNode("0")
        truncated.appendChild(truncated_txt)

        difficult = doc.createElement('difficult')
        object_new.appendChild(difficult)
        difficult_txt = doc.createTextNode(difficult_flag)
        difficult.appendChild(difficult_txt)
        # threes-1#
        bndbox = doc.createElement('bndbox')
        object_new.appendChild(bndbox)

        xmin = doc.createElement('xmin')
        bndbox.appendChild(xmin)
        xmin_txt = doc.createTextNode(x1)
        xmin.appendChild(xmin_txt)

        ymin = doc.createElement('ymin')
        bndbox.appendChild(ymin)
        ymin_txt = doc.createTextNode(y1)
        ymin.appendChild(ymin_txt)

        xmax = doc.createElement('xmax')
        bndbox.appendChild(xmax)
        xmax_txt = doc.createTextNode(x2)
        xmax.appendChild(xmax_txt)

        ymax = doc.createElement('ymax')
        bndbox.appendChild(ymax)
        ymax_txt = doc.createTextNode(y2)
        ymax.appendChild(ymax_txt)
        # threee-1#
        # threee#

    tempfile = tmp + "test.xml"
    with open(tempfile, "wb") as f:
        f.write(doc.toprettyxml(indent='\t', encoding='utf-8'))

    rewrite = open(tempfile, "r")
    lines = rewrite.read().split('\n')
    newlines = lines[1:len(lines) - 1]

    fw = open(wxml, "w")
    for i in range(0, len(newlines)):
        fw.write(newlines[i] + '\n')

    fw.close()
    rewrite.close()
    os.remove(tempfile)
    return
    
if __name__=='__main__':
    txt_to_xml()

