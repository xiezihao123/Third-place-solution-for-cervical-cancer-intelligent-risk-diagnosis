import pickle
pkl_file = open('./result/faster_rcnn_ioupred.pkl','rb')
data = pickle.load(pkl_file)
print(data[0])
