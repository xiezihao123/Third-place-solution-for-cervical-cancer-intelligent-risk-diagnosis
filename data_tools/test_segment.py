import os
def test_segemnt(num):
    path = '/mnt/C/tianchi/VOC2007/ImageSets/Main/'
    f = open(path + 'test.txt', 'r')
    lines = f.readlines()
    f.close()
    test_list = []
    for line in lines:
        test_list.append(line.split('\n')[0])
    print(len(test_list))
    a = len(test_list)//num
    print(a)
    b = len(test_list)%num
    print(b)
    for i in range(num):
        if i==num-1:
            f = open(path+'test'+str(i)+'.txt', 'w')
            for j in range(i*a, i*a+b+a):
                f.write(test_list[j]+'\n')
            f.close()
        else:
            f = open(path+'test'+str(i)+'.txt', 'w')
            for j in range(i*a, i*a+a):
                f.write(test_list[j]+'\n')
            f.close()

if __name__=='__main__':
    test_segemnt(2)
