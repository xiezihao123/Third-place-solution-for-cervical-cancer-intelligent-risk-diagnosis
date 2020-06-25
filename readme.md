# 说明文档
mmdetection 1.0    pytorch 1.1  python 3.7  相关安装步骤见https://github.com/open-mmlab/mmdetection/blob/master/docs/install.md


# 说明文档


### 1.代码文件结构

我们队伍核心代码都在目录/home/admin/jupyter/mmdetection-master里面，接下来我们以$PATH命名此目录。$PATH/data_tools里面存放的是各种数据处理的代码，$PATH/config里面存放的参数配置文件， $PATH/mmdet 里面存放的是模型构建的代码，$PATH/result里面存放的是测试生成的结果pkl文件，$PATH/tianchi_json里面存放的是通过测试结果pkl文件后处理生成的json文件。$PATH/tools里面存放的是训练和测试的脚本。另外数据存放在/home/admin/jupyter/tianchi_data下面。

### 2.  可直接执行的训练脚本和推理脚本。（训练脚本需要可复现训练出最佳模型，推理脚本需要可直接推理得到排行榜最优结果）

#### 训练脚本

cd /home/admin/jupyter/mmdetection-master

bash train.sh

#### 推理脚本

cd /home/admin/jupyter/mmdetection-master

bash test.sh

### 3. 算法创新
首先我们模型采用的backbone是对小目标友好的HRNet网络，然后为了更好的对框进行分类和回归，我们采用级联的方式进行回归。在损失函数方面，我们采用了更能反应框与ground truth的位置关系的iou_loss。在采样方面，我们提出了一种混合采样的方式，在级联的三个阶段，第一个阶段采用难样本挖掘，第二个第三个采用随机采用。在数据处理方面，我们提出在已有类别基础上增加一类正常细胞的类别，以此来利用阴性图片以及减少误检。另外由于倒数第二类目标存在极大尺寸的情况，我们提出直接用roi当训练样本，与切割图片的检测结果进行结合，以缓和目标被切碎的问题。

### 4.  其他注意事项。

在数据处理阶段，我们是首先训练了一个六类的模型，然后用它在数据上的误检框，重新定义为一类，以此来利用阴性图片和ROI中的空图片，所以最后我们训练的是7类（不包括背景）的模型。此数据生成过程比较繁琐，具体步骤在train.sh被注释部分。
