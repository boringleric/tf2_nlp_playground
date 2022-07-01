# tf2_nlp_playground

存放工作学习中整理的tf代码，目前功能有：

· 使用分类方式，训练bert向量，并将分类模型和bert向量分别导出tf1所需的pb文件；    
· 训练实体识别模型，并导出tf1所需的pb文件。

# 配置环境

python 3.7.11  
cudatoolkit 10.1.243   
cudnn 7.6.5  
tensorflow-gpu 2.3.1  
bert4keras 0.11.3

# 项目简介  

训练方式按照bert4keras的方式，参考了大量bert4keras以及bert-as-service的代码，表示感谢！  

models中的barn文件夹下包含两类模型训练文件，clsmodel代表分类模型， nermodel代表实体识别模型，调用方式见train_call.py