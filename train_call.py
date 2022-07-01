import os
from models.data_process.preprocess import load_cls_csv, load_ner_data

os.environ['TF_KERAS'] ='1'
from models.barn.clsmodel import clsModel
from models.barn.nermodel import nerModel

## ---------------- cls -------------------------------

model_name = 'chinese_roberta_wwm_ext_L-12_H-768_A-12'
config_path = './ModelZoo/TF_Model/' + model_name + '/bert_config.json'
checkpoint_path = './ModelZoo/TF_Model/' + model_name + '/bert_model.ckpt'
dict_path = './ModelZoo/TF_Model/' + model_name + '/vocab.txt'

all_data, all_labels = [], []
train_data, valid_data = [], []

# 获取cls数据
all_data, all_labels, label_dict = load_cls_csv([
    ("./data_bank/cls_test_data.csv", "utf-8", ','),  # gb18030    utf-8
    ])

# 创建模型
clsmod = clsModel(len(label_dict), config_path, dict_path, checkpoint_path=checkpoint_path)
# 训练模型
clsmod.train([all_data, all_labels], label_dict, epochs=1, batch_size=32)
# 导出模型（只需要bert的话不需要这个）
clsmod.export_all_model('./model_pool')
# 导出bert模型，需要设定为非eager模式，顺序不可颠倒
clsmod.export_bert('./model_pool', config_path)

## ------------------- ner -------------------------------

all_data, all_labels = [], []
train_data, valid_data = [], []

# 获取ner数据
all_data, categories = load_ner_data("./data_bank/ner_test_data.txt")

model_name = 'chinese_roberta_wwm_ext_L-12_H-768_A-12'
config_path = './ModelZoo/TF_Model/' + model_name + '/bert_config.json'
checkpoint_path = './ModelZoo/TF_Model/' + model_name + '/bert_model.ckpt'
dict_path = './ModelZoo/TF_Model/' + model_name + '/vocab.txt'

# 创建模型
nermod = nerModel(categories, config_path, dict_path, checkpoint_path=checkpoint_path)
# 训练模型
nermod.train(all_data, epochs=3, batch_size=32)
# 导出模型
nermod.export_all_model('./model_pool')
# 测试
ret = nermod.predict('宁波蚂蚁帝国教育科技有限公司')
print(ret)