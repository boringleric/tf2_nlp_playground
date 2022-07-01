
# 调用生成的bert pb生成词向量

from models.check.check_bert_pb import load_bert_pb_and_test

tmp_file = './model_pool/cls_bert_model.pb'
retvec = load_bert_pb_and_test(tmp_file, "你好")

# 调用ner pb生成ner识别结果

from models.check.check_ner_model_pb import load_ner_pb_and_test
from models.data_process.preprocess import load_ner_data

all_data, categories = load_ner_data("./data_bank/ner_test_data.txt")

entities = load_ner_pb_and_test('./model_pool/ner_full_model.pb', '上海欧涌实业有限公司', categories)
print(entities)