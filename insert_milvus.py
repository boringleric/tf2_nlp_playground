import tqdm
from models.data_process.preprocess import load_cls_csv
from models.utils.get_batch_embed import get_embedding, load
from bert4keras.snippets import DataGenerator
from utils.milvus_operation import insert_content, connection_and_creation
from utils.sqlite_operation import create_table, get_uuid, insert_into_table
import random

random.seed(29)
# Milvus配置
server_config = {
    'host': "192.168.7.247",
    'port': "19530"
}

_Collection_Name = "testmilvus"
_DIM = 768  
_Metric_Type = "IP"
index_type = "IVF"
ip_flag = True

load('./model_pool/cls_bert_model.pb')
create_table(_Collection_Name)
milvus = connection_and_creation(server_config, _Collection_Name, _DIM, _Metric_Type)

all_data, all_labels, _ = load_cls_csv([
    ("./DataBank/Bert_finetune/china_insureance/china_insurance_bxdq_ver6_20211020.csv", "gb18030", ','),  # gb18030    utf-8
    ])
uuidlist = get_uuid(all_data)
calc_all_data = [[all_data[idx], all_labels[idx], uuidlist[idx]] for idx, _ in enumerate(all_data)]


class data_generator(DataGenerator):
    """数据生成器
    """
    def __init__(self, data, batch_size=32, buffer_size=None):
        super().__init__(data, batch_size, buffer_size)

    def __iter__(self, random=True):
        batch_text, batch_labels, batch_uuids = [], [], []
        for is_end, (text, label, uuid) in self.sample(random):
            batch_text.append(text)
            batch_labels.append(label)
            batch_uuids.append(uuid)
            if len(batch_text) == self.batch_size or is_end:
                yield batch_text, batch_labels, batch_uuids
                batch_text, batch_labels, batch_uuids = [], [], []

train_generator = data_generator(calc_all_data, batch_size=32)

for data in tqdm.tqdm(train_generator):
    insert_into_table(_Collection_Name, data[2], data[0], data[1])
    embed = get_embedding(data[0])
    ret = insert_content(milvus, _Collection_Name, index_type, embed, data[2], ip_flag)
