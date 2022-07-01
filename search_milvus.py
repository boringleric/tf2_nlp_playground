from utils.milvus_operation import get_connection, get_result
from utils.sqlite_operation import search_content_db
from sklearn import preprocessing

# -------------- 加载配置 ----------------

# Milvus配置
server_config = {
    'host': "192.168.7.247",
    'port': "19530"
}

_Collection_Name = "testmilvus"
_Metric_Type = "IP"
search_param = {
        "nprobe": 1000,
        "search_k":-1
    }

# 数据库sqlite配置
tabel_name = _Collection_Name

from models.utils.get_batch_embed  import load, get_embedding
load('./model_pool/cls_bert_model.pb')

# -------------- 连接milvus ----------------

milvus = get_connection(server_config)

def get_search_result(text, ret_nums=10):
    retlist = []
    result = get_embedding([text])

    print(result)
    # 检索milvus
    res_new = result
    if _Metric_Type=='IP':
        embed_norm = preprocessing.normalize(res_new, norm='l2').tolist()
        results = get_result(milvus, _Collection_Name, search_param, ret_nums, embed_norm)
    else:
        results = get_result(milvus, _Collection_Name, search_param, ret_nums, res_new)
    if results == []:
        return retlist

    # 检索sqlite
    dis_array = results.distance_array[0]
    ind_array = results.id_array[0]
    for ind, uuid in enumerate(ind_array):
        res = search_content_db(tabel_name, uuid)
        for r in res:
            retlist.append([dis_array[ind], r[1], r[0]])

    return retlist


def tmp():
    while True:
        print("input pls:")
        content = input()
        print("-"*50)
        ret = get_search_result(content)
        for r in ret:
            print("%-.8f : %-10s : %s" % (r[0], r[1], r[2]))

tmp()   