from milvus import Milvus, IndexType, MetricType
from sklearn import preprocessing

def get_connection(server_config):
    '''
    连接server
    '''
    milvus = Milvus(**server_config)
    return milvus


def connection_and_creation(server_config, _Collection_Name, _DIM, _Metric_Type, index_file_size=2048):
    '''
    连接server并创建collection
    '''
    metric_type = MetricType.L2
    milvus = Milvus(**server_config)
    if _Metric_Type == 'IP':
        metric_type = MetricType.IP
    # 检查是否有该集合
    status, ok = milvus.has_collection(_Collection_Name)
    if not ok:
        param = {
            'collection_name': _Collection_Name,
            'dimension': _DIM,
            'metric_type': metric_type,
            'index_file_size': index_file_size
        }
        milvus.create_collection(param)

    return milvus


def insert_content(milvus, _Collection_Name, index_type, embed, hashlist, ip_flag, nlist=2048):
    '''
    插入embedding
    '''
    real_index_type = IndexType.FLAT
    try:
        vector_ids = hashlist
        vector_ids = hashlist
        if ip_flag:
            embed_norm = preprocessing.normalize(embed, axis=1, norm='l2').tolist()
            status, ids = milvus.insert(collection_name=_Collection_Name, records=embed_norm, ids=vector_ids)
        else:
            status, ids = milvus.insert(collection_name=_Collection_Name, records=embed, ids=vector_ids)
        if not status.OK():
            print("Insert failed: {}".format(status))
            return -1
        
        milvus.flush([_Collection_Name])
        if index_type == "IVF":
            real_index_type = IndexType.IVF_FLAT
            index_param = {
            'nlist': nlist           # if IVF
            }
            status = milvus.create_index(_Collection_Name, real_index_type, index_param)
        elif index_type == "ANNOY":
            real_index_type = IndexType.ANNOY
            index_param = {
            'n_trees': 100          
            }
            status = milvus.create_index(_Collection_Name, real_index_type, index_param)
        else:
            status = milvus.create_index(_Collection_Name, real_index_type)
        
        if not status.OK():
            print("Index failed: {}".format(status))
            return -1

    except Exception as e:
        print(e)
        return -1
    
    return 0


def get_result(milvus, _Collection_Name, search_param, topk, text):
    '''
    获得检索结果
    '''
    param = {
        'collection_name': _Collection_Name,
        'query_records': text,
        'top_k': topk,
        'params': search_param,
    }

    status, results = milvus.search(**param)
    if not status.OK():
        print(f"Search failed: {status}")
        return []
    else:
        return results

