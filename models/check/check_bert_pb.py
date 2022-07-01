from bert4keras.tokenizers import Tokenizer
import tensorflow 

def load_bert_pb_and_test(pbpath, text='你好啊', vocab_file='/home/chenlei/ModelZoo/TF_Model/chinese_roberta_wwm_ext_L-12_H-768_A-12/vocab.txt'): 
    '''
    尝试加载pbpath的文件，并输出text的embedding。
    参照bert-serving-start -pooling_layer=-2 -model_dir=/tmp_bert -num_worker=1 -graph_tmp_dir=/tmp -pooling_strategy=REDUCE_MEAN
    '''
    if tensorflow.__version__ > '2.0':
        import tensorflow.compat.v1 as tf
        tf.disable_v2_behavior()
    else:
        import tensorflow as tf

    from tensorflow.python.platform import gfile
    maxlen = 128
    
    GRAPH_PB_PATH = pbpath
    sess = tf.Session() 
    sess.run(tf.global_variables_initializer())
    print("load graph")
    with gfile.FastGFile(GRAPH_PB_PATH,'rb') as f:
        graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')

    input_x = sess.graph.get_tensor_by_name('input_ids:0')
    input_y = sess.graph.get_tensor_by_name('input_mask:0')
    input_z = sess.graph.get_tensor_by_name('input_type_ids:0')

    op = sess.graph.get_tensor_by_name('final_encodes:0')

    tokenizer = Tokenizer(vocab_file, do_lower_case=True)
    input_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)

    input_mask = [1] * len(input_ids)
    segment_ids = [0] * len(input_ids)

    retf = sess.run(op, feed_dict={input_x: [input_ids], input_y: [input_mask], input_z:[segment_ids]})

    return retf