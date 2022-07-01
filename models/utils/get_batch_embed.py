from tensorflow.python.platform import gfile
import tensorflow as tf 
import numpy as np
from bert4keras.snippets import sequence_padding
from bert4keras.tokenizers import Tokenizer

if tf.__version__ > '2.0':
    tf.compat.v1.disable_eager_execution()
    import tensorflow.compat.v1 as tf

sess = tf.Session() 

def load(pbpath):
    global sess
    
    GRAPH_PB_PATH = pbpath
    sess.run(tf.global_variables_initializer())
    print("load graph")
    with gfile.FastGFile(GRAPH_PB_PATH,'rb') as f:
        graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')


def get_embedding(textlist, maxlen=128):

    global sess
       
    vocab_file = './ModelZoo/TF_Model/chinese_roberta_wwm_ext_L-12_H-768_A-12/vocab.txt'
     
    input_x = sess.graph.get_tensor_by_name('input_ids:0')
    input_y = sess.graph.get_tensor_by_name('input_mask:0')
    input_z = sess.graph.get_tensor_by_name('input_type_ids:0')

    op = sess.graph.get_tensor_by_name('final_encodes:0')

    tokenizer = Tokenizer(vocab_file, do_lower_case=True)
    
    input_ids, input_masks, segment_ids = [], [], []
    for text in textlist:       
        input_id, segment_id = tokenizer.encode(text, maxlen=maxlen)
        input_ids.append(input_id)
        input_mask = [1] * len(input_id)
        input_masks.append(input_mask)
        segment_id = [0] * len(input_id)
        segment_ids.append(segment_id)
    
    input_ids = sequence_padding(input_ids)
    input_masks = sequence_padding(input_masks)
    segment_ids = sequence_padding(segment_ids)

    retf = sess.run(op, feed_dict={input_x: input_ids, input_y: input_masks, input_z: segment_ids})
    
    return np.array(retf)
