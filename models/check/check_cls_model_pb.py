import tensorflow as tf
from bert4keras.tokenizers import Tokenizer
from models.utils.keras_to_pb import wrap_frozen_graph
import numpy as np

def load_cls_pb_and_test(save_pb_path, text, vocab_file='./ModelZoo/TF_Model/chinese_roberta_wwm_ext_L-12_H-768_A-12/vocab.txt'): 

    tokenizer = Tokenizer(vocab_file, do_lower_case=True)

    with tf.io.gfile.GFile(save_pb_path, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        loaded = graph_def.ParseFromString(f.read())

    # Wrap frozen graph to ConcreteFunctions
    frozen_func = wrap_frozen_graph(graph_def=graph_def,
                                    inputs=["input_ids:0", "input_segments:0"],
                                    outputs=["Identity:0"],
                                    print_graph=False)
                                    
    print("Frozen model inputs: ")
    print(frozen_func.inputs)
    print("Frozen model outputs: ")
    print(frozen_func.outputs)
    
    token_ids, segment_ids = tokenizer.encode(text, maxlen=512)
    frozen_graph_predictions = frozen_func(input_ids=tf.constant([token_ids], dtype=tf.float32), input_segments=tf.constant([segment_ids], dtype=tf.float32))

    te = frozen_graph_predictions[0][0]    
    labels = np.argmax(te)

    return labels, te.numpy()


class test_cls_model():
    
    def __init__(self, save_pb_path, vocab_file='./ModelZoo/TF_Model/chinese_roberta_wwm_ext_L-12_H-768_A-12/vocab.txt') -> None:
        self.tokenizer = Tokenizer(vocab_file, do_lower_case=True)
        self.modelpath = save_pb_path
        self.loadmodel()

    def loadmodel(self):
        with tf.io.gfile.GFile(self.modelpath, "rb") as f:
            graph_def = tf.compat.v1.GraphDef()
            loaded = graph_def.ParseFromString(f.read())

        # Wrap frozen graph to ConcreteFunctions
        self.frozen_func = wrap_frozen_graph(graph_def=graph_def,
                                        inputs=["input_ids:0", "input_segments:0"],
                                        outputs=["Identity:0"],
                                        print_graph=False)
                                        
        print("Frozen model inputs: ")
        print(self.frozen_func.inputs)
        print("Frozen model outputs: ")
        print(self.frozen_func.outputs)

    def get_text_cls(self, text):
        token_ids, segment_ids = self.tokenizer.encode(text, maxlen=512)
        frozen_graph_predictions = self.frozen_func(input_ids=tf.constant([token_ids], dtype=tf.float32), input_segments=tf.constant([segment_ids], dtype=tf.float32))

        te = frozen_graph_predictions[0][0]    
        labels = np.argmax(te)

        return labels, te.numpy()