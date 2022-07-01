import tensorflow as tf
import sys
sys.modules['keras'] = tf.keras
import keras
import keras.backend as K

def variable_mapping(num_hidden_layers=12):
        """映射到官方BERT权重格式
        """
        mapping = {
            'Embedding-Token': ['bert/embeddings/word_embeddings'],
            'Embedding-Segment': ['bert/embeddings/token_type_embeddings'],
            'Embedding-Position': ['bert/embeddings/position_embeddings'],
            'Embedding-Norm': [
                'bert/embeddings/LayerNorm/beta',
                'bert/embeddings/LayerNorm/gamma',
            ],
            'Embedding-Mapping': [
                'bert/encoder/embedding_hidden_mapping_in/kernel',
                'bert/encoder/embedding_hidden_mapping_in/bias',
            ],
            'Pooler-Dense': [
                'bert/pooler/dense/kernel',
                'bert/pooler/dense/bias',
            ],
            'NSP-Proba': [
                'cls/seq_relationship/output_weights',
                'cls/seq_relationship/output_bias',
            ],
            'MLM-Dense': [
                'cls/predictions/transform/dense/kernel',
                'cls/predictions/transform/dense/bias',
            ],
            'MLM-Norm': [
                'cls/predictions/transform/LayerNorm/beta',
                'cls/predictions/transform/LayerNorm/gamma',
            ],
            'MLM-Bias': ['cls/predictions/output_bias'],
        }

        for i in range(num_hidden_layers):
            prefix = 'bert/encoder/layer_%d/' % i
            mapping.update({
                'Transformer-%d-MultiHeadSelfAttention' % i: [
                    prefix + 'attention/self/query/kernel',
                    prefix + 'attention/self/query/bias',
                    prefix + 'attention/self/key/kernel',
                    prefix + 'attention/self/key/bias',
                    prefix + 'attention/self/value/kernel',
                    prefix + 'attention/self/value/bias',
                    prefix + 'attention/output/dense/kernel',
                    prefix + 'attention/output/dense/bias',
                ],
                'Transformer-%d-MultiHeadSelfAttention-Norm' % i: [
                    prefix + 'attention/output/LayerNorm/beta',
                    prefix + 'attention/output/LayerNorm/gamma',
                ],
                'Transformer-%d-FeedForward' % i: [
                    prefix + 'intermediate/dense/kernel',
                    prefix + 'intermediate/dense/bias',
                    prefix + 'output/dense/kernel',
                    prefix + 'output/dense/bias',
                ],
                'Transformer-%d-FeedForward-Norm' % i: [
                    prefix + 'output/LayerNorm/beta',
                    prefix + 'output/LayerNorm/gamma',
                ],
            })

        return mapping

@property
def initializer():
    """默认使用截断正态分布初始化
    """
    return keras.initializers.TruncatedNormal(stddev=0.02)

def create_variable(name, value, dtype=None):
        """创建一个变量
        """
        dtype = dtype or K.floatx()
        return K.variable(
            initializer(value.shape, dtype), dtype, name=name
        ), value

def save_weights_as_checkpoint(model, filename, mapping=None, dtype=None):
    """根据mapping将权重保存为checkpoint格式
    """
    # print(tf.executing_eagerly())    
    mapping = mapping if mapping else variable_mapping()
    mapping = {k: v for k, v in mapping.items()}
    mapping = {k: v for k, v in mapping.items() if k in model.layers}

    # 使用eager方式读取value
    all_variables, all_values = [], []
    tmp, tmplist = [], []
    for layer, variables in mapping.items():
        layer = model.layers[layer]
        values = K.batch_get_value(layer.trainable_weights)
        for val in values:
            tmp.append(val)
        tmplist.append(values)
    
    
    # 使用tf1方式赋值
    if tf.__version__ > '2.0':
        tf.compat.v1.disable_eager_execution()
        tf.compat.v1.disable_v2_behavior()

    cnt = 0
    # print(tf.executing_eagerly())
    with tf.Graph().as_default():
        for layer, variables in mapping.items():
            tmpval = tmplist[cnt]
            layer = model.layers[layer]
            for name, value in zip(variables, tmpval):
                variable, value = model.create_variable(name, value, dtype)
                all_variables.append(variable)
                all_values.append(value)
            cnt += 1

        with tf.compat.v1.Session() as sess:
            K.batch_set_value(zip(all_variables, all_values))
            saver = tf.compat.v1.train.Saver()
            saver.save(sess, filename)
