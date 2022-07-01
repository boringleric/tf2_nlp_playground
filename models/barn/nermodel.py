from models.barn.base import baseModel
from tqdm import tqdm
import numpy as np
from bert4keras.tokenizers import Tokenizer
from bert4keras.backend import K
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam
from bert4keras.layers import ConditionalRandomField
from bert4keras.snippets import sequence_padding, DataGenerator, ViterbiDecoder, to_array
from models.utils.adversarial_training import fgm_attack
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense, LSTM, Bidirectional
from tensorflow.keras.models import Model

import tensorflow
from pathlib import Path
import shutil, os 

class data_generator(DataGenerator):
    """
    数据生成器
    """
    def __init__(self, data, categories, batch_size=16, maxlen=64, tokenizer=None, buffer_size=None):
        super().__init__(data, batch_size, buffer_size)
        self.maxlen = maxlen
        self.tokenizer = tokenizer
        self.categories = categories

    def __iter__(self, random=True):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, d in self.sample(random):
            tokenlist = [self.tokenizer.tokenize(t)[1] for t in list(d[0])]
            tokenlist.append('[SEP]')
            newtokenlist = ['[CLS]']
            newtokenlist.extend(tokenlist)
            token_idslist = self.tokenizer.tokens_to_ids(newtokenlist)
            labelslist = np.zeros(len(token_idslist))
            mappinglist = self.tokenizer.rematch(d[0], newtokenlist)
            segment_ids = [0] * len(newtokenlist)
            start_mappinglist = {j[0]: i for i, j in enumerate(mappinglist) if j}
            end_mappinglist = {j[-1]: i for i, j in enumerate(mappinglist) if j}
            for start, end, label in d[1:]:
                if start in start_mappinglist and end in end_mappinglist:
                    start = start_mappinglist[start]
                    end = end_mappinglist[end]
                    labelslist[start] = self.categories.index(label) * 2 + 1
                    labelslist[start + 1:end + 1] = self.categories.index(label) * 2 + 2
            
            batch_token_ids.append(token_idslist)   # token_ids
            batch_segment_ids.append(segment_ids)
            batch_labels.append(labelslist)     # labels
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


class NamedEntityRecognizer(ViterbiDecoder):
    """命名实体识别器
    """
    def __init__(self, tokenizer, model, categories, trans, starts=None, ends=None):
        super().__init__(trans, starts, ends)
        self.tokenizer = tokenizer
        self.model = model
        self.categories = categories

    def recognize(self, text):
        tokens = self.tokenizer.tokenize(text, maxlen=512)
        mapping = self.tokenizer.rematch(text, tokens)
        token_ids = self.tokenizer.tokens_to_ids(tokens)
        segment_ids = [0] * len(token_ids)
        token_ids, segment_ids = to_array([token_ids], [segment_ids])
        nodes = self.model.predict([token_ids, segment_ids])[0]
        labels = self.decode(nodes)
        entities, starting = [], False
        for i, label in enumerate(labels):
            if label > 0:
                if label % 2 == 1:
                    starting = True
                    entities.append([[i], self.categories[(label - 1) // 2]])
                elif starting:
                    entities[-1][0].append(i)
                else:
                    starting = False
            else:
                starting = False
        return [(mapping[w[0]][0], mapping[w[-1]][-1], l) for w, l in entities]


class nerModel(baseModel):
    '''
    分类训练模型，可用于训练词向量以及分类模型
    '''
    def __init__(self, categories, config_path, dict_path, learning_rate=2e-5, checkpoint_path=None, modeltype='bert', return_keras_model=False):
        # Input shape
        self.config_path = config_path
        self.dict_path = dict_path 
        self.checkpoint_path = checkpoint_path
        self.modeltype = modeltype
        self.return_keras_model = return_keras_model
        self.learning_rate = learning_rate  
        self.categories = categories     
        
        # Build and compile the model
        self.tokenizer = Tokenizer(dict_path, do_lower_case=True)
        self.model, self.CRF, self.NER = self.build_model()
        optimizer = Adam(learning_rate)
        losses = self.CRF.sparse_loss

        self.model.compile(
                    loss=losses,
                    optimizer=optimizer,
                    metrics=[self.CRF.sparse_accuracy]
                    )


    def build_model(self, bert_layers=12, crf_lr_multiplier=100):

        bert = build_transformer_model(
                    config_path=self.config_path,
                    checkpoint_path=self.checkpoint_path,
                    model=self.modeltype,
                    return_keras_model=self.return_keras_model,
                )

        output_layer = 'Transformer-%s-FeedForward-Norm' % (bert_layers - 1)
        output = bert.model.get_layer(output_layer).output
        output = Bidirectional(LSTM(300, return_sequences=True, dropout=0.5))(output)
        output = Dense(len(self.categories) * 2 + 1)(output)
        CRF = ConditionalRandomField(lr_multiplier=crf_lr_multiplier)
        output = CRF(output)

        model = Model(bert.input, output)

        NER = NamedEntityRecognizer(self.tokenizer, model, self.categories, trans=K.eval(CRF.trans), starts=[0], ends=[0])

        return model, CRF, NER


    def train(self, data, epochs, batch_size=128, use_kf=False, n_splits=10, adv_train=False, checkpoint_path='./tmp_trashbin/tmp_ner.h5'):

        # 清空checkpoint文件夹
        tmp_path = Path("./tmp_trashbin")
        if tmp_path.is_dir():
            shutil.rmtree('./tmp_trashbin')
        os.mkdir('./tmp_trashbin')

        # checkpoint
        checkpoint = ModelCheckpoint(
                    filepath=checkpoint_path, 
                    monitor='val_sparse_accuracy', 
                    mode='auto', 
                    save_best_only=True, 
                    save_weights_only=True, 
                    verbose=1
                    ) 

        # Load the dataset
        all_data = data

        # warning: only tf1! tf2待查
        if adv_train and tensorflow.__version__<'2.0':
            fgm_attack(self.model, 'Embedding-Token', 0.5)

        # use kfold or not
        if use_kf:
            from sklearn.model_selection import KFold
            kf = KFold(n_splits=n_splits, random_state=42, shuffle=True)
        else:
            from sklearn.model_selection import train_test_split
            X_train, X_test = train_test_split(all_data, test_size=0.1, random_state=42, shuffle=True)
        
        if use_kf:
            for k, (train_idx, valid_idx) in enumerate(kf.split(all_data, all_data)):
                train_data = [all_data[idx] for idx in train_idx]
                valid_data = [all_data[idx] for idx in valid_idx]
                
                train_generator = data_generator(train_data, self.categories, batch_size, tokenizer=self.tokenizer)
                valid_generator = data_generator(valid_data, self.categories, batch_size, tokenizer=self.tokenizer)

                self.model.fit(
                    train_generator.forfit(),
                    steps_per_epoch=len(train_generator),
                    epochs=epochs,
                    validation_data=valid_generator.forfit(),
                    validation_steps=len(valid_generator),
                    callbacks=[checkpoint]
                )
        else:
            train_data = [X_train[idx] for idx, _ in enumerate(X_train)]
            valid_data = [X_test[idx] for idx, _ in enumerate(X_test)]

            train_generator = data_generator(train_data, self.categories, batch_size, tokenizer=self.tokenizer)
            valid_generator = data_generator(valid_data, self.categories, batch_size, tokenizer=self.tokenizer)

            self.model.fit(
                    train_generator.forfit(),
                    steps_per_epoch=len(train_generator),
                    epochs=epochs,
                    validation_data=valid_generator.forfit(),
                    validation_steps=len(valid_generator),
                    callbacks=[checkpoint],
                )


    def predict(self, testdata, checkpoint_path='./tmp_trashbin/tmp_ner.h5'):
        '''
        模型测试
        '''
        y_pred_list = []

        if checkpoint_path:
            self.model.load_weights(checkpoint_path)

        trans = K.eval(self.CRF.trans)
        self.NER.trans = trans
        y_pred = self.NER.recognize(testdata)

        for y in y_pred:
            y_pred_list.append(["".join(list(testdata)[y[0]:y[1]+1]), y[-1]])

        return y_pred_list


    def export_all_model(self, savepath, checkpoint_path='./tmp_trashbin/tmp_ner.h5'):
        '''
        导出全部模型。切记！eager模式下执行!
        '''
        from models.utils.keras_to_pb import save_as_pb
        import tensorflow as tf
        import warnings 

        if not tensorflow.executing_eagerly():
            warnings.warn('eager模式下执行！')

        else:
            # 加载模型
            self.model.load_weights(checkpoint_path)

            # 重新指定输入
            x_tensor_spec = tf.TensorSpec(shape=self.model.inputs[0].shape, dtype=self.model.inputs[0].dtype, name='input_ids')
            y_tensor_spec = tf.TensorSpec(shape=self.model.inputs[1].shape, dtype=self.model.inputs[1].dtype, name='input_segments')  

            # 保存pb
            save_as_pb(self.model, (x_tensor_spec, y_tensor_spec), savepath+'/ner_full_model.pb', print_graph=False)
