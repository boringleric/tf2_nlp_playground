from models.barn.base import baseModel
from tqdm import tqdm
from bert4keras.tokenizers import Tokenizer
from bert4keras.backend import keras
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from models.utils.adversarial_training import fgm_attack
from tensorflow.keras.layers import Dense, Lambda
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Model
import tensorflow
from pathlib import Path
import shutil, os 

class data_generator(DataGenerator):
    """
    数据生成器
    """
    def __init__(self, data, batch_size=16, maxlen=64, tokenizer=None, buffer_size=None):
        super().__init__(data, batch_size, buffer_size)
        self.maxlen = maxlen
        self.tokenizer = tokenizer

    def __iter__(self, random=True):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text, label) in self.sample(random):
            token_ids, segment_ids = self.tokenizer.encode(text, maxlen=self.maxlen)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([label])
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


class clsModel(baseModel):
    '''
    分类训练模型，可用于训练词向量以及分类模型
    '''
    def __init__(self, num_classes, config_path, dict_path, learning_rate=2e-5, checkpoint_path=None, modeltype='bert', return_keras_model=False):
        # Input shape
        self.config_path = config_path
        self.dict_path = dict_path
        self.checkpoint_path = checkpoint_path
        self.modeltype = modeltype
        self.return_keras_model = return_keras_model
        self.num_classes = num_classes
        self.learning_rate = learning_rate

        optimizer = Adam(learning_rate)
        losses = ['sparse_categorical_crossentropy']

        # Build and compile the model
        self.tokenizer = Tokenizer(dict_path, do_lower_case=True)
        self.model, self.bert = self.build_model()
        self.model.compile(
                    loss=losses,
                    optimizer=optimizer,
                    metrics=['sparse_categorical_accuracy']
                    )


    def build_model(self):

        bert = build_transformer_model(
                    config_path=self.config_path,
                    checkpoint_path=self.checkpoint_path,
                    model=self.modeltype,
                    return_keras_model=self.return_keras_model,
                )

        output = Lambda(lambda x: x[:, 0])(bert.model.output)
        output = Dense(units=self.num_classes, activation='softmax', kernel_initializer=bert.initializer)(output)

        model = keras.models.Model(bert.model.input, output)

        return model, bert


    def train(self, data, label_dict, epochs, batch_size=128, use_kf=False, n_splits=10, adv_train=False, checkpoint_path='./tmp_trashbin/tmp_cls.h5'):

        # 清空checkpoint文件夹
        tmp_path = Path("./tmp_trashbin")
        if tmp_path.is_dir():
            shutil.rmtree('./tmp_trashbin')
        os.mkdir('./tmp_trashbin')  

        # checkpoint
        checkpoint = ModelCheckpoint(
                    filepath=checkpoint_path, 
                    monitor='val_sparse_categorical_accuracy', 
                    mode='auto', 
                    save_best_only=True, 
                    save_weights_only=True, 
                    verbose=1
                    ) 

        # Load the dataset
        all_data, all_labels = data

        # warning: only tf1! tf2待查
        if adv_train and tensorflow.__version__<'2.0':
            fgm_attack(self.model, 'Embedding-Token', 0.5)

        # use kfold or not
        if use_kf:
            from sklearn.model_selection import KFold
            kf = KFold(n_splits=n_splits, random_state=42, shuffle=True)
        else:
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(all_data, all_labels, test_size=0.1, random_state=42, shuffle=True)
        
        if use_kf:
            for k, (train_idx, valid_idx) in enumerate(kf.split(all_data, all_data)):
                train_data = [[all_data[idx], label_dict[all_labels[idx]]] for idx in train_idx]
                valid_data = [[all_data[idx], label_dict[all_labels[idx]]] for idx in valid_idx]
                
                train_generator = data_generator(train_data, batch_size, tokenizer=self.tokenizer)
                valid_generator = data_generator(valid_data, batch_size, tokenizer=self.tokenizer)

                self.model.fit(
                    train_generator.forfit(),
                    steps_per_epoch=len(train_generator),
                    epochs=epochs,
                    validation_data=valid_generator.forfit(),
                    validation_steps=len(valid_generator),
                    callbacks=[checkpoint]
                )
        else:
            train_data = [[X_train[idx], label_dict[y_train[idx]]] for idx, _ in enumerate(X_train)]
            valid_data = [[X_test[idx], label_dict[y_test[idx]]] for idx, _ in enumerate(X_test)]

            train_generator = data_generator(train_data, batch_size, tokenizer=self.tokenizer)
            valid_generator = data_generator(valid_data, batch_size, tokenizer=self.tokenizer)

            self.model.fit(
                    train_generator.forfit(),
                    steps_per_epoch=len(train_generator),
                    epochs=epochs,
                    validation_data=valid_generator.forfit(),
                    validation_steps=len(valid_generator),
                    callbacks=[checkpoint],
                )


    def predict(self, testdata, checkpoint_path='./tmp_trashbin/tmp_cls.h5', batch_size=32):
        '''
        模型测试
        '''
        y_pred_list = []

        if checkpoint_path:
            self.model.load_weights(checkpoint_path)

        test_generator = data_generator(testdata, batch_size, tokenizer=self.tokenizer)
        for x_true in tqdm(test_generator):
            y_pred = self.model.predict(x_true).argmax(axis=1)
            y_pred_list.append(y_pred)
        return y_pred_list


    def get_layer_embed(self, text, bert_layers=11, checkpoint_path='./tmp_trashbin/tmp_cls.h5'):
        import numpy as np
        
        if checkpoint_path:
            self.model.load_weights(checkpoint_path)

        #取某一层的输出为输出新建为model，采用函数模型
        output_layer = 'Transformer-%s-FeedForward-Norm' % (bert_layers - 1)

        dense1_layer_model = Model(inputs=self.model.input, outputs=self.model.get_layer(output_layer).output)
        #以这个model的预测值作为输出
    
        test_generator = data_generator([text], 1, tokenizer=self.tokenizer)
        for x_true in tqdm(test_generator):
            output = dense1_layer_model.predict(x_true)

        outmean = np.mean(output, axis=1)

        return outmean


    def export_bert(self, savepath, configpath, checkpoint_path='./tmp_trashbin/tmp_cls.h5'):
        '''
        只导出bert，需要关闭eager模式！关了本次kernel除非重启，就再也开不了eager模式
        '''
        # 清空savepath的tmp_bert文件夹
        tmp_path = Path(savepath+"/tmp_bert")
        if tmp_path.is_dir():
            shutil.rmtree(savepath+"/tmp_bert")
        os.mkdir(savepath+"/tmp_bert")  

        #print(tensorflow.executing_eagerly())
        # 加载模型
        self.model.load_weights(checkpoint_path)

        # 修改bert4keras的save_weights_as_checkpoint的代码
        from models.utils.save_bert.save_to_bert import save_weights_as_checkpoint
       
        # 另存为bert模型
        save_weights_as_checkpoint(self.bert, filename=savepath+'/tmp_bert/bert_model.ckpt')
        # 迁移config
        dirpath = os.path.dirname(configpath)
        shutil.copy(dirpath + '/vocab.txt', savepath+'/tmp_bert')
        shutil.copy(dirpath + '/bert_config.json', savepath+'/tmp_bert')
        # 另存为pb

        # 采用bert_as_service的代码，感谢
        from models.utils.save_bert.bert_trans_to_pb import args, PoolingStrategy, optimize_graph
        arg = args(
                pooling_layer=[-2], 
                model_dir=savepath+'/tmp_bert', 
                graph_tmp_dir=savepath, 
                pooling_strategy=PoolingStrategy.REDUCE_MEAN, 
                pbname='cls_bert_model'
                )

        tmp_file, bert_config = optimize_graph(arg)
        return tmp_file, bert_config


    def export_all_model(self, savepath, checkpoint_path='./tmp_trashbin/tmp_cls.h5'):
        '''
        导出全部模型。切记！eager模式下执行！必须在export_bert之前调用！
        '''
        from models.utils.keras_to_pb import save_as_pb
        import tensorflow as tf
        import warnings 

        if not tensorflow.executing_eagerly():
            warnings.warn('eager模式下执行！必须在export_bert之前调用！')

        else:
            # 加载模型
            self.model.load_weights(checkpoint_path)

            # 重新指定输入
            x_tensor_spec = tf.TensorSpec(shape=self.model.inputs[0].shape, dtype=self.model.inputs[0].dtype, name='input_ids')
            y_tensor_spec = tf.TensorSpec(shape=self.model.inputs[1].shape, dtype=self.model.inputs[1].dtype, name='input_segments')  

            # 保存pb
            save_as_pb(self.model, (x_tensor_spec, y_tensor_spec), savepath+'/cls_full_model.pb', print_graph=False)

