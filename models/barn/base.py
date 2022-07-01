class baseModel():
    def __init__(self, config_path, dict_path, checkpoint_path=None, modeltype='bert', return_keras_model=False):
        # Input shape
        self.config_path = config_path
        self.dict_path = dict_path
        self.checkpoint_path = checkpoint_path
        self.modeltype = modeltype
        self.return_keras_model = return_keras_model

    def build_model(self):
        pass

    def train(self):
        pass

    def predict(self):
        pass

    def export_bert():
        pass

    def export_all_model():
        pass
