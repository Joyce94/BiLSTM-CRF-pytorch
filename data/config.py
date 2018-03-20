from configparser import ConfigParser
import os

class Configurable(object):
    def __init__(self, config_file):
        config = ConfigParser()
        config.read(config_file)
        self._config = config

        print('Loaded config file sucessfully.')
        for section in config.sections():
            for k, v in config.items(section):
                print(k, v)
        if not os.path.isdir(self.save_dir):
            os.mkdir(self.save_dir)
        config.write(open(config_file, 'w'))        #####

    @property
    def pretrained_wordEmb_file(self):
        return self._config.get('Data', 'pretrained_wordEmb_file')
    @property
    def pretrained_charEmb_file(self):
        return self._config.get('Data', 'pretrained_charEmb_file')
    @property
    def data_dir(self):
        return self._config.get('Data', 'data_dir')
    @property
    def train_file(self):
        return self._config.get('Data', 'train_file')
    @property
    def dev_file(self):
        return self._config.get('Data', 'dev_file')
    @property
    def test_file(self):
        return self._config.get('Data', 'test_file')
    @property
    def shrink_feature_thresholds(self):
        return self._config.getint('Data', 'shrink_feature_thresholds')
    @property
    def run_insts(self):
        return self._config.getint('Data', 'run_insts')

    @property
    def save_dir(self):
        return self._config.get('Save', 'save_dir')
    @property
    def config_file(self):
        return self._config.get('Save', 'config_file')
    @property
    def save_model_path(self):
        return self._config.get('Save', 'save_model_path')
    @property
    def save_vocab_path(self):
        return self._config.get('Save', 'save_vocab_path')
    @property
    def load_dir(self):
        return self._config.get('Save', 'load_dir')
    @property
    def load_model_path(self):
        return self._config.get('Save', 'load_model_path')
    @property
    def load_vocab_path(self):
        return self._config.get('Save', 'load_vocab_path')
    @property
    def lstm_layers(self):
        return self._config.getint('Network', 'lstm_layers')
    @property
    def word_dims(self):
        return self._config.getint('Network', 'word_dims')

    @property
    def char_dims(self):
        return self._config.getint('Network', 'char_dims')
    @property
    def lstm_hiddens(self):
        return self._config.getint('Network', 'lstm_hiddens')
    @property
    def dropout_emb(self):
        return self._config.getfloat('Network', 'dropout_emb')
    @property
    def dropout_lstm(self):
        return self._config.getfloat('Network', 'dropout_lstm')
    @property
    def learning_rate(self):
        return self._config.getfloat('Optimizer', 'learning_rate')
    @property
    def decay(self):
        return self._config.getfloat('Optimizer', 'decay')
    @property
    def clip_grad(self):
        return self._config.getfloat('Optimizer', 'clip_grad')
    @property
    def thread_nums(self):
        return self._config.getint('Run', 'thread_nums')
    @property
    def maxIters(self):
        return self._config.getint('Run', 'maxIters')
    @property
    def train_batch_size(self):
        return self._config.getint('Run', 'train_batch_size')
    @property
    def test_batch_size(self):
        return self._config.getint('Run', 'test_batch_size')
    @property
    def validate_every(self):
        return self._config.getint('Run', 'validate_every')
    @property
    def save_after(self):
        return self._config.getint('Run', 'save_after')
    @property
    def update_every(self):
        return self._config.getint('Run', 'update_every')












