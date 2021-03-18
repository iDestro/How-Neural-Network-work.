class Config(object):
    def __init__(self):
        self.path = './data/jaychou_lyrics.txt'
        self.batch_size = 32
        self.num_steps = 35
        self.num_epochs = 200
        self.lr = 0.01
