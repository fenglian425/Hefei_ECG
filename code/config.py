from utils import load_and_clean_label

DATA_DIR = '../data/'
TRAIN_DIR = '../data/train/'
TESTA_DIR = '../data/testA/'
LABEL_TXT = '../data/hf_round1_label.txt'
MODEL_DIR = '../weights'

with open('../data/hf_round1_arrythmia.txt', 'r') as file:
    arrylist = file.readlines()
LABEL_TO_ARRYTHMIA = [x.split()[0] for x in arrylist]
print(LABEL_TO_ARRYTHMIA)
ARRYTHMIA_TO_LABEL = dict([[x, i] for i, x in enumerate(LABEL_TO_ARRYTHMIA)])
print(ARRYTHMIA_TO_LABEL)
LABEL = load_and_clean_label(LABEL_TXT)


# print(len(LABEL_TO_ARRYTHMIA))

# print(arrythmia)
class Config():
    NAME = 'base'

    # dataset params
    DATASET = 'default'
    FOLD = 0
    FOLD_NUM = 10
    CLASS_NUM = 55

    # model params
    BACKBONE = 'resnet50'
    LOSS = 'bce'

    # train params
    IMG_PER_GPU = 256
    OPTIMIZER = 'sgd'
    BASE_LR = 1e-2
    SCHEDULER = 'multstep'
    ACCUMULATION_STEPS = 4
    EPOCHS = 200

    def get_snapshot(self):
        snapshot = {}
        for name in dir(self):
            value = getattr(self, name)
            if not name.startswith('__') and not callable(value):
                snapshot[name] = value
        return snapshot
