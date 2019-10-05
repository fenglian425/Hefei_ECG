from torch.utils.data import Dataset, DataLoader
from config import *
import os
import numpy as np
from tqdm import tqdm


class ECGDataset(Dataset):
    def __init__(self, list_ids, cfg: Config, use_cache=True):
        self.list_ids = list_ids
        self.cfg = cfg

        self.use_cache = use_cache
        self.cache = {}

    def __len__(self):
        return len(self.list_ids)

    def __getitem__(self, index):
        ecg_id = self.list_ids[index]

        if ecg_id not in self.cache:
            ecg = np.loadtxt(TRAIN_DIR + ecg_id, skiprows=1)
            anno = LABEL.loc[int(ecg_id.split('.')[0])]
            label = np.zeros((self.cfg.CLASS_NUM), dtype=np.longlong)
            for a in anno['arrythmia']:
                label[ARRYTHMIA_TO_LABEL[a]] = 1

            if self.use_cache:
                self.cache[ecg_id] = (ecg, label)
        else:
            ecg, label = self.cache[ecg_id]
        return ecg, label


def split_train_val(cfg: Config, seed=666):
    if cfg.DATASET == 'default':
        list_ids = sorted(os.listdir(TRAIN_DIR))
    rng = np.random.RandomState(seed)
    rng.shuffle(list_ids)

    fold = cfg.FOLD
    fold_num = cfg.FOLD_NUM

    step = len(list_ids) // fold_num
    val_ids = list_ids[step * fold:step * (fold + 1)]
    train_ids = list_ids[:step * fold] + list_ids[step * (fold + 1):]
    print(f'dataset mode:{cfg.DATASET}, train nums:{len(list_ids)}, val nums:{len(val_ids)}')

    return train_ids, val_ids


def get_loaders(cfg: Config):
    train_ids, val_ids = split_train_val(cfg, seed=666)

    train_loader = ECGDataset(train_ids, cfg)
    val_loader = ECGDataset(val_ids, cfg)
    return train_loader, val_loader


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    cfg = Config()
    train_ids, val_ids = split_train_val(cfg)

    train_ids = os.listdir(TRAIN_DIR)
    print(train_ids)
    print(val_ids)
    train_loader = ECGDataset(train_ids, cfg)

    LABEL_TO_ARRYTHMIA = np.array(LABEL_TO_ARRYTHMIA)
    res = []
    for ecg, label in tqdm(train_loader):
        print(ecg.max())
        # res.append(ecg)
    # res = np.array(res)
    # print(np.mean(res, axis=[0, 1]))
    # print(np.std(res, axis=[0, 1]))

    # plt.figure()
    # fig, axs = plt.subplots(8, 1, sharex=True)
    #
    # for i in range(8):
    #     axs[i].plot(ecg[:, i])
    #
    # plt.show()
    # print(LABEL_TO_ARRYTHMIA[label.astype(np.bool)])
    # break
    # pass
