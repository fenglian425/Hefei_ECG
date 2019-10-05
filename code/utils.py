import pandas as pd
import numpy as np


# https://tianchi.aliyun.com/notebook-ai/detail?spm=5176.12586969.1002.3.2f003c3ayfY9Fj&postId=74096
def sf_age(x):
    try:
        t = int(x)
    except:
        t = -1
    return t


def sf_gender(x):
    if x == 'FEMALE':
        t = 0
    elif x == 'MALE':
        t = 1
    else:
        t = -1
    return t


def load_and_clean_sub(path):
    labels = pd.read_csv(path, header=None, sep='.txt', encoding='utf-8', engine='python')
    res = pd.DataFrame()
    res['id'] = labels[0]

    tmp = []
    for x in labels[1].values:
        try:
            t = x.split('\t')
            tmp.append([t[1], t[2]])
        except:
            tmp.append([np.nan, np.nan])
    tmp = np.array(tmp)
    res['age'] = tmp[:, 0]
    res['gender'] = tmp[:, 1]

    res['age'] = res['age'].apply(sf_age)
    res['gender'] = res['gender'].apply(sf_gender)
    return res


def load_and_clean_label(path):
    labels = pd.read_csv(path, header=None, sep='.txt', encoding='utf-8', engine='python')
    res = pd.DataFrame()
    res['id'] = labels[0]

    tmp = []
    for x in labels[1].values:
        t = x.split('\t')
        tmp.append([t[1], t[2], t[3:]])
    tmp = np.array(tmp)
    res['age'] = tmp[:, 0]
    res['gender'] = tmp[:, 1]
    res['arrythmia'] = tmp[:, 2]

    res['age'] = res['age'].apply(sf_age)
    res['gender'] = res['gender'].apply(sf_gender)

    res.set_index('id', inplace=True)
    return res


if __name__ == '__main__':
    from config import *

    # 训练数据的id、age、gender、arrythmia
    label = load_and_clean_label(LABEL_TXT)
    for arr in LABEL_TO_ARRYTHMIA:
        label[arr] = label['arrythmia'].apply(lambda x: arr in x)
    print(label.head())
    print(label.corr())
    label.corr().to_csv('corr.csv')
    arrythmia = label['arrythmia']

    select = ['窦性心动过速', '窦性心律不齐', '窦性心动过缓', '窦性心律', '慢心室率', '左心室高电压', '快室率心房颤动', '快心室率', '心房颤动']

    select = ['窦性心动过缓', '窦性心律', '窦性心律不齐']

    # print(*[a for a in arrythmia if '窦性心律不齐' in a], sep='\n')

    arr_count = {a: 0 for a in LABEL_TO_ARRYTHMIA}
    for a in arrythmia:
        res = 0
        for aa in a:
            arr_count[aa] += 1
        for s in select:
            res += s in a
        if res > 1:
            # print(a)
            # print('wrong')
            # break
            pass
    else:
        print('okay')
    # print(set([a[0] for a in arrythmia]))
    # print(*arr_count.items(), sep='\n')
