# -*- coding = utf-8 -*-
# @Time : 2022/5/27 15:41
# @Author : Tetsuya Chen
# @File : split_trantest.py
# @software : PyCharm
from sklearn.model_selection import train_test_split, KFold
import pandas as pd
import os


def split_train_test(data_csv, save_path, test_size = 0.1, k_fold = 10):
    print(data_csv)
    data_m = data_csv[data_csv.label == 1]
    data_b = data_csv[data_csv.label == 0]
    print(len(data_m))
    data_m_train, data_m_test = train_test_split(data_m, test_size=test_size, random_state=100)
    data_b_train, data_b_test = train_test_split(data_b, test_size=test_size, random_state=100)

    kf = KFold(n_splits=k_fold, shuffle=True, random_state=90)

    # data_m_train, data_m_valid = train_test_split(data_m_train, test_size=test_size/(1-test_size), random_state=100)
    # data_b_train, data_b_valid = train_test_split(data_b_train, test_size=test_size/(1-test_size), random_state=100)

    # data_train = pd.concat([data_m_train, data_b_train])
    # data_valid = pd.concat([data_m_valid, data_b_valid])
    # data_train.to_csv(os.path.join(save_path, 'train.csv'), index=0)
    # data_valid.to_csv(os.path.join(save_path, 'valid.csv'), index=0)
    index_m_t, index_m_v, index_b_t, index_b_v = [], [], [], [],

    for t,v in kf.split(data_m_train):
        index_m_t.append(t)
        index_m_v.append(v)
    for t,v in kf.split(data_b_train):
        index_b_t.append(t)
        index_b_v.append(v)

    for i in range(k_fold):
        data_m_t1 = data_m_train.iloc[[x for x in index_m_t[i]]]
        data_b_t1 = data_b_train.iloc[[x for x in index_b_t[i]]]
        data_m_v1 = data_m_train.iloc[[x for x in index_m_v[i]]]
        data_b_v1 = data_b_train.iloc[[x for x in index_b_v[i]]]

        data_train = pd.concat([data_m_t1, data_b_t1])
        data_valid = pd.concat([data_m_v1, data_b_v1])

        data_train.to_csv(os.path.join(save_path, f'train_{i}.csv'), index=0)
        data_valid.to_csv(os.path.join(save_path, f'valid_{i}.csv'), index=0)

    data_test = pd.concat([data_m_test, data_b_test])
    data_test.to_csv(os.path.join(save_path, 'test.csv'), index=0)

def sample_df(df, rate):
    df_ = df[::rate].reset_index(drop=True)

    return (df_)

def stack_df(df, img_floder):
    df_ = pd.DataFrame(columns=['id','path','label'])
    for i in range(len(df)):
        id = df['id'].apply(lambda x: str('{:0>5d}'.format(x)))[i]
        label = df['label'][i]
        try:
            img_dir = os.path.join(img_floder, str(id), 'img_stack/30/crop/')

            for stack in os.listdir(img_dir):
                stack_path = os.path.join(img_dir, stack)
                df_.loc[len(df_)] = [id, stack_path, label]
        except:
            pass
    return df_

if __name__ == '__main__':
    # data_file = "/home/chenpeizhe/Dataset/dongyang/group_Label.csv"
    # data_csv = pd.read_csv(data_file)
    # data_csv['id'] = data_csv['id'].apply(lambda x: str('{:0>5d}'.format(x)))
    # save_path = "/home/chenpeizhe/Dataset/dongyang/group_train_test/"
    # os.makedirs(save_path, exist_ok=True)
    # split_train_test(data_csv, save_path)

    # df = pd.read_csv("/home/chenpeizhe/Dataset/dongyang/img_train_test/train.csv")
    # sample_df(df,5)

    # df = pd.read_csv("/home/chenpeizhe/Dataset/dongyang/group_train_test/valid.csv")
    # img_folder = '/home/chenpeizhe/Dataset/dongyang/dongyang_clips_preprocess800/'
    # df_ = stack_df(df, img_folder)
    # df_.to_csv('/home/chenpeizhe/Dataset/dongyang/stack/valid.csv', index=0)

    df = pd.read_csv("/home/chenpeizhe/Dataset/dongyang_plus/Label.csv")
    save_path = '/home/chenpeizhe/Dataset/dongyang_plus/data_split/img/10folds_re'
    os.makedirs(save_path, exist_ok=True)
    split_train_test(df, save_path)