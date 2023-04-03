import scipy.io as io
import os
import pandas as pd
import numpy as np

patient = ['ZAB', 'ZDM', 'ZDN', 'ZJM', 'ZJN', 'ZJS', 'ZKH', 'ZKW', 'ZMG']
file_name = f"D:/programming_project/preprocessed/results{patient[0]}_SR.mat"

data = io.loadmat(file_name, squeeze_me=True, struct_as_record=False)['sentenceData']

count = 0

for i in range(len(data)):

    df = pd.DataFrame()
    sub_df = pd.DataFrame()
    word_df = pd.DataFrame()

    for j in range(len(data[i].word)):
        FFD_a1 = data[i].word[j].FFD_a1_diff[0:48]
        FFD_a2 = data[i].word[j].FFD_a2_diff[0:48]

        FFD_b1 = data[i].word[j].FFD_b1_diff[0:48]
        FFD_b2 = data[i].word[j].FFD_b2_diff[0:48]

        FFD_g1 = data[i].word[j].FFD_g1_diff[0:48]
        FFD_g2 = data[i].word[j].FFD_g2_diff[0:48]

        word = data[i].word[j].content
        if FFD_a1 == []:
            FFD_a1=np.array([])
        if FFD_a2 == []:
            FFD_a2=np.array([])
        if FFD_b1 == []:
            FFD_b1=np.array([])
        if FFD_b2 == []:
            FFD_b2=np.array([])
        if FFD_g1 == []:
            FFD_g1=np.array([])
        if FFD_g2 == []:
            FFD_g2=np.array([])
        arr = np.concatenate((FFD_a1, FFD_a2, FFD_b1, FFD_b2, FFD_g1,FFD_g2))
        if len(arr)==0:
            arr=np.zeros(288)
        elif np.isnan(arr).all():
            arr = np.zeros(288)
        arr = pd.DataFrame(arr).T
        word = pd.Series(word)
        word_df = word_df.append(word, ignore_index=True)
        df = df.append(arr, ignore_index=True)

    df = pd.concat([word_df, df], axis=1, ignore_index=True)

    df.to_csv(f'word_eeg/{patient[0]}{count}_word.csv', index=False)
    count += 1

e = []
for j in range(400):
    data = pd.read_csv(f'word_eeg/ZAB{j}_word.csv',index_col=0)
    data = data.values
    i = np.zeros((43-len(data),288))
    data = np.concatenate((data,i))
    e.append(data)
e = np.array(e)
np.save(f"{patient[0]}_eeg_word",e)
