import pandas as pd
import os
import random
import numpy as np
from string import punctuation

def strip_punctuation(dataframe):
    for i in range(len(dataframe)):
        caption=dataframe.caption.loc[i].strip(punctuation)
        dataframe.caption.loc[i]=caption
    caption_length=[]
    for caption in dataframe.caption:
        caption_length.append(len(caption.split(' ')))

    dataframe['length']=caption_length 
    image_deleted=set(dataframe[dataframe.length<6].image)
    dataframe=dataframe[~dataframe.image.isin(image_deleted)]
    dataframe.drop(['length'],axis=1,inplace=True)
    return dataframe

def split_captions_file():
    data=pd.read_csv('Flickr8k/captions.txt',sep=';')
    data_test=data[:10000]
    data_train=data[10000:]
    data_train.index=range(len(data_train))
    data_train=strip_punctuation(data_train)
    data_test=strip_punctuation(data_test)
    data_train.to_csv('Flickr8k/captions_train.txt',sep=';',index=False)
    data_test.to_csv('Flickr8k/captions_test.txt',sep=';',index=False)
    return data_test

def rename_image(data_test):
    path_ori=os.getcwd()
    path=os.path.join(path_ori,'Image')
    if os.path.exists(path) is False:
        os.mkdir(path)
    path=os.path.join(path,'Test')
    if os.path.exists(path) is False:
        os.mkdir(path)

    images=set(data_test.image)
    for idx,image in enumerate(images):
        try:
            os.rename(path_ori+"/Flickr8k/Images/"+image,path+"/{}.jpg".format(idx))
            indexs=np.where(data_test.image==image)[0]
            data_test.iloc[indexs,0]="{}.jpg".format(idx)
        except:
            print(image)

    data_test.to_csv('Flickr8k/captions_test.txt',index=False)


def split():
    data_test=split_captions_file()
    #rename_image(data_test)


split()