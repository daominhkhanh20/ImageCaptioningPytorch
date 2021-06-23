import pandas as pd
import os
import random
import numpy as np
from string import punctuation
import re 
contractions_dict = { "ain’t": "are not", "’s":" is", "aren’t": "are not", "can’t": "cannot", "can’t’ve": "cannot have", "‘cause": "because", "could’ve": "could have", "couldn’t": "could not", "couldn’t’ve": "could not have", "didn’t": "did not", "doesn’t": "does not", "don’t": "do not", "hadn’t": "had not", "hadn’t’ve": "had not have", "hasn’t": "has not", "haven’t": "have not", "he’d": "he would", "he’d’ve": "he would have", "he’ll": "he will", "he’ll’ve": "he will have", "how’d": "how did", "how’d’y": "how do you", "how’ll": "how will", "I’d": "I would", "I’d’ve": "I would have", "I’ll": "I will", "I’ll’ve": "I will have", "I’m": "I am", "I’ve": "I have", "isn’t": "is not", "it’d": "it would", "it’d’ve": "it would have", "it’ll": "it will", "it’ll’ve": "it will have", "let’s": "let us", "ma’am": "madam", "mayn’t": "may not", "might’ve": "might have", "mightn’t": "might not", "mightn’t’ve": "might not have", "must’ve": "must have", "mustn’t": "must not", "mustn’t’ve": "must not have", "needn’t": "need not", "needn’t’ve": "need not have", "o’clock": "of the clock", "oughtn’t": "ought not", "oughtn’t’ve": "ought not have", "shan’t": "shall not", "sha’n’t": "shall not", "shan’t’ve": "shall not have", "she’d": "she would", "she’d’ve": "she would have", "she’ll": "she will", "she’ll’ve": "she will have", "should’ve": "should have", "shouldn’t": "should not", "shouldn’t’ve": "should not have", "so’ve": "so have", "that’d": "that would", "that’d’ve": "that would have", "there’d": "there would", "there’d’ve": "there would have", "they’d": "they would", "they’d’ve": "they would have","they’ll": "they will",
 "they’ll’ve": "they will have", "they’re": "they are", "they’ve": "they have", "to’ve": "to have", "wasn’t": "was not", "we’d": "we would", "we’d’ve": "we would have", "we’ll": "we will", "we’ll’ve": "we will have", "we’re": "we are", "we’ve": "we have", "weren’t": "were not","what’ll": "what will", "what’ll’ve": "what will have", "what’re": "what are", "what’ve": "what have", "when’ve": "when have", "where’d": "where did", "where’ve": "where have",
 "who’ll": "who will", "who’ll’ve": "who will have", "who’ve": "who have", "why’ve": "why have", "will’ve": "will have", "won’t": "will not", "won’t’ve": "will not have", "would’ve": "would have", "wouldn’t": "would not", "wouldn’t’ve": "would not have", "y’all": "you all", "y’all’d": "you all would", "y’all’d’ve": "you all would have", "y’all’re": "you all are", "y’all’ve": "you all have", "you’d": "you would", "you’d’ve": "you would have", "you’ll": "you will", "you’ll’ve": "you will have", "you’re": "you are", "you’ve": "you have"}
contractions_re = re.compile('(%s)'%'|'.join(contractions_dict.keys()))
def expand_contractions(s, contractions_dict=contractions_dict):
    def replace(match):
        return contractions_dict[match.group(0)]
    return contractions_re.sub(replace, s)


def strip_punctuation(dataframe):
    dataframe.caption=dataframe.caption.apply(lambda x:expand_contractions(x,contractions_dict))
    table=str.maketrans('','',punctuation)
    dataframe.caption=dataframe.caption.apply(lambda x:x.translate(table))
    dataframe.caption=dataframe.caption.apply(lambda x:re.sub("\s\s+"," ",x.lower()))

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