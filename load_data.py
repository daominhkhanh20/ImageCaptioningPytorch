import os
import torch
from torchvision import transforms as T
from torch.utils.data import Dataset,DataLoader
from PIL import Image
import spacy
import pandas as pd
from collections import Counter
from torch.nn.utils.rnn import pad_sequence
from tensorflow.keras.preprocessing.text import Tokenizer

class Vocabulary:
    def __init__(self):
        self.size_vocab=7000
        self.tokenizer=Tokenizer(num_words=self.size_vocab,
                                filters='!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n',
                                lower=True,
                                oov_token='<unk>'
        )

    def __len__(self):
        return len(self.tokenizer.word_index)

    def build_vocab(self,captions_list):
        self.tokenizer.fit_on_texts(captions_list)
        self.sequences=self.tokenizer.texts_to_sequences(captions_list)
        self.tokenizer.word_index['<pad>']=0
        self.tokenizer.index_word[0]='<pad>'   
        self.stoi=self.tokenizer.word_index
        self.itos=self.tokenizer.index_word

    def numericalize(self,text):
        return self.tokenizer.texts_to_sequences([text])[0]


class Flickr8k(Dataset):
    def __init__(self,root_dir,caption_file,transforms=None):
        path=os.getcwd()
        self.transforms=transforms
        path_to_caption_file=os.path.join(path,root_dir+"/"+caption_file)
        df=pd.read_csv(path_to_caption_file,sep=',',skipinitialspace=True)
        self.images=df.image
        self.captions='<sos> '+df.caption+' <eos>'
        self.captions=self.captions.tolist()
        self.vocabulary=Vocabulary()
        self.vocabulary.build_vocab(self.captions)

    def __len__(self):
        return len(self.images)

    def __getitem__(self,idx):
        image_name=self.images[idx]
        path=os.path.join(os.getcwd(),"Flickr8k/Images/"+str(image_name))
        image=Image.open(path).convert("RGB")
        if self.transforms is not None:
            image=self.transforms(image)
        caption_vector=self.vocabulary.sequences[idx]
        return image,torch.tensor(caption_vector)



class MyCollate:
    def __init__(self,pad_idx):
        self.pad_idx=pad_idx

    def __call__(self,batch):
        #batch has a lot of pait (image,caption)      
        images=[item[0].unsqueeze(dim=0) for item in batch]
        captions=[item[1] for item in batch]
        images=torch.cat(images,dim=0)#batch_size*3*height*width
        captions=pad_sequence(captions,batch_first=True,padding_value=self.pad_idx)
        return images,captions


def get_dataset(root_dir,caption_file,transforms,batch_size=64,num_workers=2,shuffle=False,pin_memory=True):
    dataset=Flickr8k(root_dir=root_dir,
                    caption_file=caption_file,
                    transforms=transforms,
    )
    pad_idx=dataset.vocabulary.stoi['<pad>']

    data_loader=DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            shuffle=shuffle,
                            pin_memory=pin_memory,#if pin_memory=True, which enables fast data transfer to CUDA-enable GPUS
                            collate_fn=MyCollate(pad_idx)
    )
    return dataset,data_loader




# transforms=T.Compose([
#         T.Resize(256),
#         T.CenterCrop(224),
#         T.ToTensor(),
#         T.Normalize(mean=(0.485,0.456,0.406),std=(0.229,0.224,0.225))
#     ])
# dataset,loader=get_dataset(root_dir='Flickr8k',
#                             caption_file='captions.txt',
#                             transforms=transforms                            
# )

# for idx,(image,caption) in enumerate(loader):
#     print(caption.size())

