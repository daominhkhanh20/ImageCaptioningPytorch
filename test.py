import torch
from torchvision import transforms as T
from matplotlib import pyplot as plt
import numpy as np
from model import Encoder,Decoder,image_caption_with_beam_search
import pickle
from random import randint
import argparse
from PIL import Image 
import os 
import pandas as pd 
import time
from load_data import Vocabulary,Flickr8k,MyCollate

test=pd.read_csv('Flickr8k/captions_test.csv',sep='|',skipinitialspace=True)
path=os.path.join(os.getcwd(),'Result')
if os.path.exists(path) is False:
    os.mkdir(path)

ap=argparse.ArgumentParser()
ap.add_argument('-i','--image',required=False,help="add image for testing")
args=ap.parse_args()

with open(f'dataset.pickle','rb') as file:
    dataset=pickle.load(file)


state_dict=torch.load('model40.pth',map_location=torch.device('cpu'))
attention_dim=state_dict['attention_dim']
decoder_dim=state_dict['decoder_dim']
embedding_dim=state_dict['embedding_size']
vocab_size=state_dict['vocab_size']
encoder=Encoder()
decoder=Decoder(attention_dim=attention_dim,decoder_dim=decoder_dim,embedding_dim=embedding_dim,vocab_size=vocab_size,vocabulary=dataset.vocabulary)
encoder.load_state_dict(state_dict['encoder_state_dict'])
decoder.load_state_dict(state_dict['decoder_state_dict'])
encoder.eval()
decoder.eval()


def preprocessing():
    transform=T.Compose([
        T.Resize(224),
        T.RandomCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.2])
    ])
    image = Image.open(args.image)
    image=transform(image)
    return image


def get_list_caption(image_name):
    indxs=np.where(test.image==image_name)[0]
    result=[]
    for index in indxs:
        s=str(test.iloc[index]['caption'])
        s=s.strip()
        result.append(s.split(' '))

    return result


def show_image(image,caption,caption_beam=None):
    #plt.figure(figsize=(10,10))
    image[0]=image[0]*0.229+0.485
    image[1]=image[1]*0.224+0.456
    image[2]=image[2]*0.225+0.406

    fig,ax=plt.subplots(1,1)
    image=image.permute(1,2,0).numpy()
    im=ax.imshow(image,interpolation='nearest')
    if caption_beam is not None:
        title='Greedy search: '+caption+'\nBeam seach: '+caption_beam
        print(f"Greedy Search: {caption}\nBeam Search: {caption_beam}")
    
    else:
        title='Greedy search: '+caption
        print(f"Greedy Search: {caption}")

    ax.set_title(title)

    s=str(args.image)
    name_file="result"+s[s.rfind('/')+1:]
    plt.savefig('Result/{}'.format(name_file))
    plt.show()
    print("save done:",name_file)


if __name__=="__main__":
    start_time=time.time()
    if args.image is not None:
        with torch.no_grad():
            image=preprocessing()
            feature=encoder(image.unsqueeze(dim=0))
            caption=decoder.image_captions(feature,dataset.vocabulary)
            caption_beam=image_caption_with_beam_search(decoder,feature,dataset.vocabulary,beam_size=3)
            print(f"Time process: {time.time()-start_time}s")
            #references=get_list_caption(args.image)
            #print("Bleu socre:",final_bleu_score(references,caption))
            caption=' '.join([word for word in caption])
            caption_beam=' '.join([word for word in caption_beam])
            show_image(image,caption,caption_beam)

#     # else:
#     #     """
#     #     * Basically iter() calls the __iter__ method on the iris loader which return
#     #     an iterator.

#     #     * next() then call the __next__ method on that iterator to get the first 
#     #     iteration. Run next() again will get the second item of the iterator
#     #     """
#     #     dataiter=iter(data_test)
#     #     image,_,image_name=next(dataiter)
#     #     caption=model.image_captions(image[0:1],dataset.vocabulary)
#     #     print(f"Time process: {time.time()-start_time}s")
#     #     show_image(image[0:1].squeeze(dim=0),caption)
