import pandas as pd
from PIL import Image
import torch
import numpy as np
from model_attention import CNN2RNN
from torchvision import transforms as T
import load_data
import pickle 
from bleu_score import final_bleu_score

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
test=pd.read_csv('Flickr8k/captions_test.txt')
with open(f'dataset.pickle','rb') as file:
  dataset=pickle.load(file)

state_dict=torch.load('model50.pth',map_location='cpu')
attention_dim=state_dict['attention_dim']
encoder_dim=2048
decoder_dim=state_dict['decoder_dim']
embedding_dim=state_dict['embedding_size']
vocab_size=state_dict['vocab_size']
model=CNN2RNN(attention_dim=512,decoder_dim=512,embedding_dim=256,vocab_size=vocab_size).to(device)
model.eval()
model.load_state_dict(state_dict['state_dict'])


def preprocessing(image_name):
    transform=T.Compose([
        T.Resize(226),
        T.RandomCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.2])
    ])
    image = Image.open("Image/Test/"+image_name)
    image=transform(image)
    return image

def get_list_caption(image_name):
    indxs=np.where(test.image==image_name)[0]
    result=[]
    for index in indxs:
        s=str(test.iloc[index]['caption'])
        #s=re.sub(r'[^\w\s]','',s)
        result.append(s.split())
    return result

def evaluate_bleu_score():
    bleu1,bleu2,bleu3,bleu4=0,0,0,0
    images=set(test.image)
    print(f"We have {len(images)} images")
    for image_name in images:
        image=preprocessing(image_name)
        image=image.to(device)
        caption=model.image_captions(image.unsqueeze(dim=0),dataset.vocabulary).split()
        list_caption=get_list_caption(image_name)
        bleu1+=final_bleu_score(list_caption,caption,weights=[1,0,0,0])
        bleu2+=final_bleu_score(list_caption,caption,weights=[0.5,0.5,0,0])
        bleu3+=final_bleu_score(list_caption,caption,weights=[0.333,0.333,0.333,0])
        bleu4+=final_bleu_score(list_caption,caption)

    print("Bleu for 1-gram:",bleu1*1.0/len(images))
    print("Bleu for 2-gram:",bleu3*1.0/len(images))
    print("Bleu for 3-gram:",bleu3*1.0/len(images))
    print("Bleu for 4-gram:",bleu4*1.0/len(images))

evaluate_bleu_score()