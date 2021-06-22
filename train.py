import pickle
import time

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from PIL import Image
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms as T

from bleu_score import final_bleu_score
from model import Decoder, Encoder
from load_data import MyCollate, get_dataset
from split_data  import split
split()
transforms=T.Compose([
        T.Resize(226),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=(0.485,0.456,0.406),std=(0.229,0.224,0.225))
    ])
dataset,data_loader=get_dataset(
            root_dir="Flickr8k",
            caption_file="captions_train.txt",
            transforms=transforms
    )

def convert_data_loader(loader):
    pad_idx=dataset.vocabulary.stoi["<PAD>"]
    return DataLoader(
            dataset=loader,
            batch_size=16,#how many sample which load to the batch
            shuffle=True,
            num_workers=2,#how many subprocess to uses for data loading
            pin_memory=True,
            collate_fn=MyCollate(pad_idx)#merge list of sample to form a mini batch tensor
        )

n=len(data_loader.dataset)
train_size=int(n*0.75)
val_size=int(n*0.2)
test_size=n-train_size-val_size
train,val,test=random_split(data_loader.dataset,[train_size,val_size,test_size])
data_train=convert_data_loader(train)
data_val=convert_data_loader(val)
data_test=convert_data_loader(test)
print(f"we have {len(dataset.vocabulary)} words")
with open('/content/drive/MyDrive/Attention3/dataset.pickle','wb') as file:
    pickle.dump(dataset,file,protocol=pickle.HIGHEST_PROTOCOL)

torch.save(data_test,"/content/drive/MyDrive/Attention1/data_test.pt")
torch.save(data_train,"/content/drive/MyDrive/Attention1/data_train.pt")
torch.save(data_val,"/content/drive/MyDrive/Attention1/data_val.pt")

# #hyparameters
vocab_size=len(dataset.vocabulary)
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
learning_rate=1e-4
epochs=300
attention_dim=512
decoder_dim=512
embedding_dim=100


encoder=Encoder().to(device)
encoder_optim=optim.RMSprop(encoder.parameters(),lr=1e-3)
decoder=Decoder(attention_dim=attention_dim,decoder_dim=decoder_dim,embedding_dim=embedding_dim,vocab_size=vocab_size,vocabulary=dataset.vocabulary).to(device)
decoder_optim=optim.RMSprop(decoder.parameters(),lr=4e-3)
criterion=nn.CrossEntropyLoss()


def save_loss(loss_train,loss_val,epoch):
    train=np.asarray(loss_train)
    val=np.asarray(loss_val)
    np.save('/content/drive/MyDrive/Attention1/train{}.npy'.format(epoch),train)
    np.save('/content/drive/MyDrive/Attention1/val{}.npy'.format(epoch),val)
    print("save loss done")


def preprocessing(image_name):
    image = Image.open("Flickr8k/Images/"+image_name)
    image=transforms(image)
    return image

def get_list_caption(image_name):
    vocabulary=dataset.vocabulary
    indxs=np.where(test.image==image_name)[0]
    result=[]
    for index in indxs:
        s=str(test.iloc[index]['caption'])
        t=vocabulary.numericalize(s)
        result.append([vocabulary.tokenizer.index_word[i] for i in t])

    return result

def evaluate_bleu_score_test_set(epoch):
    test=pd.read_csv('Flickr8k/captions_test.txt')
    bleu1,bleu2,bleu3,bleu4=0,0,0,0
    images=set(test.image)
    file=open('/content/drive/MyDrive/Attention4/caption_result{}.txt'.format(epoch),'w')
    for image_name in images:
        image=preprocessing(image_name)
        image=image.to(device)
        feature=encoder(image.unsqueeze(dim=0))
        caption=decoder.image_captions(feature,dataset.vocabulary)
        cap=' '.join([word for word in caption])
        file.write(f'{image_name},{cap}\n')
        list_caption=get_list_caption(image_name)
        bleu1+=final_bleu_score(list_caption,caption,weights=(1,0,0,0))
        bleu2+=final_bleu_score(list_caption,caption,weights=(0.5,0.5,0,0))
        bleu3+=final_bleu_score(list_caption,caption,weights=(0.333,0.333,0.333,0))
        bleu4+=final_bleu_score(list_caption,caption,weights=(0.25,0.25,0.25,0.25))
    file.close()

    print("Bleu for 1-gram:",bleu1*1.0/len(images))
    print("Bleu for 2-gram:",bleu3*1.0/len(images))
    print("Bleu for 3-gram:",bleu3*1.0/len(images))
    print("Bleu for 4-gram:",bleu4*1.0/len(images))

def save_model(epoch):
    model_state={
        'epochs':epochs,
        'current_epoch':epoch,
        'attention_dim':attention_dim,
        'decoder_dim':decoder_dim,
        'embedding_size':embedding_dim,
        'vocab_size':vocab_size,
        'encoder_state_dict':encoder.state_dict(),
        'decoder_state_dict':decoder.state_dict()
    }
    torch.save(model_state,"/content/drive/MyDrive/Attention3/model{}.pth".format(epoch))
    print("save model done")

def evaluate(epoch):
    loss_total=0
    for idx,(images,captions) in enumerate(data_val):
        images=images.to(device)
        captions=captions.to(device)
        features=encoder(images)
        outputs=decoder(features,captions)
        loss=criterion(outputs.reshape(-1,vocab_size),captions[:,1:].reshape(-1))
        loss_total+=loss.item()
    return loss_total/len(data_val)

def show_image(image,caption,epoch):
    image[0]=image[0]*0.229+0.485
    image[1]=image[1]*0.224+0.456
    image[2]=image[2]*0.225+0.406
    image=image.cpu().numpy().transpose((1,2,0))
    fig,ax=plt.subplots(1,1)
    img=ax.imshow(image,interpolation='nearest')
    if caption is not None:
        ax.set_title(caption)
    plt.savefig('/content/drive/MyDrive/Attention3/image{}.png'.format(epoch))
    print("save image done")

def main():
    loss_train=[]
    loss_val=[]
    encoder.train()
    decoder.train()
    for epoch in range(0,epochs+1):
        start_time=time.time()
        for idx,(images,captions) in enumerate(data_train):
            images=images.to(device)
            captions=captions.to(device)
            features=encoder(images)
            outputs=decoder(features,captions)
            targets=captions[:,1:]
            loss=criterion(outputs.reshape(-1,vocab_size),targets.reshape(-1))
            encoder_optim.zero_grad()
            decoder_optim.zero_grad()
            loss.backward()
            encoder_optim.step()
            decoder_optim.step()

        loss_train.append(loss.item())
        print("Epoch:{}----Train:{}----Run:{}s".format(epoch,loss.item(),start_time-time.time()))

        if epoch %10==0:
            encoder.eval()
            decoder.eval()
            save_loss(loss_train,loss_val,epoch)
            save_model(epoch)
            evaluate_bleu_score_test_set()  
            encoder.train()
            decoder.train()    
            # dataiter=iter(data_test)
            # image,caption_ori=next(dataiter)
            # caption=model.image_captions_greedy(image[0:1].to(device),dataset.vocabulary)
            # show_image(image[0:1].squeeze(dim=0),caption,epoch)




if __name__=="__main__":
    main()
