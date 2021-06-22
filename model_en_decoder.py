import torch
from torch import nn
from torchvision import models


class Encoder(nn.Module):
    def __init__(self,embedding_size,train_CNN=False,dropout=0.2):
        super(Encoder,self).__init__()
        self.embedding_size=embedding_size
        self.train_CNN=train_CNN
        self.model=models.inception_v3(pretrained=True,aux_logits=False)
        self.model.fc=nn.Linear(self.model.fc.in_features,embedding_size)
        self.dropout=nn.Dropout(dropout)
        self.relu=nn.ReLU()

    def forward(self,images):
        features=self.model(images)
        return self.dropout(self.relu(features)) #batch_size * embedding_size


class Decoder(nn.Module):
    def __init__(self,embedding_size,hidden_size,vocab_size,num_layers,dropout=0.2):
        super(Decoder,self).__init__()
        self.embedding_size=embedding_size
        self.hidden_size=hidden_size
        self.vocab_size=vocab_size

        self.embedded=nn.Embedding(vocab_size,embedding_size)#vocab_size *batch_size*embedding_size
        self.lstm=nn.LSTM(
                input_size=embedding_size,
                hidden_size=hidden_size,
                num_layers=num_layers
            )
        self.linear=nn.Linear(hidden_size,vocab_size)
        self.dropout=nn.Dropout(dropout)

    def forward(self,features,captions):
        """
        features:  batch_size*embeeding_size
        captions:  vocab_size*batch_size
        """
        features=features.unsqueeze(dim=0)
        #print("Features:",features.shape)
        captions=self.dropout(self.embedded(captions))#vocab_batch_size*batch_size*embedding_size
        #print("Captions:",captions.shape)
        embedding=torch.cat((features,captions),dim=0)
        #print("Embedding:",embedding.shape)
        #outputs: vocab_batch_size*batch_size*embedding_size
        outputs_encoder,(h_n,c_n)=self.lstm(embedding)
        #print("Output_encoder:",outputs_encoder.shape)
        outputs=self.linear(outputs_encoder)
        return outputs


class CNN2RNN(nn.Module):
    def __init__(self,embedding_size,hidden_size,vocab_size,num_layers):
        super(CNN2RNN,self).__init__()
        self.encoder=Encoder(embedding_size)
        self.decoder=Decoder(embedding_size,hidden_size,vocab_size,num_layers)

    def forward(self,images,captions):
        features=self.encoder(images)
        outputs=self.decoder(features,captions)
        return outputs

    def caption_image(self,image,vocabulary,max_length=20):
        caption_result=[]
        states=None

        with torch.no_grad():
            x=self.encoder(image).unsqueeze(dim=0)#1*embedding_size
            for _ in range(max_length):
                hiddens,states=self.decoder.lstm(x,states)
                #print(hiddens.shape)
                output=self.decoder.linear(hiddens.squeeze(dim=0))
                #print(output.shape)
                pred=output.argmax(dim=1)
                caption_result.append(pred.item())

                if vocabulary.itos[pred.item()]=="<EOS>":
                    break

                x=self.decoder.embedded(pred).unsqueeze(dim=0)#1*embedding_size

        caption=[vocabulary.itos[index] for index in caption_result]
        return " ".join([cap for cap in caption[1:-1]])

