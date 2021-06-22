import torch
from torch import nn
from torchvision import models
from torch.nn import functional
from torch.autograd import Variable
import numpy as np 
import numpy as np
from collections import Counter
from torch.nn import functional as F
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
embedding_dict=np.load('glove.840B.300d.pkl',allow_pickle=True)
class Encoder(nn.Module):
    def __init__(self,units=512,encode_image_size=8):
        super(Encoder,self).__init__()
        self.resnet=models.resnet101(pretrained=True)
        for param in self.resnet.parameters():
            param.requires_grad=False

        modules=list(self.resnet.children())[:-2]
        self.resnet=nn.Sequential(*modules)
        self.adap=nn.AdaptiveAvgPool2d((encode_image_size,encode_image_size))

    def forward(self,images):
        features=self.resnet(images)
        features=self.adap(features)
        features=features.permute(0,2,3,1)
        features=features.view(features.size(0),-1,features.size(-1))#batch_size*num_pixels*encoder_dim
        return features

class AttentionBahdanau(nn.Module):
    def __init__(self,attention_dim,encoder_dim,decoder_dim):
        super(AttentionBahdanau,self).__init__()
        self.attention_dim=attention_dim
        self.encoder_dim=encoder_dim
        self.decoder_dim=decoder_dim

        self.weight_decoder=nn.Linear(decoder_dim,attention_dim)
        self.weight_encoder=nn.Linear(encoder_dim,attention_dim)
        self.full=nn.Linear(attention_dim,1)
        self.relu=nn.ReLU()
        self.softmax=nn.Softmax(dim=1)

    def forward(self,encoder_out,hidden_decoder):
        #encoder_out: batch_size*num_pixel*encoder_dim
        #hidden_decoder: batch_size*decoder_dim

        energies=self.full(
                torch.tanh(
                    self.weight_encoder(encoder_out)+self.weight_decoder(hidden_decoder).unsqueeze(1)
                    )
            )#batch_size*num_pixels*1
        energies=energies.squeeze(dim=2)
        attention_weights=self.softmax(energies)#batch_size*num_pixels
        context_vector=(encoder_out*attention_weights.unsqueeze(dim=2)).sum(dim=1)#batch_size*encoder_dim
        return context_vector,attention_weights


class Decoder(nn.Module):
    def __init__(self,attention_dim,decoder_dim,embedding_dim,vocab_size,vocabulary,encoder_dim=2048,drop_pro=0.2):
        super(Decoder,self).__init__()
        self.attention_dim=attention_dim
        self.encoder_dim=encoder_dim
        self.decoder_dim=decoder_dim
        self.embedding_dim=embedding_dim
        self.vocab_size=vocab_size
        self.vocabulary=vocabulary
        self.beam_size=3
        self.embedding=nn.Embedding(vocab_size,embedding_dim)
        self.attention=AttentionBahdanau(attention_dim,encoder_dim,decoder_dim)
        self.lstm_cell1=nn.LSTMCell(input_size=embedding_dim+encoder_dim,
                        hidden_size=decoder_dim,
                        bias=True
        )   

        self.fcn=nn.Linear(decoder_dim,vocab_size)
        self.dropout=nn.Dropout(drop_pro)
        self.init_h=nn.Linear(encoder_dim,decoder_dim)
        self.init_c=nn.Linear(encoder_dim,decoder_dim)
        self.softmax=nn.Softmax(dim=1)
        self.init_weights(pretrain_embedding=True)
  
    
    def fine_tune_embeddings(self,fine_tune=True):
        for param in self.embedding.parameters():
            param.requires_grad=fine_tune
    
    def load_pretrained_embedding(self):        
        embedding_matrix=np.zeros((self.vocab_size+1,self.embedding_dim))
        for word,idx in self.vocabulary.stoi.items():
            if idx >self.vocab_size:
              break
            embedding_vector=embedding_dict.get(word)
            if embedding_vector is not None:
                embedding_matrix[idx]=embedding_vector
            elif word in ['<pad>','<sos>','<eos>']:
                embedding_matrix[idx]=np.random.rand(1,300)
            else:
              embedding_matrix[idx]=embedding_dict['<unk>']

        return embedding_matrix

    def init_weights(self,pretrain_embedding=False):
        if pretrain_embedding is False:
          self.embedding.weight.data.uniform_(-0.1, 0.1)
        else:
          self.load_embedding_pretrained()
          self.fine_tune_embeddings(fine_tune=False)
        self.fcn.bias.data.fill_(0)
        self.fcn.weight.data.uniform_(-0.1,0.1)

    def load_embedding_pretrained(self):
        embeddings=self.load_pretrained_embedding()
        embeddings=torch.from_numpy(embeddings.astype(np.float32)).to(device)
        self.embedding.weight=nn.Parameter(embeddings)
    
    def fine_tune_embeddings(self,fine_tune=True):
        for param in self.embedding.parameters():
            param.requires_grad=fine_tune

    def init_hidden_state(self,features):
        mean_features=features.mean(dim=1)
        h1=self.init_h(mean_features)#batch_size*decoder_dim
        c1=self.init_c(mean_features)
        return h1,c1

    def forward(self,features,captions):
        #features: batch_size*num_pixels*encoder_dim
        #captions: batch_size*seq_length
        embedded=self.embedding(captions)#batch_size*seq_length*embedding_size
        seq_length=captions.size(1)-1
        batch_size=captions.size(0)
        h1,c1=self.init_hidden_state(features)
        num_pixels=features.size(1)

        preds=torch.zeros(batch_size,seq_length,self.vocab_size).to(device)
        attention_weights=torch.zeros(batch_size,seq_length,num_pixels).to(device)

        for i in range(seq_length):
            #context_vectore:batch_size*encoder_dim
            #attention_weight:batch_size*num_pixel
            context_vector,attention_weight=self.attention(features,h1)
            attention_weights[:,i,:]=attention_weight
            lstm_input=torch.cat((embedded[:,i,:],context_vector),dim=1)
            h1,c1=self.lstm_cell1(lstm_input,(h1,c1))#batch_size*decoder_dim
            outputs=self.fcn(self.dropout(h1))#batch_size*vocab_size
            preds[:,i,:]=outputs
        return preds,attention_weights


    def image_captions(self,feature,vocabulary,max_length=30):
        start_word=torch.tensor(vocabulary.stoi['<sos>']).view(1,-1).to(device)
        embedding=self.embedding(start_word)#1*batch_size*embedding_size
        h1,c1=self.init_hidden_state(feature)
        attention_weights=[]
        caption=[]
        for _ in range(max_length):
            context,attention_weight=self.attention(feature,h1)
            attention_weights.append(attention_weight)
            lstm_in=torch.cat((embedding[:,0,:],context),axis=1)
            h1,c1=self.lstm_cell1(lstm_in,(h1,c1))#batch_size*decoder_dim
            output=self.fcn(h1)
            predict_word=output.argmax(dim=1)
            if vocabulary.itos[predict_word.item()]=="<eos>":
                break
            
            caption.append(predict_word.item())
            embedding=self.embedding(predict_word.unsqueeze(dim=0))

        return [vocabulary.itos[index] for index in caption]



    def through_model(self,feature,word,hidden_state,beam_size):
        h1,c1=hidden_state
        word=torch.tensor(word).view(1,-1).to(device)
        embedding=self.embedding(word)
        context_vector,_=self.attention(feature,h1)
        lstm_in=torch.cat((embedding[:,0,:],context_vector),dim=1)
        h1,c1=self.lstm_cell1(lstm_in,(h1,c1))
        #output self.softmax() 1*vocab_size
        output=self.softmax(self.fcn(h1)).cpu().detach().numpy()[0]
        preds=np.argsort(output)[-beam_size:]
        return preds,[h1,c1],output

    def image_caption_with_beam_search(self,feature,vocabulary,beam_size,max_length=30,max_collection=15):
        hidden_state=self.init_hidden_state(feature)
        preds,hidden_state,output=self.through_model(feature,vocabulary.stoi['<sos>'],hidden_state,beam_size)
        words=[]
        cap_collection=[]
        prob_likelihood=list()
        has_full=False
        for i in range(beam_size):
            h1,c1=hidden_state
            words.append([[h1,c1],[[preds[i]],output[i]]])
        
        for _ in range(max_length-1):
            temp=[]
            for idx,[hidden_state_curr,pair] in enumerate(words):
                last_word=pair[0][-1]
                preds,hidden_state_curr,output=self.through_model(feature,last_word,hidden_state_curr,beam_size)

                for index in preds:
                    cap_current,prob=pair[0].copy(),pair[1]
                    prob+=np.log(output[index])
                    if vocabulary.itos[index]=="<eos>":
                        cap_collection.append(cap_current)
                        prob_likelihood.append(prob/(len(cap_current)+1))
                        if len(cap_collection)==max_collection:
                            has_full=True
                            break
                    
                    else:
                        cap_current.append(index)
                        temp.append([hidden_state_curr,[cap_current,prob]])
                    
            
            if has_full is True or len(temp)==0:
                break

            words=temp
            words=sorted(words,key=lambda l:l[1][1],reverse=False)[-beam_size:]

        if len(cap_collection)==0:
          return []
        index=prob_likelihood.index(max(prob_likelihood))
        caption=cap_collection[index]
        temp=[vocabulary.itos[index] for index in caption]
        return temp      
    
    def beam_search(self,feature,vocabulary,max_length=40,beam_size=3,max_collection=10):
        k=beam_size
        #encoder_image_size=8
        encoder_dim=feature.size(-1)
        num_pixel=feature.size(1)

        feature=feature.expand(k,num_pixel,encoder_dim)
        #tensor store k previous word at each step, now they're just <sos>
        k_prev_words=torch.LongTensor([[vocabulary.stoi['<sos>']]] * k).to(device)#(k,1)
        
        #tensor store top k sequences
        seqs=k_prev_words
        top_k_scores=torch.zeros(k,1).to(device)
        complete_seqs=list()
        complete_seq_scores=list()

        step=1
        h1,c1=self.init_hidden_state(feature)
        #s is a number less than or equal to k, because sequences are removed from this 
        #process once they hit <end>
        size_collection=0
        while True:
            embeddings=self.embedding(k_prev_words).squeeze(1)#s*embedding_dim
            contex_vec,_=self.attention(feature,h1)
            lstm_in=torch.cat((embeddings,contex_vec),dim=1)
            h1,c1=self.lstm_cell1(lstm_in,(h1,c1))
            output=self.fcn(h1)
            scores=F.log_softmax(output,dim=1)
            scores=top_k_scores.expand_as(scores)+scores

            if step==1:
                top_k_scores,top_k_words=scores[0].topk(k,0,True,True)
            else:
                top_k_scores,top_k_words=scores.view(-1).topk(k,0,True,True)

            pre_word_indexs=top_k_words / self.vocab_size
            pre_word_indexs=pre_word_indexs.type(torch.long)
            next_word_indexs=top_k_words%self.vocab_size
            seqs=torch.cat((seqs[pre_word_indexs],next_word_indexs.unsqueeze(1)),dim=1)
            incomplete_indexs=[id for id, next_word in enumerate(next_word_indexs) if next_word != vocabulary.stoi["<eos>"]]
            complete_indexs=list(set(range(len(next_word_indexs)))-set(incomplete_indexs))

            if len(complete_indexs)>0:
                complete_seqs.extend(seqs[complete_indexs].tolist())
                complete_seq_scores.extend(top_k_scores[complete_indexs])
                size_collection+=len(complete_indexs)
                
            
            k-=len(complete_indexs)
            if k==0:
              break
            
            seqs=seqs[incomplete_indexs]
            h1 = h1[pre_word_indexs[incomplete_indexs]]
            c1 = c1[pre_word_indexs[incomplete_indexs]]

            feature=feature[pre_word_indexs[incomplete_indexs]]
            top_k_scores=top_k_scores[incomplete_indexs].unsqueeze(1)
            k_prev_words=next_word_indexs[incomplete_indexs].unsqueeze(1)

            if step>max_length:
                break
            step+=1
        if len(complete_seq_scores)==0:
          return []
        index=complete_seq_scores.index(max(complete_seq_scores))
        temp=complete_seqs[index]
        temp=temp[1:-1]
        return [vocabulary.itos[index] for index in temp]
