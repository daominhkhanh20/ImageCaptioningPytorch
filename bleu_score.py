import math 
from nltk import ngrams 
from collections import Counter 
import numpy as np 
from nltk.translate.bleu_score import sentence_bleu,corpus_bleu




def count_gram(text,ngram=1):
    return Counter(ngrams(text,ngram))

def count_clip(candidate,references,ngram=1):
    candidate=count_gram(candidate,ngram=ngram)
    max_reference_count={}
    for setence in references:
        sentence_gram=count_gram(setence,ngram=ngram)
        for words in sentence_gram:
            max_reference_count[words]=max(max_reference_count.get(words,0),sentence_gram[words])
    temp={
        word:min(count,max_reference_count.get(word,0)) for word,count in candidate.items()
    }
    return temp,candidate


def modifed_precision(candidate,references,ngram):
    count_clip_sentence,candidate_after=count_clip(candidate,references,ngram)
    #print(count_clip_sentence)
    #print(candidate)
    return max(
        1.0*sum(count_clip_sentence.values())/max(sum(candidate_after.values()),1),
        1e-10#smooth values
        )

def get_closest_reference_length(candidate,references):
    temp=len(candidate)
    idx=np.argmin([abs(len(sentence)-temp) for sentence in references])
    return len(references[idx])


def brevity_penalty(candidate,references):
    c=len(candidate)
    r=get_closest_reference_length(candidate,references)
    if c>r:
        return 1
    return np.exp(1-r*1.0/c)


def final_bleu_score(references,candidate,weights=(0.25,0.25,0.25,0.25)):
    bp=brevity_penalty(candidate,references)
    modifed_precisions=[
        modifed_precision(candidate,references,ngram=i) for i in range(1,5,1)
    ]
    print(modifed_precisions)
    s=[wi*math.log(mp_i) for wi,mp_i in zip(weights,modifed_precisions)]
    return bp*np.exp(sum(s))

# references=[
#     "This is a middle test in the year".split(),
#     "This is new test in this year".split()
# ]

# #print(references)
# translation_1='This is test in this year'.split()
# #print(translation_1)
# score=sentence_bleu(references,translation_1)
# print(score)
# score1=final_bleu_score(references,translation_1)
# print(score1)

