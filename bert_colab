from google.colab import drive 
drive.mount('/content/gdrive')
corpus=pd.read_csv('gdrive/My Drive/etl.tsv', sep='\t')
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import torch
from transformers import AutoTokenizer, AutoModel

import pandas as pd
import numpy as np
import os


INPUT = 'etl' # 'etl_nothree'

DIVISOR = 200 # 3 or 200 :)

    # different setups might need different solutions here
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained('SZTAKI-HLT/hubert-base-cc')
model = AutoModel.from_pretrained('SZTAKI-HLT/hubert-base-cc')

    # tokenize corpus
    # note: it does not check for length requirements at this time

tokenized = corpus["text"].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))
print(tokenized)

    # create padding and attention masks
max_len = 0
for i in tokenized.values:
    if len(i) > max_len:
        max_len = len(i)
padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])
print(padded)

attention_mask = np.where(padded != 0, 1, 0)
print(attention_mask)

    # for computationally weaker setups, batch execution is the only way to process the texts
    # manipulate floor divisor if a different batch size is needed
batchsize = (len(corpus) // DIVISOR) + 1
print('Number of batches:', batchsize)
splitpadded = np.array_split(padded, batchsize)
splitmask = np.array_split(attention_mask, batchsize)


last_hidden_states = []
model = model.to(device)

DIMS = 768 # 768 at most (because of using BERT Base)

featuresfinal = np.empty((0, DIMS), dtype='float32')

    # take batches of tokenized texts
    # to extract BERT's last hidden states, i.e. contextual word embeddings
    #
    # XXX handling attention_mask was erroneous here,
    # because array_split() gives variable length!
    # now: zip() ensures that text and attention data is taken strictly in parallel
for count, (batch, mask) in enumerate(zip(splitpadded, splitmask)):
    batch_cnt = count + 1
    print(f'Batch #{batch_cnt}')
    paddedsplit = np.array(batch, dtype='float64')

    input_batch = torch.tensor(batch).to(torch.long)
    mask_batch = torch.tensor(mask)
    print('Batches established!')

        # put data onto GPU
    input_batch = input_batch.to(device)
    mask_batch = mask_batch.to(device)
    print('Lengths', input_batch.size(0), mask_batch.size(0))

        # no_grad ensures there is no gradient update in the model,
        # as we are not looking for recursive training here
    with torch.no_grad():
        print('Model is running on', model.device)
        last_hidden_states = model(input_batch, attention_mask=mask_batch)
        print('Hidden states created for batch', batch_cnt)

        # tensor dimensions: 0=sents, 1=words, 2=coords

        lhs = last_hidden_states[0][:, :, 0:DIMS].cpu().numpy()
        features = np.mean(lhs, axis=1) # average above words

        print(features.shape)

        featuresfinal = np.append(featuresfinal, features, axis=0)

        print('Finished with batch', batch_cnt)
        
from google.colab import files
np.save("featuresfinal", featuresfinal)
!cp featuresfinal.npy "/content/gdrive/My Drive/"
np.save("labels", corpus["topik"])
!cp labels.npy "/content/gdrive/My Drive/"
featuresfinal = np.load("/content/gdrive/My Drive/featuresfinal.npy")

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


# MinMax scaling is applied to the features
scaler = MinMaxScaler()
featuresfinal = scaler.fit_transform(featuresfinal)

# the parameter space is defined below
C = [0.1, 1]
tol = [0.001, 0.005, 0.01]
weighting = ['balanced']
solver = ['liblinear']
max_iter = [8000]
parameters = dict(C=C, tol=tol, class_weight=weighting, solver=solver, max_iter=max_iter)

clasrep = list()
paramlist = list()



labels = corpus["topik"].to_numpy()
#labels = np.load("labels.npy")

for i in range(3):
    train_features, test_features, train_labels, test_labels = train_test_split(featuresfinal, labels, stratify=labels)
    lr = LogisticRegression()
    lrmodel = GridSearchCV(lr, parameters, cv = 3, scoring = 'f1_weighted', n_jobs = -1)
    lrmodel.fit(train_features, train_labels)
    predictions = lrmodel.predict(test_features)
    classifrep = classification_report(test_labels, predictions, output_dict = True)
    clasrep.append(classifrep)
    paramlist.append(lrmodel.best_params_)
    print("Finished with run!")
    
import json

MyFile = open('clasrep_bert_three.json', 'w')
json.dump(clasrep, MyFile)
MyFile.close()

MyFile = open('param_bert_three.json', 'w')
json.dump(paramlist, MyFile)
MyFile.close()

!cp clasrep_bert_three.json "/content/gdrive/My Drive/"
!cp param_bert_three.json "/content/gdrive/My Drive/"
