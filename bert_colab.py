# There are several ways of connecting files to a Google Colab (Google Drive or direct upload), this example shows using your own Google Drive account.
# These are the packages necessary for the script.
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) # sklearn prints many futurewarnings, this option suppresses these
import torch
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import numpy as np
import os
from google.colab import drive
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import json


drive.mount('/content/gdrive')
corpus=pd.read_csv('gdrive/My Drive/corpus.tsv', sep='\t') # you need to provide your corpus file here


DIVISOR = 200 # batch length for the model - can be arbitrarily low if computational setup is weak
tokenizer = AutoTokenizer.from_pretrained('SZTAKI-HLT/hubert-base-cc') # here you can change your preferred pretrained model
model = AutoModel.from_pretrained('SZTAKI-HLT/hubert-base-cc') # here you can change your preferred pretrained model

# tokenize corpus
# note: it does not check for length requirements at this time

tokenized = corpus["text"].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))

# create padding and attention masks based on automatic token length measurement
max_len = 0
for i in tokenized.values:
    if len(i) > max_len:
        max_len = len(i)
padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])
print(max_len)
attention_mask = np.where(padded != 0, 1, 0)


# for computationally weaker setups, batch execution is the only way to process the texts
# manipulate floor divisor if a different batch size is needed
batchsize = (len(corpus) // DIVISOR) + 1
print('Number of batches:', batchsize)
splitpadded = np.array_split(padded, batchsize)
splitmask = np.array_split(attention_mask, batchsize)

last_hidden_states = []

DIMS = 768 # 768 at most (because of using BERT Base, otherwise 1024 for Large models)

features = np.empty((0, DIMS), dtype='float32')

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

    print('Lengths', input_batch.size(0), mask_batch.size(0))

    # no_grad ensures there is no gradient update in the model,
    # as we are not looking for recursive training here
    with torch.no_grad():
        print('Model is running on', model.device)
        last_hidden_states = model(input_batch, attention_mask=mask_batch)
        print('Hidden states created for batch', batch_cnt)

        # tensor dimensions: 0=sents, 1=words, 2=coords

        lhs = last_hidden_states[0][:, :, 0:DIMS].numpy() # this part can be manipulated not to use means
        features = np.mean(lhs, axis=1) # average above words

        print(features.shape)

        featuresfinal = np.append(featuresfinal, features, axis=0)

        print('Finished with batch', batch_cnt)

        
# this snippet can save the tokens and labels to your drive, keeping them for later use
from google.colab import files
np.save("featuresfinal", featuresfinal)
!cp featuresfinal.npy "/content/gdrive/My Drive/"
np.save("labels", corpus["topik"])
!cp labels.npy "/content/gdrive/My Drive/"
featuresfinal = np.load("/content/gdrive/My Drive/featuresfinal.npy")


# MinMax scaling is applied to the features, as this helps (as usual with many ML applications)
scaler = MinMaxScaler()
featuresfinal = scaler.fit_transform(featuresfinal)

# the parameter space is defined below - follow the readme on how to update these
C = [0.1, 1]
tol = [0.001, 0.005, 0.01]
weighting = ['balanced']
solver = ['liblinear']
max_iter = [8000]
parameters = dict(C=C, tol=tol, class_weight=weighting, solver=solver, max_iter=max_iter)

clasrep = list()
paramlist = list()

labels = corpus["topik"].to_numpy()

# this is in essence a repeated k-fold cross validated Logistic Regression, with a list of dicts output
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


keylist = list(clasrep[0].keys())
results = pd.DataFrame()

for i in range(0,len(param)):
    results = results.append(pd.DataFrame.from_dict(param[i], orient = 'index').transpose(), ignore_index = True)

results.mean()

# this normalizes the output list as if it were a json import
results = pd.io.json.json_normalize(param)
results.mean()

# you can save the classification reports and the best parameter lists as json files
MyFile = open('clasrep_bert.json', 'w')
json.dump(clasrep, MyFile)
MyFile.close()

MyFile = open('param_bert.json', 'w')
json.dump(paramlist, MyFile)
MyFile.close()

# if you use Colab, it is mandatory to transfer json files to your Drive storage or save them to your computer
!cp clasrep_bert.json "/content/gdrive/My Drive/"
!cp param_bert.json "/content/gdrive/My Drive/"
