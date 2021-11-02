"""
Obtain BERT contextual embeddings for sentences.
"""

import argparse

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import torch
from transformers import AutoTokenizer, AutoModel
import json
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
import os

np.set_printoptions(threshold=np.inf)


def main():
    """Main."""
    args = get_args()

    INPUT = args.input # 'etl_nothree'

    DIVISOR = args.divisor # 3 or 200 :)

    # different setups might need different solutions here
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained('SZTAKI-HLT/hubert-base-cc')
    model = AutoModel.from_pretrained('SZTAKI-HLT/hubert-base-cc')


    # read corpus from tsv
    corpus = pd.read_csv(f"corpora_and_labels/{INPUT}.tsv", sep='\t')
    print(corpus)

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
    if args.first_word_pooling:
        featuresfinal = np.empty((0, DIMS), dtype='float32')
    elif args.all_word_pooling:
        featuresfinal = np.empty((0, max_len, DIMS), dtype='float32')
    elif args.mean_pooling:
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
        print(input_batch)
        print(mask_batch)
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
        if args.first_word_pooling:
            # we take the vector of the _first_ word, seriously :)
            features = last_hidden_states[0][:, 0, 0:DIMS].cpu().numpy()
        elif args.all_word_pooling:
            features = last_hidden_states[0][:, :, 0:DIMS].cpu().numpy()
        elif args.mean_pooling:
            lhs = last_hidden_states[0][:, :, 0:DIMS].cpu().numpy()
            features = np.mean(lhs, axis=1) # average above words

        if args.verbose:
            print(features.shape)
            print(features)

        featuresfinal = np.append(featuresfinal, features, axis=0)

        print('Finished with batch', batch_cnt)

    # output + labels are saved as separate files
    labels = corpus["topik"]

    np.save(f"featuresfinal_{INPUT}", featuresfinal)
    np.save(f"labels_{INPUT}", labels)

    if not args.verbose:

        print(list(featuresfinal))
        print(labels)

    else:

        print()
        print('Vectors')

        for padd, feat, label in zip(padded, featuresfinal, labels):
            print()
            for p, f in zip(padd, feat):
                # XXX why do we have additional whitespaces?
                token = ''.join(filter(lambda x: x != " ", tokenizer.decode(int(p))))
                print(f'{label}\t{p}\t"{token}"\t{f}')

        print()
        print('Distances')

        import itertools as it
        from scipy.spatial import distance

        SAMPLEWORD = 2765 # "vár"

        for i, j in it.combinations(range(len(padded)), 2):
            ap, af, al = padded[i], featuresfinal[i], labels[i]
            bp, bf, bl = padded[j], featuresfinal[j], labels[j]
            a = [x[1] for x in zip(ap, af) if x[0] == SAMPLEWORD]
            b = [x[1] for x in zip(bp, bf) if x[0] == SAMPLEWORD]
            dist = distance.cosine(a, b)
            issame = al == bl
            issamemark = '==' if issame else '<>'
            anom = '!ERR!' if dist >= 0.08 and issame else ''
            print(f'#{i} L{al} {issamemark} #{j} L{bl} = {dist} {anom}')

        # XXX add clustering at the end -> polysemy will be solved :)
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
        
    MyFile = open('clasrep_bert.json', 'w')
    json.dump(clasrep, MyFile)
    MyFile.close()

    MyFile = open('param_bert.json', 'w')
    json.dump(paramlist, MyFile)
    MyFile.close()

def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '-i', '--input',
        help='input corpus (without ".tsv")',
        required=True,
        type=str,
        default=argparse.SUPPRESS
    )
    parser.add_argument(
        '-d', '--divisor',
        help='split corpus to this many batches',
        required=True,
        type=int,
        default=argparse.SUPPRESS
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '-a', '--all-word-pooling',
        help='all words = sentence representation',
        action='store_true'
    )
    group.add_argument(
        '-f', '--first-word-pooling',
        help='first word = sentence representation',
        action='store_true'
    )
    group.add_argument(
        '-m', '--mean-pooling',
        help='average of all words = sentence representation',
        action='store_true'
    )

    parser.add_argument(
        '-v', '--verbose',
        help='verbose output for investigation',
        action='store_true'
    )

    return parser.parse_args()


if __name__ == '__main__':
    main()
