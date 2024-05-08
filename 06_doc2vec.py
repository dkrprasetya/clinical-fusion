import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from sklearn.utils import shuffle

from utils import clean_text, text2words
# Update: need to install scipy 1.10.1 as triu will be removed in later version: pip install scipy==1.10.1
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

import json
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Help')
    parser.add_argument('--phase', type=str, default='infer',
                        help='train, or infer')
    parser.add_argument('--epochs', type=int, default=30,
                        help='number of training epochs')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    df = pd.read_csv('./data/processed/earlynotes.csv')
    df['text'] = df['text'].astype(str).apply(text2words)

    if args.phase != 'infer':
        if not os.path.exists("models"):
            os.makedirs("models") 

        epochs = args.epochs
        splits = range(10)
        data = json.load(open('./data/processed/files/splits.json'))
        train_ids = np.hstack([data[t] for t in splits[:7]])
        # Update: first convert to float then to int as sometimes it could be value like 7452.0
        # Update: from x[-10:-4] to x[-12:-4] as the list is not complete (actually, might just update to x[-12:-6] andcan avoid adding float, but just in case)
        train_ids = list(map(lambda x: int(float(x[-12:-4])), train_ids))
        print("train_ids:", train_ids)
        print("train_ids len:", len(train_ids))
        train = df[df['hadm_id'].isin(train_ids)]['text'].tolist()
        print("train:", train)

        train_tagged = []
        for idx, text in enumerate(train):
            train_tagged.append(TaggedDocument(text, tags=[str(idx)]))

        model = Doc2Vec(dm=0, vector_size=200, negative=5, alpha=0.025, hs=0, min_count=5, sample=0, workers=16)
        print("train_tagged size:", len(train_tagged))
        model.build_vocab([x for x in train_tagged])
        for epoch in tqdm(range(epochs)):
            model.train(shuffle([x for x in train_tagged]), total_examples=len(train_tagged), epochs=1)
            model.alpha -= 0.0002
            model.min_alpha = model.alpha
        
        model.save('./models/doc2vec.model')
    else:
        print('Infering...')
        doc2vec = Doc2Vec.load('./models/doc2vec.model')
        df['vector'] = df['text'].apply(lambda note: doc2vec.infer_vector(note).tolist())
        df = df.groupby('hadm_id')['vector'].apply(list).reset_index()
        vector_dict = {}
        for idx, row in df.iterrows():
            vector_dict[str(int(row['hadm_id']))] = row['vector']
        json.dump(vector_dict, open('./data/processed/files/vector_dict.json', 'w'))
        