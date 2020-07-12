import numpy as np
import pandas as pd
import pickle
dfc=pd.read_csv('itdf.csv')
dfd=dfc.copy()
t=[]
for i in dfc['title']:
    t.append(i.split(' (')[0])
dfd['title']=t

dfc=dfc[dfc['rating']>2500].reset_index(drop=True)

from sklearn.feature_extraction.text import TfidfVectorizer
tfv = TfidfVectorizer(analyzer='word',ngram_range=(1, 3),min_df=1)
x = tfv.fit_transform(dfc['genres'])
from sklearn.metrics.pairwise import sigmoid_kernel
model = sigmoid_kernel(x, x)

def recommendations(title):
    sug =[]
    i_d = []
    indices = pd.Series(dfd.index, index=dfd['title']).drop_duplicates()
    idx = indices[title]
    dis_scores = list(enumerate(model[idx]))
    dis_scores = sorted(dis_scores, key=lambda x: x[1], reverse=True)
    dis_scores = dis_scores[1:31]
    idn = [i[0] for i in dis_scores]
    final = dfd.iloc[idn].reset_index()
    idn = [i for i in final['index']]
    for j in idn:
        if (j < 15951):
            i_d.append(j)
    indices = pd.Series(dfc.index, index=dfc['title']).drop_duplicates()
    for i in range(1, 8):
        sug.append(indices.iloc[idn].index[i])
    return sug

m=pickle.dump(model,open('model.pickle','wb'))

print(recommendations('Toy Story'))



