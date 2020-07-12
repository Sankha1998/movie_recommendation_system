from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle


dfc=pd.read_csv('itdf.csv')

dfd=dfc.copy()
t=[]
for i in dfc['title']:
    t.append(i.split(' (')[0])
dfd['title']=t

model= open('model.pickle','rb')
model = pickle.load(model)

app=Flask(__name__)



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


@app.route('/')
def home():
    return render_template("home.html")


@app.route('/recommendation', methods=['POST','GET'])
def search_val():
    movie_name = request.form.get('name')
    return render_template("recommendation.html",mov_name=movie_name,rec=recommendations(movie_name))




if __name__=="__main__":
    app.run(debug=True)