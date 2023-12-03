import pandas as pd 
import numpy as np 
import datasets 
from transformers import  AutoModel,AutoTokenizer
from tqdm import tqdm 


model_name = "THUDM/chatglm2-6b" #或者远程 “THUDM/chatglm2-6b”
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name,trust_remote_code=True).half().cuda()

def predict(text):
    response, history = model.chat(tokenizer, f"{text} ->", history=[],
    temperature=0.01)
    return response 
df = pd.read_csv("data/waimai_10k.csv")

df['tag'] = df['label'].map({0:'差评',1:'好评'})
df = df.rename({'review':'text'},axis = 1)

dfgood = df.query('tag=="好评"')
dfbad = df.query('tag=="差评"').head(len(dfgood)) #采样部分差评，让好评差评平衡
df = pd.concat([dfgood,dfbad])

print(df['tag'].value_counts())


ds_dic = datasets.Dataset.from_pandas(df).train_test_split(
    test_size = 2000,shuffle=True, seed = 43)
dftrain = ds_dic['train'].to_pandas()
dftest = ds_dic['test'].to_pandas()
dftrain.to_parquet('data/dftrain.parquet')
dftest.to_parquet('data/dftest.parquet')

preds = ['' for x in dftest['tag']] 
for i in tqdm(range(len(dftest))):
    text = dftest['text'].loc[i]
    preds[i] = predict(text)

dftest['pred'] = preds 

dftest.pivot_table(index='tag',columns = 'pred',values='text',aggfunc='count')
acc = len(dftest.query('tag==pred'))/len(dftest)
print('acc=',acc)