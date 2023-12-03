import pandas as pd 
import numpy as np  
import datasets 
from tqdm import tqdm
import transformers
import torch 
from transformers import AutoTokenizer, AutoModel, TrainingArguments, AutoConfig
import torch
from tqdm import tqdm 
import torch.nn as nn
from peft import get_peft_model, LoraConfig, TaskType
import warnings
from peft import PeftModel 
from torchkeras import KerasModel 
from accelerate import Accelerator
import keras

warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

'''List[Tuple[str, str]]'''
# model_name = "THUDM/chatglm2-6b"

'''List[Dict],{"role": "user", "content": query}'''
model_name = 'THUDM/chatglm3-6b-32k' 

# his = [("太贵了 -> ","差评"),("非常快，味道好 -> ","好评"),\
#     ("这么咸真的是醉了 -> ","差评"),("价格感人 优惠多多 -> ","好评")]


his = [{'role':'user','content':"太贵了 -> 差评"},{'role':'user','content':"非常快，味道好 -> 好评"},\
    {'role':'user','content':"这么咸真的是醉了 -> 差评"},{'role':'user','content':"价格感人 优惠多多 -> 好评"}]

ckpt_path = 'checkpoints'
save_path = 'chatglm6b_lora'

tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

'''Build Dataset'''
print('=================================Building Dataset====================================')
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

dftrain = pd.read_parquet('data/dftrain.parquet')
dftest = pd.read_parquet('data/dftest.parquet')
dftrain['tag'].value_counts()


def build_inputs(query, history):
    prompt = """文本分类任务：将一段用户给外卖服务的评论进行分类，分成好评或者差评。

下面是一些范例:

味道真不错 -> 好评
太辣了，吃不下都 -> 差评
太贵了-> 差评
非常快，味道好 -> 好评
这么咸真的是醉了 -> 差评
价格感人 优惠多多 -> 好评
狗喜欢吃 -> 差评

请对下述评论进行分类。返回'好评'或者'差评'，无需其它说明和解释。

"""
    prompt += "{} -> ".format(query)
    return prompt 

dftrain['context'] = [build_inputs(x,history=his) for x in dftrain['text']]
dftrain['target'] = [x for x in dftrain['tag']]
dftrain = dftrain[['context','target']]

dftest['context'] = [build_inputs(x,history=his) for x in dftest['text']]
dftest['target'] = [x for x in dftest['tag']]
dftest = dftest[['context','target']]
ds_train = datasets.Dataset.from_pandas(dftrain)
ds_val = datasets.Dataset.from_pandas(dftest)

max_seq_length = 512
skip_over_length = True



config = transformers.AutoConfig.from_pretrained(
    model_name, trust_remote_code=True, device_map='auto')

def preprocess(example):
    context = example["context"]
    target = example["target"]
    
    context_ids = tokenizer.encode(
            context, 
            max_length=max_seq_length,
            truncation=True)
    
    target_ids = tokenizer.encode(
        target,
        max_length=max_seq_length,
        truncation=True,
        add_special_tokens=False)
    
    input_ids = context_ids + target_ids + [config.eos_token_id]
    
    # -100标志位后面会在计算loss时会被忽略不贡献损失，我们集中优化target部分生成的loss
    labels = [-100]*len(context_ids)+ target_ids + [config.eos_token_id]
    
    return {"input_ids": input_ids,
            "labels": labels,
            "context_len": len(context_ids),
            'target_len':len(target_ids)+1}


ds_train_token = ds_train.map(preprocess).select_columns(['input_ids','labels', 'context_len','target_len'])
if skip_over_length:
    ds_train_token = ds_train_token.filter(
        lambda example: example["context_len"]<max_seq_length and example["target_len"]<max_seq_length)

ds_val_token = ds_val.map(preprocess).select_columns(['input_ids', 'labels','context_len','target_len'])
if skip_over_length:
    ds_val_token = ds_val_token.filter(
        lambda example: example["context_len"]<max_seq_length and example["target_len"]<max_seq_length)

'''prepare for train'''
def data_collator(examples: list):
    len_ids = [len(example["input_ids"]) for example in examples]
    longest = max(len_ids) #之后按照batch中最长的input_ids进行padding
    
    input_ids = []
    labels_list = []
    
    for length, example in sorted(zip(len_ids, examples), key=lambda x: -x[0]):
        ids = example["input_ids"]
        labs = example["labels"]
        
        ids = ids + [tokenizer.pad_token_id] * (longest - length)
        labs = labs + [-100] * (longest - length)
        
        input_ids.append(torch.LongTensor(ids))
        labels_list.append(torch.LongTensor(labs))
          
    input_ids = torch.stack(input_ids)
    labels = torch.stack(labels_list)
    return {
        "input_ids": input_ids,
        "labels": labels,
    }

dl_train = torch.utils.data.DataLoader(ds_train_token,num_workers=2,batch_size=4,
                                       pin_memory=True,shuffle=True,
                                       collate_fn = data_collator)
dl_val = torch.utils.data.DataLoader(ds_val_token,num_workers=2,batch_size=4,
                                    pin_memory=True,shuffle=True,
                                     collate_fn = data_collator)

dl_train.size = 300 #每300个step视作一个epoch，做一次验证

model = AutoModel.from_pretrained(model_name,
                                  load_in_8bit=False, 
                                  trust_remote_code=True).cuda()

# model = AutoModel.from_pretrained(model_name,
#                                   load_in_8bit=True,
#                                   device_map='auto', 
#                                   trust_remote_code=True).cuda()

'''unknown'''
model.supports_gradient_checkpointing = True  #节约cuda
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

# peft == 0.6.2
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    # target_modules=["query_key_value"], 
    inference_mode=False,
    r=8,
    lora_alpha=32, 
    lora_dropout=0.1,
)

model = get_peft_model(model, peft_config)
model.is_parallelizable = True
model.model_parallel = True

'''打印可训练参数与总参数的对比'''
model.print_trainable_parameters()
print('================================model=================================\n',model)
# for i in model.children():
#     print(i)

class StepRunner:
    def __init__(self, net, loss_fn, accelerator=None, stage = "train", metrics_dict = None, 
                 optimizer = None, lr_scheduler = None
                 ):
        self.net,self.loss_fn,self.metrics_dict,self.stage = net,loss_fn,metrics_dict,stage
        self.optimizer,self.lr_scheduler = optimizer,lr_scheduler
        self.accelerator = accelerator if accelerator is not None else Accelerator() 
        if self.stage=='train':
            self.net.train() 
        else:
            self.net.eval()
    
    def __call__(self, batch):
        
        #loss
        with self.accelerator.autocast():
            loss = self.net(input_ids=batch["input_ids"],labels=batch["labels"]).loss

        #backward()
        if self.optimizer is not None and self.stage=="train":
            self.accelerator.backward(loss)
            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(self.net.parameters(), 1.0)
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            self.optimizer.zero_grad()
            
        all_loss = self.accelerator.gather(loss).sum()
        
        #losses (or plain metrics that can be averaged)
        step_losses = {self.stage+"_loss":all_loss.item()}
        
        #metrics (stateful metrics)
        step_metrics = {}
        
        if self.stage=="train":
            if self.optimizer is not None:
                step_metrics['lr'] = self.optimizer.state_dict()['param_groups'][0]['lr']
            else:
                step_metrics['lr'] = 0.0
        return step_losses,step_metrics
    
KerasModel.StepRunner = StepRunner 


#仅仅保存lora可训练参数
def save_ckpt(self, ckpt_path='checkpoint', accelerator = None):
    unwrap_net = accelerator.unwrap_model(self.net)
    unwrap_net.save_pretrained(ckpt_path)
    
def load_ckpt(self, ckpt_path='checkpoint'):
    import os
    self.net.load_state_dict(
        torch.load(os.path.join(ckpt_path,'adapter_model.bin')),strict =False)
    self.from_scratch = False
    
KerasModel.save_ckpt = save_ckpt 
KerasModel.load_ckpt = load_ckpt 


keras_model = KerasModel(model,
                        loss_fn = None,
                        optimizer=torch.optim.AdamW(model.parameters(),
                        lr=2e-6))


keras_model.fit(train_data = dl_train,
                val_data = dl_val,
                epochs=5,patience=2,
                # epochs=100,patience=5,
                monitor='val_loss',mode='min',
                ckpt_path = ckpt_path,
                mixed_precision='fp16'
               )

print('===========================Training Finished==============================')

'''重新加载原模型'''
model = AutoModel.from_pretrained(model_name,
                                  load_in_8bit=False, 
                                  trust_remote_code=True).cuda()
# model = AutoModel.from_pretrained(model_name,
#                                   load_in_8bit=True,
#                                   device_map='auto', 
#                                   trust_remote_code=True).cuda()

model = PeftModel.from_pretrained(model,ckpt_path)
model = model.merge_and_unload() #合并lora权重

print('===========================Saving Checkpoint=============================')
model.save_pretrained(save_path, max_shard_size='1GB')
tokenizer.save_pretrained(save_path)

def predict(text):
    response, history = model.chat(tokenizer, f"{text} -> ", history=his,
    temperature=0.01)
    return response 



print('===========================Evaluating Acc on Dataset=====================================')

dftest = pd.read_parquet('data/dftest.parquet')
preds = ['' for x in dftest['text']]
for i in tqdm(range(len(dftest))):
    text = dftest['text'].loc[i]
    preds[i] = predict(text)

dftest['pred'] = preds 
dftest.pivot_table(index='tag',columns = 'pred',values='text',aggfunc='count')

acc = len(dftest.query('tag==pred'))/len(dftest)
print('acc=',acc)





