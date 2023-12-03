from transformers import  AutoModel,AutoTokenizer


model_name = "THUDM/chatglm3-6b-32k" #或者远程 “THUDM/chatglm2-6b”
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name,trust_remote_code=True).half().cuda()

prompt = """文本分类任务：将一段用户给外卖服务的评论进行分类，分成好评或者差评。

下面是一些范例:

味道真不错 -> 好评
太辣了，吃不下都  -> 差评

请对下述评论进行分类。返回'好评'或者'差评'，无需其它说明和解释。

xxxxxx ->

"""

def get_prompt(text):
    return prompt.replace('xxxxxx',text)

response, his = model.chat(tokenizer, get_prompt('味道不错，下次还来'), history=[])
print(response)  


#增加4个范例
his.append(("太贵了 -> ","差评"))
his.append(("非常快，味道好 -> ","好评"))
his.append(("这么咸真的是醉了 -> ","差评"))
his.append(("价格感人 优惠多多 -> ","好评"))

response, history = model.chat(tokenizer, "一言难尽啊 -> ", history=his)
print(response) 

response, history = model.chat(tokenizer, "还凑合一般般 -> ", history=his)
print(response) 

response, history = model.chat(tokenizer, "我家狗狗爱吃的 -> ", history=his)
print(response) 

#封装成一个函数吧~
def predict(text):
    response, history = model.chat(tokenizer, f"{text} ->", history=his,
    temperature=0.01)
    return response 

predict('死鬼，咋弄得这么有滋味呢') #在我们精心设计的一个评论下，ChatGLM2-6b终于预测错误了~

