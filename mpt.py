'''
Descripttion: 
version: 
Contributor: Minjun Lu
Source: Original
LastEditTime: 2023-07-23 15:50:39
'''
import os
import torch
from transformers import AutoModel,AutoTokenizer,AutoConfig,AutoModelForCausalLM,pipeline
from flask import Flask,jsonify,request,session
from flask import render_template,make_response
from flask_cors import CORS

app = Flask(__name__)
CORS(app, supports_credentials=True)
app.config['SECRET_KEY']=os.urandom(24)
name = "mosaicml/mpt-7b-storywriter"
config = AutoConfig.from_pretrained(name, trust_remote_code=True)
config.init_device = 'cuda:0' # 这个进程能看到的第一个显卡
config.max_seq_len = 65000 # (input + output) tokens can now be up to 4096
tokenizer_storywriter = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b", trust_remote_code=True,cache_dir='/server18/lmj/LLM/download/MPT-7b/storywriter')
model_storywriter = AutoModelForCausalLM.from_pretrained(name, trust_remote_code=True,config = config,torch_dtype=torch.bfloat16,cache_dir='/server18/lmj/LLM/download/MPT-7b/storywriter')

name = "mosaicml/mpt-7b-instruct"
config = AutoConfig.from_pretrained(name, trust_remote_code=True)
config.init_device = 'cuda:1'
config.max_seq_len = 4096 # (input + output) tokens can now be up to 4096
tokenizer_instruct = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b", trust_remote_code=True,cache_dir='/server18/lmj/LLM/download/MPT-7b/instruct')
model_instruct = AutoModelForCausalLM.from_pretrained(name, trust_remote_code=True,config = config,torch_dtype=torch.bfloat16,cache_dir='/server18/lmj/LLM/download/MPT-7b/instruct')

@app.route('/help',methods=['GET','POST'])
def help():
    context='''
    Welcome to GREATLLM!!!
    
    ip: 58.199.165.40
    port: 34502
    model name: MPT-7b
    route: /mpt-instruct,/mpt-storywriter
    input:{
        input: str
        task: str, [summarization|text-generation|...]
        max_new_tokens: int, [100|150]
        early_stopping: bool, [True|False]
        history: str,''
        **kw
    }
    output:{
        input: str
        task: str, [summarization|text-generation|...]
        max_new_tokens: int, [100|150]
        history: str
        output: str
    }
    
    Some ports are available for you to try other models:
        ChatGLM: 34501
    '''
    return jsonify(context)

@app.route('/mpt-instruct',methods=['GET'])
def instruct():
    if request.method == 'GET':
        history=str(request.args.get('history'))
        _input = str(request.args.get('input'))
        _input = history + _input
        _task = str(request.args.get('task'))
        _max_new_tokens = int(request.args.get('max_new_tokens'))
        _early_stopping = bool(request.args.get('early_stopping'))
        pipe = pipeline(_task, model=model_instruct, tokenizer=tokenizer_instruct,device='cuda:1')
        output = pipe(_input,max_new_tokens=_max_new_tokens,do_sample=True,early_stopping=_early_stopping)
        for output in output[0].values():
            pass
        return jsonify({'input':_input,'output':output.replace(_input,''),'history':output,'task':_task,'max_new_tokens':_max_new_tokens})

@app.route('/mpt-storywriter',methods=['GET'])
def storywriter():
    if request.method == 'GET':
        history=str(request.args.get('history'))
        _input = str(request.args.get('input'))
        _input = history + _input
        _task = str(request.args.get('task'))
        _max_new_tokens = int(request.args.get('max_new_tokens'))
        _early_stopping = bool(request.args.get('early_stopping'))
        pipe = pipeline(_task, model=model_storywriter, tokenizer=tokenizer_storywriter,device='cuda:0')
        output = pipe(_input,max_new_tokens=_max_new_tokens,do_sample=True,early_stopping=_early_stopping)
        for output in output[0].values():
            pass
        return jsonify({'input':_input,'output':output.replace(_input,''),'history':output,'task':_task,'max_new_tokens':_max_new_tokens})

if __name__ == '__main__':
    app.run(host='0.0.0.0' ,port=34502,threaded=True)