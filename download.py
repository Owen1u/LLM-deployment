'''
Descripttion: 
version: 
Contributor: Minjun Lu
Source: Original
LastEditTime: 2023-07-23 15:48:46
'''

from transformers import AutoModel
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("THUDM/glm-large-chinese", trust_remote_code=True,cache_dir='/server18/lmj/LLM/download/ChineseGLM/large')
model = AutoModel.from_pretrained("THUDM/glm-large-chinese", trust_remote_code=True,cache_dir='/server18/lmj/LLM/download/ChineseGLM/large').half().cuda()
