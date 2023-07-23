'''
Descripttion: 
version: 
Contributor: Minjun Lu
Source: Original
LastEditTime: 2023-06-22 23:34:19
'''
import os
from flask import Flask,jsonify,request,session
from flask import render_template,make_response
from flask_cors import CORS

app = Flask(__name__)
CORS(app, supports_credentials=True)
app.config['SECRET_KEY']=os.urandom(24)
@app.route('/LLM')
def index():
    resp = make_response(render_template("index.html"))
    return resp

if __name__ == '__main__':
    app.run(host='0.0.0.0' ,port=34500,threaded=True)