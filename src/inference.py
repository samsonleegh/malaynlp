import json
import numpy as np
import tensorflow as tf
import nltk
import numpy as np
from nltk import Tree
from transformers import XLNetTokenizer
from fastapi import FastAPI, Request, Form, File, UploadFile, status
from fastapi.encoders import jsonable_encoder
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import FileResponse

from parse_nk_xlnet_base import BERT_TOKEN_MAPPING

import pyximport
pyximport.install(setup_args={"include_dirs": np.get_include()})
import chart_decoder_py

# download of chart_decoder pyx file
# import wget
# url_chart_decoder = 'https://raw.githubusercontent.com/michaeljohns2/self-attentive-parser/michaeljohns2-support-tf2-patch/benepar/chart_decoder.pyx'
# wget.download(url_chart_decoder)

app = FastAPI()

app.mount(
    "/static", StaticFiles(directory="./static"), name="static")
templates = Jinja2Templates(directory="./templates")

tokenizer = XLNetTokenizer.from_pretrained(
    'huseinzol05/xlnet-base-bahasa-cased', do_lower_case = False
)

with tf.gfile.GFile('../export_model/xlnet-base.pb.quantized', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def)

input_ids = graph.get_tensor_by_name('import/input_ids:0')
word_end_mask = graph.get_tensor_by_name('import/word_end_mask:0')
charts = graph.get_tensor_by_name('import/charts:0')
tags = graph.get_tensor_by_name('import/tags:0')
sess = tf.InteractiveSession(graph = graph)

with open('../export_model/vocab-xlnet-base.json') as fopen:
    data = json.load(fopen)
    
LABEL_VOCAB = data['label']
TAG_VOCAB = data['tag']

PTB_TOKEN_ESCAPE = {u"(": u"-LRB-",
    u")": u"-RRB-",
    u"{": u"-LCB-",
    u"}": u"-RCB-",
    u"[": u"-LSB-",
    u"]": u"-RSB-"}

def make_feed_dict_bert(sentences):

    BERT_MAX_LEN = 512
    all_input_ids = np.zeros((len(sentences), BERT_MAX_LEN), dtype=int)
    all_word_end_mask = np.zeros((len(sentences), BERT_MAX_LEN), dtype=int)

    subword_max_len = 0
    for snum, sentence in enumerate(sentences):
        tokens = []
        word_end_mask = []

        cleaned_words = []
        for word in sentence:
            word = BERT_TOKEN_MAPPING.get(word, word)
            if word == "n't" and cleaned_words:
                cleaned_words[-1] = cleaned_words[-1] + "n"
                word = "'t"
            cleaned_words.append(word)

        for word in cleaned_words:
            word_tokens = tokenizer.tokenize(word)
            for _ in range(len(word_tokens)):
                word_end_mask.append(0)
            word_end_mask[-1] = 1
            tokens.extend(word_tokens)
        tokens.append("<sep>")
        word_end_mask.append(1)
        tokens.append("<cls>")
        word_end_mask.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        subword_max_len = max(subword_max_len, len(input_ids))

        all_input_ids[snum, :len(input_ids)] = input_ids
        all_word_end_mask[snum, :len(word_end_mask)] = word_end_mask

    all_input_ids = all_input_ids[:, :subword_max_len]
    all_word_end_mask = all_word_end_mask[:, :subword_max_len]
    return all_input_ids, all_word_end_mask

def make_nltk_tree(sentence, tags, score, p_i, p_j, p_label):

    # Python 2 doesn't support "nonlocal", so wrap idx in a list
    idx_cell = [-1]
    def make_tree():
        idx_cell[0] += 1
        idx = idx_cell[0]
        i, j, label_idx = p_i[idx], p_j[idx], p_label[idx]
        label = LABEL_VOCAB[label_idx]
        if (i + 1) >= j:
            word = sentence[i]
            tag = TAG_VOCAB[tags[i]]
            tag = PTB_TOKEN_ESCAPE.get(tag, tag)
            word = PTB_TOKEN_ESCAPE.get(word, word)
            tree = Tree(tag, [word])
            for sublabel in label[::-1]:
                tree = Tree(sublabel, [tree])
            return [tree]
        else:
            left_trees = make_tree()
            right_trees = make_tree()
            children = left_trees + right_trees
            if label:
                tree = Tree(label[-1], children)
                for sublabel in reversed(label[:-1]):
                    tree = Tree(sublabel, [tree])
                return [tree]
            else:
                return children

    tree = make_tree()[0]
    tree.score = score
    return tree

def make_str_tree(sentence, tags, score, p_i, p_j, p_label):
    idx_cell = [-1]
    def make_str():
        idx_cell[0] += 1
        idx = idx_cell[0]
        i, j, label_idx = p_i[idx], p_j[idx], p_label[idx]
        label = LABEL_VOCAB[label_idx]
        if (i + 1) >= j:
            word = sentence[i]
            tag = TAG_VOCAB[tags[i]]
            tag = PTB_TOKEN_ESCAPE.get(tag, tag)
            word = PTB_TOKEN_ESCAPE.get(word, word)
            s = u"({} {})".format(tag, word)
        else:
            children = []
            while ((idx_cell[0] + 1) < len(p_i)
                and i <= p_i[idx_cell[0] + 1]
                and p_j[idx_cell[0] + 1] <= j):
                children.append(make_str())

            s = u" ".join(children)
            
        for sublabel in reversed(label):
            s = u"({} {})".format(sublabel, s)
        return s
    return make_str()

@app.get("/")
def index(request: Request):

    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/get_results")
async def get_results(request: Request, sentence: str = Form(...)):
    s = sentence.split()
    sentences = [s]
    i, m = make_feed_dict_bert(sentences)
    charts_val, tags_val = sess.run((charts, tags), {input_ids: i, word_end_mask: m})

    for snum, sentence in enumerate(sentences):
        chart_size = len(sentence) + 1
        chart = charts_val[snum,:chart_size,:chart_size,:]

    # make ntlk tree
    tree = make_nltk_tree(s, tags_val[0], *chart_decoder_py.decode(chart))
    print(str(tree))
    
    # get tree in sinlge line of string
    str_tree = make_str_tree(s, tags_val[0], *chart_decoder_py.decode(chart))
    print(str(str_tree))

    return templates.TemplateResponse("index.html", {"request": request, "results": str(tree)})
