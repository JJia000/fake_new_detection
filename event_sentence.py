from transformers import BertTokenizer, BertModel
from rouge import Rouge
from sklearn.metrics.pairwise import cosine_similarity
import torch
import pandas as pd
from tqdm import tqdm
import heapq
import re
import numpy as np

# 中文分句函数
def cut_sent(raw_text):
    sens_List = []
    query = re.finditer("\【.*?\】", raw_text, re.I|re.M)
    index_list = []
    for i in query:
        index_list.append(i.span()[0])
        index_list.append(i.span()[1])
    if len(index_list) == 0:
        sens_List.append(raw_text)
    else:
        for i in range(len(index_list)):
            if i == 0:
                sens_List.append(raw_text[0:index_list[i]])
            else:
                sens_List.append(raw_text[index_list[i-1]:index_list[i]])
                if i == len(index_list)-1:
                    sens_List.append(raw_text[index_list[i]:])
                    break
    sentenceList = []
    for sen in sens_List:
        if sen != "":
            if sen[0] == "【":
                sentenceList.append(sen)
            else:
                temp_sens = cut_sent_small(sen)
                sentenceList.extend(temp_sens)
    return sentenceList

def cut_sent_small(raw_text):
    cutLineFlag = ["？", "！", "。","…","】"] #本文使用的终结符，可以修改
    sentenceList = []
    oneSentence = ""
    words = raw_text.strip()
    if len(oneSentence)!=0:
        sentenceList.append(oneSentence.strip() + "\r")
        oneSentence=""
    # oneSentence = ""
    for word in words:
        if word not in cutLineFlag:
            oneSentence = oneSentence + word
        else:
            oneSentence = oneSentence + word
            if oneSentence.__len__() > 4:
                sentenceList.append(oneSentence.strip() + "\r")
            oneSentence = ""
    return sentenceList

# 归一化
def MaxMinNormalization(x):
    small_e = 1e-6
    Max_x = max(x)
    Min_x = min(x)
    x_result = []
    for x_temp in x:
        x_result.append((x_temp - Min_x) / (Max_x - Min_x + small_e))
    return x_result

# 提取一条新闻的事件句子,k为返回几条事件句子，默认为1
def event_ext(text,rouge,tokenizer,model,alpha,beta,k=1,one_new=True):
    if type(text) == list:
        Sentence_list = text
    else:
        Sentence_list = cut_sent(text)

    # 计算每个句子与其他句子的ROUGE值
    rouge_list = []
    rouge_no_len_list = []
    for Sentence in Sentence_list:
        curr_sen = Sentence
        # print(curr_sen)
        temp_rouge = 0.0
        temp_no_len_rouge = 0.0
        for sen in Sentence_list:
            if sen == curr_sen:
                continue
            rouge_score = rouge.get_scores(' '.join(list(curr_sen)),' '.join(list(sen)))
            temp_rouge += rouge_score[0]["rouge-l"]['r']/len(curr_sen)                    # 除以句子长度，可以保证在句子最短的情况下反映最多信息
            temp_no_len_rouge += rouge_score[0]["rouge-l"]['r']
        rouge_list.append(temp_rouge)
        rouge_no_len_list.append(temp_no_len_rouge)
    # print(rouge_list)

    # 计算每个句子与其他句子的余弦相似度
    # 先求每个句子的嵌入
    sentence_emb_list = []
    for Sentence in Sentence_list:
        encoded_input = tokenizer(Sentence, return_tensors='pt')
        output = model(**encoded_input)
        # sentence_emb_list.append(output[0][:,0,:])    # 取bert的<cls>符号嵌入作为句子嵌入
        output_add = torch.mean(output.last_hidden_state[:,1:-1,:], dim=1)
        sentence_emb_list.append(output_add)            # 取每个单词的平均值聚合成句子嵌入

    cosine_list = []
    for sentence_emb in sentence_emb_list:
        curr_sen_emb = sentence_emb
        temp_cosine = 0.0
        for emb in sentence_emb_list:
            if emb.equal(curr_sen_emb):
                continue
            cosine_score = cosine_similarity(curr_sen_emb.detach().numpy(),emb.detach().numpy())
            temp_cosine += cosine_score[0][0]
        cosine_list.append(temp_cosine)
    # print(cosine_list)

    # 计算每个句子的总得分
    score_list = []
    rouge_list = MaxMinNormalization(rouge_list)
    cosine_list = MaxMinNormalization(cosine_list)
    if one_new:
        for i in range(len(rouge_list)):
            score_list.append(alpha*rouge_list[i]+beta*cosine_list[i])
    else:
        for i in range(len(rouge_list)):
            score_list.append(rouge_list[i])                                 # 在一系列事件句子中提取总体事件时，仅需考虑独特性

    # 取得分最高的一句，或k句
    if one_new:
        score_list[0] += 0.3       # 若是第一句，提高0.3得分
        print("一条新闻：")
        for i in range(len(Sentence_list)):
            print(Sentence_list[i])
            print("cosine:",cosine_list[i])
            print("rouge:",rouge_no_len_list[i])
            print("rouge\len(sen):",rouge_list[i])
            print("score:",score_list[i])
            print("\n")
        max_index = score_list.index(max(score_list))
        return Sentence_list[max_index]
    else:
        # for i in range(len(Sentence_list)):
        #     print(Sentence_list[i])
        #     print("cosine:",cosine_list[i])
        #     print("rouge:",rouge_list[i]*len(Sentence_list[i]))
        #     print("rouge\len(sen):",rouge_list[i])
        #     print("score:",score_list[i])
        #     print("\n")
        # 去掉所有重复的事件句子
        score_list_single = list(set(score_list))
        result_index = list(map(score_list_single.index, heapq.nsmallest(k+5,score_list_single)))     # 多取五个，避免那些完全不和事件字符重叠的事件句子
        result_index = result_index[5:]

        # result_index = list(map(score_list.index, heapq.nsmallest(k,score_list)))
        result_sens = []
        result_scores = []
        for i in result_index:
            index = score_list.index(score_list_single[i])
            result_sens.append(Sentence_list[index])
            result_scores.append(score_list[index])
        return result_sens

# 事件句子提取的main函数
def event_sentence_main(config):
    rouge = Rouge()
    tokenizer = BertTokenizer.from_pretrained(config.bert_model)
    model = BertModel.from_pretrained(config.bert_model)
    pretrained_dict = torch.load(config.bert_pre_pra)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    alpha = config.alpha        # rouge权重
    beta = config.beta     # 语义余弦相似度权重

    all_event_sent = []
    chongqing_bus = pd.read_csv(config.data_path,encoding="gbk")
    # 提取每一条真实新闻的事件句子
    for index, row in tqdm(chongqing_bus.iterrows()):
        if row['label'] == int(1):
            continue
        else:
            if not pd.isnull(row['title']):
                text = "【"+row['title']+"】"+row['text']
            else:
                text = row['text']
            event_sen = event_ext(text,rouge,tokenizer,model,alpha,beta)
            all_event_sent.append(event_sen)

    # 提取所有真实新闻的总体事件句子（可不局限于一条）
    event_sen_final = event_ext(all_event_sent,rouge,tokenizer,model,alpha,beta,k=config.k,one_new=False)
    print(event_sen_final)
    return event_sen_final




    
