from logging import Logger
import os
import sys
import time
import torch
import numpy as np
import random
from config import Config
from transformers import BertTokenizer, BertModel, AdamW
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from detection_model import Detection_model
import datetime
from sklearn.metrics import precision_recall_fscore_support
from event_sentence import cut_sent
from torch import tensor

from weight_score import weight_score_main


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def cut_sent_all(data):
    data_sen = []
    for new in tqdm(data):
        tp_new = new.replace("%","")
        tp_new = cut_sent(tp_new)
        data_sen.append(tp_new)
    return data_sen

def read_weights(path):
    weight_file = open(path,"r")
    lines = weight_file.readlines()
    temp = "tensor([])"
    sens_weights = []
    for line in lines:
        if line[0] == "t":
            sens_weights.append(eval(temp))
            temp = line.strip()
        else:
            temp = temp + line.strip()
    weight_file.close()
    sens_weights = sens_weights[1:]
    return sens_weights

def read_event_sens(path):
    event_sens_file = open(path,"r")
    lines = event_sens_file.readlines()
    event_sens = []
    for line in lines:
        event_sens.append(line.strip())
    return event_sens

def Test_fake_new(testset_path):
    
    # 初始化各种参数及导入预训练模型
    SEED = 777
    set_seed(SEED)
    cfg = Config()
    tokenizer = BertTokenizer.from_pretrained(cfg.bert_model)

    # 读取测试集
    Test_Data = pd.read_csv(testset_path)
    test_set = []
    test_labels = []
    for index, row in tqdm(Test_Data.iterrows()):
        text = row['content']
        test_set.append(text)
        test_labels.append(int(row['label']))
    #test_set, test_labels
    
    # 中间处理，以供模块2和模块3使用
    test_set = cut_sent_all(test_set)

    # 模块1，事件句子的提取 ==================================================================================================================
    Event_sentences = read_event_sens(cfg.pred_dataset_path+"event_sentence/event_sentence.txt")
    print("事件句子为:",Event_sentences)
    print("\n")

    # 模块2，测试集新闻句子权重的提取 ================================================================================================================
    print("② 新闻句子权重提取模块...\n")
    # test_sens_weights = []
    # test_New_id = 0
    # for New1 in tqdm(test_set):
    #     test_New_id += 1
    #     sen_weight1 = weight_score_main(cfg, Event_sentences, New1, test_New_id, "Test")
    #     test_sens_weights.append(sen_weight1)
    #
    # test_weight_file = open(cfg.pred_dataset_path+"weights/test_weights.txt","w")
    # for t_w in test_sens_weights:
    #     test_weight_file.write(str(t_w))
    #     test_weight_file.write("\n")
    # test_weight_file.close()
    test_sens_weights = read_weights("./dataset/topic_34/weights/test_weights.txt")
    # test_sens_weights = []
    # for New in tqdm(test_set):
    #     num_sen = len(New)
    #     wei = 1 / num_sen
    #     sen_weight = torch.tensor([wei for i in range(num_sen)])
    #     test_sens_weights.append(sen_weight)
    print("\n")

    # 模块3，最终分类 =========================================================================================================================
    print("③ 新闻最终分类...\n")
    ## 将权重补全到新闻句子数量最大值
    test_sens_weights_pad = []
    max_len = cfg.limit_num_sen                                          # 取句子数最大值，用于后续pad
    softmax_net = torch.nn.Softmax(dim=0)
    for weight in test_sens_weights:                                         # 补全权重集合
        if len(weight) < max_len:
            pad_num = max_len - len(weight)
            temp_weight = torch.cat((weight, torch.zeros(pad_num)),dim=0)
            test_sens_weights_pad.append(temp_weight)
            continue
        test_sens_weights_pad.append(softmax_net(weight[:max_len]))     # 将有切割的权重重新softmax
    
    test_sens_weights_pad = torch.stack(test_sens_weights_pad,0)
    
    ## 将测试数据、事件句子统一规格（batch_size, 50(max_len), 100(words_len)
    def convert_text_to_token(tokenizer, new_sen, limit_sens, limit_words): 
        result_new = []
        for sen in new_sen:
            tokens = tokenizer(sen, return_tensors='pt')  
            curr_len = len(tokens['input_ids'][0])
            if curr_len < limit_words:                       # 补齐（pad的索引号就是0）
                input_ids = torch.cat((tokens['input_ids'][0],torch.Tensor([0] * (limit_words - curr_len))),0)
                tokens['input_ids'] = [input_ids]
            result_new.append(tokens['input_ids'][0][:limit_words])
        if len(result_new) < limit_sens:
            need_pad = limit_sens - len(result_new)
            result_new.extend([torch.Tensor([0] * limit_words) for i in range(need_pad)])
        else:
            result_new = result_new[:limit_sens]
        result_new = torch.stack(result_new,0)
        return result_new

    test_inputs = [convert_text_to_token(tokenizer, new, max_len, cfg.limit_num_words) for new in test_set]

    ### tensor测试集输入
    test_inputs = torch.stack(test_inputs,0)
    test_labels = torch.tensor(test_labels)

    ### 处理事件句子，并复制多份
    event_sen_input = []
    for sen in Event_sentences:
        tokens = tokenizer(sen, return_tensors='pt')  
        curr_len = len(tokens['input_ids'][0])
        if curr_len < cfg.limit_num_words:                       # 补齐（pad的索引号就是0）
            input_ids = torch.cat((tokens['input_ids'][0],torch.Tensor([0] * (cfg.limit_num_words - curr_len))),0)
            tokens['input_ids'] = [input_ids]
        event_sen_input.append(tokens['input_ids'][0][:cfg.limit_num_words])
    event_sen_input = torch.stack(event_sen_input,0)
    event_sen_input = torch.unsqueeze(event_sen_input,0)

    test_event_sen_input = torch.repeat_interleave(event_sen_input, repeats=len(test_labels), dim=0)
    '''
        目前所有输入为:
        test_inputs                             文本的token输入     torch.Size([30, 50, 100])
        test_labels                             所有的标签信息      torch.Size([30])
        test_event_sen_input           事件句子的token输入 torch.Size([30, 5, 100])
        test_sens_weights_pad         所有新闻句子权重    torch.Size([30, 50])
    '''

    ## 构建dataset和dataloader类，用于管理及训练模型时抽取batch
    class MyDataSet(Dataset):
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, item):
            assert item < len(self.data)
            data = self.data[item]
            event = data[0]
            input = data[1]
            weight = data[2]
            labels = data[3]
            return event, input, weight, labels
    test_data = MyDataSet(list(zip(test_event_sen_input,test_inputs,test_sens_weights_pad,test_labels)))
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=cfg.batch_size)

    ## 加载训练好的模型
    My_model = Detection_model(cfg).to(cfg.device)
    My_model.load_state_dict(torch.load(cfg.model_path,map_location=cfg.device))

    ## 训练过程
    def format_time(elapsed):
        elapsed_rounded = int(round(elapsed))
        return str(datetime.timedelta(seconds = elapsed_rounded)) # 返回 hh:mm:ss 形式的时间
    def binary_acc(preds, labels): # preds.shape = [16, 2] labels.shape = [16, 1]
        # torch.max: [0]为最大值, [1]为最大值索引
        correct = torch.eq(torch.max(preds, dim=1)[1], labels.flatten()).float()
        acc = correct.sum().item() / len(correct)
        return acc

    ### 模型验证
    test_label_batch, test_output_batch = [], []
    t0 = time.time()
    My_model.eval()         # 表示进入测试模式
    with torch.no_grad():
        for step, batch  in enumerate(test_dataloader):
            # 每隔1个batch 输出一下所用时间.
            if step % 1 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(test_dataloader), elapsed))

            test_event_sen_input = batch[0].long().to(cfg.device)
            test_new_input = batch[1].long().to(cfg.device)
            test_weight_input = batch[2].float().to(cfg.device)
            test_label = batch[3].long().to(cfg.device)

            output = My_model(test_event_sen_input,test_new_input,test_weight_input)

            test_label_batch.append(test_label)
            test_output_batch.append(output)

    test_label_batch = torch.cat(test_label_batch,0)
    test_output_batch = torch.cat(test_output_batch,0)
    test_label_ = test_label_batch.cpu().detach().numpy().tolist()
    test_output_ = [np.argmax(outp) for outp in test_output_batch.cpu().detach().numpy()]
    test_avg_prec, test_avg_rec, test_avg_f1, _ = precision_recall_fscore_support(test_label_, test_output_, average="weighted")
    test_avg_acc = binary_acc(test_output_batch,test_label_batch.unsqueeze(1))
    print(test_label_)
    print(test_output_)

    print('Test_set:acc={:.4f}，prec={:.4f}，rec={:.4f}，f1={:.4f}'.format(test_avg_acc, test_avg_prec, test_avg_rec, test_avg_f1))

if __name__ == '__main__':
    Test_fake_new("./dataset/topic_34/test_data.csv")
