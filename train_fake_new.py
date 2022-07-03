import os
from event_sentence import event_sentence_main, cut_sent
from config import Config
from weight_score import weight_score_main, data_prepare
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch
from transformers import BertTokenizer, BertModel, AdamW
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader, RandomSampler
from detection_model import Detection_model
import torch.nn as nn
import time
import datetime
from sklearn.metrics import precision_recall_fscore_support
from torch.nn.utils import clip_grad_norm_
from torch import tensor
import time
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
from earlystopping import EarlyStopping
from test_fake_new import Test_fake_new

class Logger(object):
    def __init__(self, filename="log.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")
 
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
 
    def flush(self):
        pass

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def convert_text_to_token(tokenizer, sentence, limit_size):
    tokens = tokenizer(sentence, return_tensors='pt')       # 直接截断
    tokens['input_ids'] = tokens['input_ids'][0][:limit_size]
    curr_len = len(tokens['input_ids'])
    if curr_len < limit_size + 2:                       # 补齐（pad的索引号就是0）
        input_ids = torch.cat((tokens['input_ids'],torch.Tensor([0] * (limit_size + 2 - curr_len))),0)
        tokens['input_ids'] = input_ids
    return tokens['input_ids']

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

def save_train_test(data,label,path,test=False):
    save_file = pd.DataFrame({'content':data,'label':label})
    if test:
        save_file.to_csv(path+"test_data.csv",index=False)
        return
    save_file.to_csv(path+"train_data.csv",index=False)

if __name__ == '__main__':
    # 将输出重定向到指定文件夹
    cur_time = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime())
    save_path = "./result/"+cur_time
    os.mkdir(save_path)

    # path = os.path.abspath(os.path.dirname(__file__))
    # type = sys.getfilesystemencoding()
    # sys.stdout = Logger(save_path+"/log.txt")

    # 初始化各种参数及导入预训练模型
    SEED = 777
    set_seed(SEED)
    cfg = Config()
    tokenizer = BertTokenizer.from_pretrained(cfg.bert_model)

    # 读取数据集、划分训练集和测试集
    Data = pd.read_csv(cfg.data_path)
    Data_content = []
    Data_label = []
    for index, row in tqdm(Data.iterrows()):
        if not pd.isnull(row['title']):
            text = "【"+row['title']+"】"+row['content']
        else:
            text = row['content']
        Data_content.append(text)
        Data_label.append(int(row['label']))
    train_set, test_set, train_labels, test_labels = train_test_split(Data_content, Data_label, 
                                                                        random_state=SEED, test_size=cfg.test_size)
    print("已保存训练测试集数据")
    save_train_test(train_set, train_labels,cfg.pred_dataset_path)
    save_train_test(test_set, test_labels,cfg.pred_dataset_path,test=True)
    
    # 中间处理，以供模块1,模块2和模块3使用
    train_set = cut_sent_all(train_set)
    # test_set = cut_sent_all(test_set)

    # 模块1，事件句子的提取 ==================================================================================================================
    print("① 事件句子提取模块...\n")
    # 首先分离出训练集中的真实新闻
    true_news = []
    for i in range(len(train_labels)):
        if train_labels[i] == int(0):
            true_news.append(train_set[i])
    # 提取真实新闻中的事件句子
    Event_sentences = event_sentence_main(cfg,true_news)
    # Event_sentences = []
    # Event_new = random.sample(true_news,5)
    # for ne in Event_new:
    #     Event_sentences.append(random.choice(ne))
    Event_sentences_file = open("./dataset/topic_34/event_sentence/event_sentence.txt","w")
    for sen in Event_sentences:
        Event_sentences_file.write(str(sen))
        Event_sentences_file.write("\n")
    Event_sentences_file.close()
    # Event_sentences = read_event_sens("./dataset/topic_34/event_sentence/event_sentence.txt")
    print("事件句子为:", Event_sentences)
    print("\n")

    # 模块2，新闻句子权重的提取 ================================================================================================================
    print("② 新闻句子权重提取模块...\n")
    # 首先data_prepare事件句子
    Sentences_ids = ["event_{}".format(i) for i in range(len(Event_sentences))]
    prefix = "Event"
    event_token_dict, event_deny_dict, event_tri_entity_dict, event_number_dict, event_comparative_dict = \
        data_prepare(Event_sentences, Sentences_ids, prefix, cfg.words_path, cfg.dict_sets)

    train_sens_weights = []
    train_New_id = 0
    for New in tqdm(train_set):
        train_New_id += 1
        sen_weight = weight_score_main(cfg, Event_sentences, New, train_New_id, "Train")
        train_sens_weights.append(sen_weight)

    train_weight_file = open("./dataset/topic_34/weights/train_weights.txt","w")
    for t_w in train_sens_weights:
        train_weight_file.write(str(t_w))
        train_weight_file.write("\n")
    train_weight_file.close()
    # train_sens_weights = read_weights("./dataset/topic_34/weights/train_weights.txt")
    # train_sens_weights = []
    # for New in tqdm(train_set):
    #     num_sen = len(New)
    #     wei = 1/num_sen
    #     sen_weight = torch.tensor([wei for i in range(num_sen)])
    #     train_sens_weights.append(sen_weight)
    print("\n")

    # 模块3，最终分类 =========================================================================================================================
    print("③ 新闻最终分类...\n")
    train_num = len(train_set)
    val_num = int(train_num*cfg.val_size)
    # test_num = len(test_set)
    ## 将权重补全到新闻句子数量最大值
    total_sens_weights_pad = []
    # total_weights = train_sens_weights + test_sens_weights
    total_weights = train_sens_weights
    max_len = cfg.limit_num_sen                                          # 取句子数最大值，用于后续pad
    softmax_net = torch.nn.Softmax(dim=0)
    for weight in total_weights:                                         # 补全权重集合
        if len(weight) < max_len:
            pad_num = max_len - len(weight)
            temp_weight = torch.cat((weight, torch.zeros(pad_num)),dim=0)
            total_sens_weights_pad.append(temp_weight)
            continue
        total_sens_weights_pad.append(softmax_net(weight[:max_len]))     # 将有切割的权重重新softmax
    
    total_sens_weights_pad = torch.stack(total_sens_weights_pad,0)
    train_sens_weights_pad = total_sens_weights_pad[:train_num-val_num]
    val_sens_weights_pad = total_sens_weights_pad[train_num-val_num:train_num]
    # test_sens_weights_pad = total_sens_weights_pad[train_num:]
    
    ## 将训练数据、验证数据和测试数据、事件句子统一规格（batch_size, 50(max_len), 100(words_len)
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

    total_data = train_set
    total_inputs = [convert_text_to_token(tokenizer, new, max_len, cfg.limit_num_words) for new in total_data]

    ### 划分训练、验证和测试集
    total_inputs = torch.stack(total_inputs,0)
    train_inputs = total_inputs[:train_num-val_num]
    val_inputs = total_inputs[train_num-val_num:train_num]

    total_labels = torch.tensor(train_labels)
    train_labels = total_labels[:train_num-val_num]
    val_labels = total_labels[train_num-val_num:train_num]

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

    train_event_sen_input = torch.repeat_interleave(event_sen_input, repeats=len(train_labels), dim=0)
    val_event_sen_input = torch.repeat_interleave(event_sen_input, repeats=len(val_labels), dim=0)
    # test_event_sen_input = torch.repeat_interleave(event_sen_input, repeats=len(test_labels), dim=0)
    '''
        目前所有输入为:
        train_inputs、val_inputs                             文本的token输入     torch.Size([30, 50, 100])
        train_labels、val_labels                             所有的标签信息      torch.Size([30])
        train_event_sen_input、val_event_sen_input           事件句子的token输入 torch.Size([30, 5, 100])
        train_sens_weights_pad、val_sens_weights_pad        所有新闻句子权重    torch.Size([30, 50])
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
    
    train_data = MyDataSet(list(zip(train_event_sen_input,train_inputs,train_sens_weights_pad,train_labels)))
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=cfg.batch_size)

    val_data = MyDataSet(list(zip(val_event_sen_input,val_inputs,val_sens_weights_pad,val_labels)))
    val_sampler = RandomSampler(val_data)
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=cfg.batch_size)

    # test_data = MyDataSet(list(zip(test_event_sen_input,test_inputs,test_sens_weights_pad,test_labels)))
    # test_sampler = RandomSampler(test_data)
    # test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=cfg.batch_size)

    ## 实例化模型、优化器、损失、早停法等
    My_model = Detection_model(cfg).to(cfg.device)
    Optimizer = AdamW(My_model.parameters(), lr = cfg.learning_rate, eps = cfg.epsilon,weight_decay=cfg.weight_decay)
    Loss = nn.CrossEntropyLoss()
    early_stopping = EarlyStopping(patience=cfg.patience, verbose=True, model_path=cfg.model_path)

    ## 训练过程
    def format_time(elapsed):
        elapsed_rounded = int(round(elapsed))
        return str(datetime.timedelta(seconds = elapsed_rounded)) # 返回 hh:mm:ss 形式的时间
    def binary_acc(preds, labels): # preds.shape = [16, 2] labels.shape = [16, 1]
        # torch.max: [0]为最大值, [1]为最大值索引
        correct = torch.eq(torch.max(preds, dim=1)[1], labels.flatten()).float()
        acc = correct.sum().item() / len(correct)
        return acc

    torch.set_num_threads(8)
    train_loss_plt = []
    val_loss_plt = []
    train_acc_plt = []
    train_pre_plt = []
    train_rec_plt = []
    train_F1_plt = []
    val_acc_plt = []
    val_pre_plt = []
    val_rec_plt = []
    val_F1_plt = []
    for epoch in range(cfg.epochs):

        ### 模型训练
        t0 = time.time()
        My_model.train()
        avg_loss = 0.0
        label_batch = []
        logit_batch = []
        for step, batch in enumerate(train_dataloader):
            # 每隔1个batch 输出一下所用时间.
            if step % 1 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))
            
            event_sen_input = batch[0].long().to(cfg.device)
            new_input = batch[1].long().to(cfg.device)
            weight_input = batch[2].float().to(cfg.device)
            label = batch[3].long().to(cfg.device)

            logits = My_model(event_sen_input,new_input,weight_input)

            my_loss=Loss(logits,label)
            # avg_loss.append(my_loss.item())
            avg_loss += label.size()[0] * my_loss.item()

            label_batch.append(label)
            logit_batch.append(logits)

            Optimizer.zero_grad()
            my_loss.backward()
            clip_grad_norm_(My_model.parameters(), 1.0) # 大于1的梯度将其设为1.0, 以防梯度爆炸
            Optimizer.step()                            # 更新模型参数
        
        label_batch = torch.cat(label_batch,0)
        logit_batch = torch.cat(logit_batch,0)
        label_ = label_batch.cpu().detach().numpy().tolist()
        logits_ = [np.argmax(logi) for logi in logit_batch.cpu().detach().numpy()]
        avg_prec, avg_rec, avg_f1, _ = precision_recall_fscore_support(label_, logits_, average="weighted")
        avg_acc = binary_acc(logit_batch,label_batch.unsqueeze(1))

        avg_loss /= train_num - val_num
        train_loss_plt.append(avg_loss)
        train_acc_plt.append(avg_acc)
        train_pre_plt.append(avg_prec)
        train_rec_plt.append(avg_rec)
        train_F1_plt.append(avg_f1)
        print('epoch={},acc={:.4f}，prec={:.4f}，rec={:.4f}，f1={:.4f}，loss={:.4f}'.format(epoch, avg_acc, avg_prec, avg_rec, avg_f1, avg_loss))

        ### 模型验证
        val_avg_loss = 0.0
        val_label_batch, val_output_batch = [], []
        My_model.eval()         # 表示进入测试模式
        with torch.no_grad():
            for batch in val_dataloader:
                val_event_sen_input = batch[0].long().to(cfg.device)
                val_new_input = batch[1].long().to(cfg.device)
                val_weight_input = batch[2].float().to(cfg.device)
                val_label = batch[3].long().to(cfg.device)

                output = My_model(val_event_sen_input,val_new_input,val_weight_input)

                val_my_loss=Loss(output,val_label)
                # val_avg_loss.append(val_my_loss.item())
                val_avg_loss += val_label.size()[0] * val_my_loss.item()

                val_label_batch.append(val_label)
                val_output_batch.append(output)

        val_label_batch = torch.cat(val_label_batch,0)
        val_output_batch = torch.cat(val_output_batch,0)
        val_label_ = val_label_batch.cpu().detach().numpy().tolist()
        val_output_ = [np.argmax(outp) for outp in val_output_batch.cpu().detach().numpy()]
        val_avg_prec, val_avg_rec, val_avg_f1, _ = precision_recall_fscore_support(val_label_, val_output_, average="weighted")
        val_avg_acc = binary_acc(val_output_batch,val_label_batch.unsqueeze(1))

        val_avg_loss /= val_num
        val_loss_plt.append(val_avg_loss)
        val_acc_plt.append(val_avg_acc)
        val_pre_plt.append(val_avg_prec)
        val_rec_plt.append(val_avg_rec)
        val_F1_plt.append(val_avg_f1)
        print('epoch={},acc={:.4f}，prec={:.4f}，rec={:.4f}，f1={:.4f}，loss={:.4f}'.format(epoch, val_avg_acc, val_avg_prec, val_avg_rec, val_avg_f1, val_avg_loss))

        early_stopping(val_avg_loss, My_model)
        # 若满足 early stopping 要求
        if early_stopping.early_stop:
            print("Early stopping!")
            # 结束模型训练
            break

    # 画图，loss趋势图和各个指标图
    plt.figure(1)
    plt.title('The loss of Train and Validation')
    x_axis = range(len(train_loss_plt))
    plt.plot(x_axis, train_loss_plt, 'p-', color='blue', label='Train_loss')
    plt.plot(x_axis, val_loss_plt, '*-', color='orange', label='val_loss')
    plt.legend() # 显示图例
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.savefig(save_path+"/Loss_fig.png")

    plt.figure(2)
    plt.title('Train_metric')
    x_axis = range(len(train_acc_plt))
    plt.plot(x_axis, train_acc_plt, '.-', color='blue', label='train_acc')
    plt.plot(x_axis, train_pre_plt, '^-', color='orange', label='train_precision')
    plt.plot(x_axis, train_rec_plt, 'p-', color='pink', label='train_recall')
    plt.plot(x_axis, train_F1_plt, '*-', color='green', label='train_F1')
    plt.legend() # 显示图例
    plt.xlabel('epochs')
    plt.ylabel('Train_metric')
    plt.savefig(save_path+"/Train_metric.png")

    plt.figure(3)
    plt.title('Validation_metric')
    x_axis = range(len(val_acc_plt))
    plt.plot(x_axis, val_acc_plt, '.-', color='blue', label='val_acc')
    plt.plot(x_axis, val_pre_plt, '^-', color='orange', label='val_precision')
    plt.plot(x_axis, val_rec_plt, 'p-', color='pink', label='val_recall')
    plt.plot(x_axis, val_F1_plt, '*-', color='green', label='val_F1')
    plt.legend() # 显示图例
    plt.xlabel('epochs')
    plt.ylabel('Validation_metric')
    plt.savefig(save_path+"/Validation_metric.png")

    with open("config.py","r") as f_con:
        with open(save_path+"/config.txt","w") as f1_wri:
            lines = f_con.readlines()
            for line in lines:
                f1_wri.write(line)
    f_con.close()
    f1_wri.close()

    ## 进行测试
    print("\n进行测试!")
    Test_fake_new("./dataset/topic_34/test_data.csv")







    

