import numpy as np
import random
import torch
import matplotlib.pylab as plt 
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, BertModel
from transformers import get_linear_schedule_with_warmup
import pandas as pd
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn

SEED = 777
BATCH_SIZE = 10
learning_rate = 2e-5
weight_decay = 1e-2
epsilon = 1e-8
limit_size = 510
epochs = 16

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

pos_text = []
neg_text = []
chongqing_bus = pd.read_csv("./dataset/chongqing_bus_event.csv",encoding="gbk")
# 提取每一条真实新闻的事件句子
for index, row in tqdm(chongqing_bus.iterrows()):
    if not pd.isnull(row['title']):
        text = "【"+row['title']+"】"+row['text']
    elif pd.isnull(row['title']):
        text = row['text']
    if row['label'] == int(1):
        neg_text.append(text)
    elif row['label'] == int(0):
        pos_text.append(text)
sentences = pos_text + neg_text


# 设定标签
pos_targets = np.zeros([len(pos_text)])  # (19, )
neg_targets = np.ones([len(neg_text)]) # (21, )
targets = np.concatenate((pos_targets, neg_targets), axis=0).reshape(-1, 1) # (40, 1)
total_targets = torch.tensor(targets)

model_name = './chinese-bert-wwm-ext'

tokenizer = BertTokenizer.from_pretrained(model_name)

from sklearn.model_selection import train_test_split

def convert_text_to_token(tokenizer, sentence, limit_size = 510):
    tokens = tokenizer(sentence[:limit_size], return_tensors='pt')       # 直接截断
    curr_len = len(tokens['input_ids'][0])
    if curr_len < limit_size + 2:                       # 补齐（pad的索引号就是0）
        input_ids = torch.cat((tokens['input_ids'][0],torch.Tensor([0] * (limit_size + 2 - curr_len))),0)
        tokens['input_ids'] = [input_ids]
    return tokens['input_ids'][0]
total_inputs = [convert_text_to_token(tokenizer, sen, limit_size) for sen in sentences]

train_inputs, test_inputs, train_labels, test_labels = train_test_split(total_inputs, total_targets, 
                                                                        random_state=SEED, test_size=0.2)

class MyDataSet(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        assert item < len(self.data)
        data = self.data[item]
        input = data[0]
        labels = data[1]
        return input, labels

train_data = MyDataSet(list(zip(train_inputs,train_labels)))
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)

test_data = MyDataSet(list(zip(test_inputs, test_labels)))
test_sampler = RandomSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=BATCH_SIZE)

class My_model(torch.nn.Module):
    def __init__(self,model_name,num_labels):
        super(My_model,self).__init__()
        self.bertmodel = BertModel.from_pretrained(model_name) # num_labels表示2个分类,真实新闻和虚假新闻
        self.hidden_1 = torch.nn.Linear(768, 768, bias=True)
        self.hidden_2 = torch.nn.Linear(768, num_labels, bias=True)

    def forward(self, x):
        encode_output = self.bertmodel(x)
        # output_add = torch.mean(encode_output.last_hidden_state[:,1:-1,:], dim=1)  # 取每个单词的平均值聚合成句子嵌入
        output_add = encode_output.last_hidden_state[:,0,:]                          # 取cls标识的嵌入作为句子嵌入
        x = F.relu(self.hidden_1(output_add))
        logists = self.hidden_2(x)
        return logists

model = My_model(model_name, num_labels = 2) # num_labels表示2个分类,真实新闻和虚假新闻
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

optimizer = AdamW(model.parameters(), lr = learning_rate, eps = epsilon)
loss = nn.CrossEntropyLoss()


# training steps 的数量: [number of batches] x [number of epochs].
total_steps = len(train_dataloader) * epochs      # 8*2

# 设计 learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, 
                                            num_training_steps = total_steps)


def binary_acc(preds, labels): # preds.shape = [16, 2] labels.shape = [16, 1]
    # torch.max: [0]为最大值, [1]为最大值索引
    correct = torch.eq(torch.max(preds, dim=1)[1], labels.flatten()).float()
    acc = correct.sum().item() / len(correct)
    return acc

import time
import datetime

def format_time(elapsed):
    elapsed_rounded = int(round(elapsed))
    return str(datetime.timedelta(seconds = elapsed_rounded)) # 返回 hh:mm:ss 形式的时间

def train(model, optimizer):
    t0 = time.time()
    avg_loss, avg_acc = [],[]

    model.train()
    for step, batch in enumerate(train_dataloader):

        # 每隔1个batch 输出一下所用时间.
        if step % 1 == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

        b_input_sens, b_labels = batch[0].long().to(device), batch[1].long().to(device)

        output = model(b_input_sens)
        logits = output      # logits: predict

        my_loss=loss(logits,np.squeeze(b_labels))
        avg_loss.append(my_loss.item())

        acc = binary_acc(logits, b_labels)       # (predict, label)
        avg_acc.append(acc)

        optimizer.zero_grad()
        my_loss.backward()
        clip_grad_norm_(model.parameters(), 1.0) # 大于1的梯度将其设为1.0, 以防梯度爆炸
        optimizer.step()                         # 更新模型参数
        scheduler.step()                         # 更新learning rate

    avg_acc = np.array(avg_acc).mean()
    avg_loss = np.array(avg_loss).mean()
    torch.save(model.state_dict(),'./model/my_pre_bert.pth')
    return avg_loss, avg_acc

def evaluate(model):
    avg_acc = []
    model.eval()         # 表示进入测试模式

    with torch.no_grad():
        for batch in test_dataloader:
            b_input_sens, b_labels = batch[0].long().to(device), batch[1].long().to(device)

            output = model(b_input_sens)

            acc = binary_acc(output, b_labels)
            avg_acc.append(acc)

    avg_acc = np.array(avg_acc).mean()
    return avg_acc

for epoch in range(epochs):

    train_loss, train_acc = train(model, optimizer)
    print('epoch={},训练准确率={}，损失={}'.format(epoch, train_acc, train_loss))
    
    test_acc = evaluate(model)
    print("epoch={},测试准确率={}".format(epoch, test_acc))

def predict(sen):
    input = convert_text_to_token(tokenizer, sen, limit_size)
    print(input.shape)
    input = input.view(1, -1).long().to(device)
    print(input.shape)
    output = model(input)     #torch.Size([128])->torch.Size([1, 128])否则会报错
    print(output)

    return torch.max(output, dim=1)[1]

text = """【坠江事故女司机没逆行！重庆公交车坠江因突然越线，车上共有驾乘人员10余人】警方初步确认！重庆坠江公交上共有驾乘人员10多人。警方通报：10月28日10时08分，一辆公交客车与一辆小轿车在重庆万州区长江二桥相撞后，公交客车坠入江中。经初步事故现场调查，系公交客车在行驶中突然越过中心实线，撞击对向正常行驶的小轿车后冲上路沿，撞断护栏，坠入江中。"""

label = predict(text)

print('虚假新闻' if label==1 else '真实新闻')

# print(model.state_dict())