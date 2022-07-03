import torch
from transformers import BertTokenizer, BertModel
import torch.nn.functional as F


class Detection_model(torch.nn.Module):
    def __init__(self, config):
        super(Detection_model, self).__init__()
        # 加载在自己的语料库上训练过的bert模型
        self.bertmodel = BertModel.from_pretrained(config.bert_model)
        pretrained_dict = torch.load(config.bert_pre_pra, map_location=config.device)
        model_dict = self.bertmodel.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.bertmodel.load_state_dict(model_dict)

        self.Bi_GRU = torch.nn.GRU(768, 768, batch_first=True, bidirectional=False)
        # ******************************************
        self.MLP = torch.nn.Sequential(
            torch.nn.Linear(768 * 2, 768, bias=True),
            torch.nn.Dropout(config.dropout),
            torch.nn.ReLU(),
            torch.nn.Linear(768, config.num_labels, bias=True)
        )

    def forward(self, event_sentence, new, new_sen_weights):
        encode_event = [self.bertmodel(event_sen) for event_sen in event_sentence]
        # output_add = torch.mean(encode_output.last_hidden_state[:,1:-1,:], dim=1)       # 取每个单词的平均值聚合成句子嵌入
        encode_event = [encode_e.last_hidden_state[:, 0, :] for encode_e in encode_event]  # 取cls标识的嵌入作为句子嵌入
        encode_event = torch.stack(encode_event, 0)

        encode_new = [self.bertmodel(n) for n in new]
        # output_add = torch.mean(encode_output.last_hidden_state[:,1:-1,:], dim=1)  # 取每个单词的平均值聚合成句子嵌入
        encode_new = [encode_n.last_hidden_state[:, 0, :] for encode_n in encode_new]  # 取cls标识的嵌入作为句子嵌入
        encode_new = torch.stack(encode_new, 0)

        encode_new, _ = self.Bi_GRU(encode_new)

        # print("encode_event:",encode_event.shape)     # torch.Size([4, 5, 768])
        # print("encode_new:",encode_new.shape)         # torch.Size([4, 23, 768])
        diff = []
        for encode_new_one, encode_event_one in zip(encode_new, encode_event):
            diff_one = []
            for event in encode_event_one:
                diff_one_event = []
                for new in encode_new_one:
                    diff_one_event.append(new.sub(event))  # 暂用最简单的减法
                diff_one_event = torch.stack(diff_one_event, 0)
                diff_one.append(diff_one_event)
            diff_one = torch.stack(diff_one, 0)
            diff.append(diff_one)
        diff = torch.stack(diff, 0)
        # print("diff:",diff.shape)                      # torch.Size([4, 5, 23, 768])

        diff = torch.sum(diff, dim=1)
        # print("diff:",diff.shape)                      # torch.Size([4, 23, 768])

        D = torch.mul(diff, new_sen_weights.unsqueeze(-1))
        D = torch.sum(D, dim=1)
        # print("D：", D.shape)                          # torch.Size([4, 768])
        E = torch.mul(encode_new, new_sen_weights.unsqueeze(-1))
        E = torch.sum(E, dim=1)
        # print("E:",E.shape)                            # torch.Size([4, 768])

        New = torch.cat((D, E), dim=1)
        # print("New:",New.shape)                        # torch.Size([4, 1536])
        # x = F.relu(self.hidden_1(New))
        # print("x:",x.shape)                            # torch.Size([4, 768])
        # logits = self.hidden_2(x)
        # print("logits:",logits.shape)                  # torch.Size([4, 2])
        logits = self.MLP(New)
        return logits
