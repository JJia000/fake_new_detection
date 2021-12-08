from event_sentence import event_sentence_main, cut_sent
from config import Config
from rouge import Rouge
from ddparser import DDParser
import warnings
import torch
import torch.nn.functional as F
import math
warnings.filterwarnings("ignore")

# 读取词库
def read_words(words_detail_path,col=1):
    words_dict = {}
    with open(words_detail_path,'r',encoding='utf8') as f:
        lines = f.readlines()
        i = 0
        for line in lines:
            line = line.strip().split()
            if col == 1:
                words_dict[line[0]] = i
                i = i + 1
            else:
                words_dict[line[0]] = line[1]
    f.close()
    return words_dict

# 进行预处理，提取词语、否定词、实体-实体关系-实体三元组、数字、比较级
def data_prepare(Sentences,Sentences_ids,words_path):
    # 提取实体-实体关系-实体三元组，并且同时获得分词的词语
    ddp = DDParser()
    tri_entity_dict = {}
    token_dict = {}
    for i in range(len(Sentences)):
        tri_entity_list = ddp.parse(Sentences[i])[0]     # {word:[],head:[],deprel:[]}
        token_list = tri_entity_list['word']
        tri_entity_dict[Sentences_ids[i]] = tri_entity_list
        token_dict[Sentences_ids[i]] = token_list
    # 提取否定词
    deny_dict = {}
    deny_words_dict = read_words(words_path+'deny_words.txt')
    for k,v in token_dict.items():
        deny_list = [word for word in v if word in deny_words_dict.keys()]
        deny_dict[k] = deny_list
    # 提取数字
    number_dict = {}
    number_words_dict = read_words(words_path+'number_words.txt')
    for k,v in token_dict.items():
        number_list = []
        for word in v:
            word1 = list(word)
            for w in word1:
                if w in number_words_dict.keys():
                    number_list.append(word)
                    break
                else:
                    continue
        number_dict[k] = number_list

    # 提取比较级词
    comparative_dict = {}
    comparative_words_dict = read_words(words_path+'comparative_words.txt',col=2)
    for k,v in token_dict.items():
        comparative_list = [word for word in v if word in comparative_words_dict.keys()]
        comparative_dict[k] = comparative_list

    torch.save(token_dict,"./dataset/dict_sets/token_dict.pth")
    torch.save(deny_dict,"./dataset/dict_sets/deny_dict.pth")
    torch.save(tri_entity_dict,"./dataset/dict_sets/tri_entity_dict.pth")
    torch.save(number_dict,"./dataset/dict_sets/number_dict.pth")
    torch.save(comparative_dict,"./dataset/dict_sets/comparative_dict.pth")
    return token_dict, deny_dict, tri_entity_dict, number_dict, comparative_dict

# 计算否认分数
def deny_score(event_sentence_id,new_sentence_id,cfg):
    # 否定词个数差异
    deny_dict = torch.load(cfg.dict_sets+'deny_dict.pth')
    score_1 = abs(len(deny_dict[event_sentence_id]) - len(deny_dict[new_sentence_id]))

    # 反义词差异
    antonyms_words_dict = read_words(cfg.words_path+'antonyms_words.txt',col=2)
    token_dict = torch.load(cfg.dict_sets+'token_dict.pth')
    score_2 = 0
    for word in token_dict[new_sentence_id]:
        if word in antonyms_words_dict.keys():
            antonyms_words = antonyms_words_dict[word]
            if antonyms_words in token_dict[event_sentence_id]:
                score_2 += 1
            else:
                continue
        else:
            continue
    # 总否认分数
    deny_total_score = (score_1 + score_2) / len(token_dict[new_sentence_id])
    return deny_total_score

# 计算混淆分数
def confusion_score(event_sentence_id,new_sentence_id,cfg):
    # 定义一个小函数，用于tri_entity_dict转为三元组list
    def dict_to_tri(tri_dict):
        tri_list = []
        words = tri_dict['word']
        heads = tri_dict['head']
        deprels = tri_dict['deprel']
        for i in range(len(words)):
            if heads[i] == 0:
                tri = (words[i], deprels[i], 'ROOT')
            else:
                tri = (words[i], deprels[i], words[heads[i]-1])
            tri_list.append(tri)
        return tri_list

    # 定义一个小函数，用于对比两个三元组中重叠部分的个数
    def cmp_tri(triple1, triple2):
        count = 0
        if triple1[0] == triple2[0] or triple1[0] == triple2[2]:
            count += 1
        if triple1[1] == triple2[1]:
            count += 1
        if triple1[2] == triple2[0] or triple1[2] == triple2[2]:
            count += 1
        return count

    # 构建出事件和新闻的 实体-实体关系-实体 三元组list
    tri_entity_dict = torch.load(cfg.dict_sets+'tri_entity_dict.pth')
    event_tri_list = dict_to_tri(tri_entity_dict[event_sentence_id])
    new_tri_list = dict_to_tri(tri_entity_dict[new_sentence_id])

    m_new_tri = []     # 统计当前新闻与事件，至少有2个重叠部分的三元组
    m_event_tri = []
    m = []             # 重叠部分个数

    # 计算score1
    for new_tri in new_tri_list:
        for event_tri in event_tri_list:
            cmp = cmp_tri(new_tri, event_tri)
            if cmp < 2:
                continue
            else:
                m_new_tri.append(new_tri)
                m_event_tri.append(event_tri)
                m.append(cmp)
    score_1 = 1 - (len(m)/len(new_tri_list))

    # 计算score2
    score_2 = m.count(2) / len(m)

    # 计算总混淆分数
    confusion_total_score = score_1 * score_2
    return confusion_total_score

# 计算夸大事实分数
def exaggera_score(event_sentence_id,new_sentence_id,cfg):
    # 比较级分数差异
    comparative_dict = torch.load(cfg.dict_sets+'comparative_dict.pth')
    comparative_words_dict = read_words(cfg.words_path+'comparative_words.txt',col=2)
    event_comparative = 0.0
    for com in comparative_dict[event_sentence_id]:
        event_comparative += float(comparative_words_dict[com])
    new_comparative = 0.0
    for com in comparative_dict[new_sentence_id]:
        new_comparative += float(comparative_words_dict[com])

    # 计算总夸大事实分数
    exaggera_total_score = abs(event_comparative - new_comparative)
    return exaggera_total_score

# 计算rouge分数，即无中生有
def rouge_score(event_sentence,new_sentence,rouge):
    rouge_score_ = rouge.get_scores(' '.join(list(new_sentence)),' '.join(list(event_sentence)))
    rouge_score_ = rouge_score_[0]["rouge-l"]['r']
    return rouge_score_

# 归一化
def MaxMinNormalization(x):
    small_e = 1e-6
    Max_x = max(x)
    Min_x = min(x)
    x_result = []
    for x_temp in x:
        x_result.append((x_temp - Min_x) / (Max_x - Min_x + small_e))
    return x_result
# 功能函数，用于部分归一化0-0.8
def MaxMinNormalization_8(score_list, flag_list):
    if flag_list.count(True) == 0:
        return score_list
    else:
        True_index = [i for i,v in enumerate(flag_list) if v==True]   # 找出所有True的index保存
        subset_score_list = [score_list[index] for index in True_index]
        subset_score_list = MaxMinNormalization(subset_score_list)
        subset_score_list = [x * 0.8 for x in subset_score_list]
        for i in range(len(True_index)):
            score_list[True_index[i]] = subset_score_list[i]
        return score_list

# 先求平均，再使用四分之一圆函数（反比例函数）
def yuan(score1, score2, score3, score4):
    mean = (score1 + score2 + score3 + score4) / 4
    # result = math.exp(-mean*mean/2)
    result = 1 - mean*mean
    return result

if __name__ == '__main__':
    cfg = Config()
    rouge = Rouge()
    # Event_sentences = event_sentence_main(cfg)
    Event_sentences = ['据@平安万州，10月28日10时08分，一辆公交客车与一辆小轿车在重庆万州区长江二桥相撞后，公交车坠入江中。\r', '万州警方称，10月28日10时08分，一辆公交客车与一辆小轿车在重庆万州区长江二桥相撞后，公交客车坠入江中。\r', '根据警方最新通报，该司机并非逆行，而是公交车突然越过中心实线，撞向该正常行驶的小轿车！\r', '最新消息，重庆官方通报：\n重庆万州区坠江事故经初步现场调查，系公交客车在行驶中突然越过中心实线，撞击对向正常行驶的小轿车后冲上路沿，撞断护栏，坠入江中。\r', '【坠江事故女司机没逆行！重庆公交车坠江因突然越线，车上共有驾乘人员10余人】']

    New = """"重庆万州一大巴车坠江，或系一女司机驾驶私家车逆行导致。
据重庆青年报消息，28日上午，重庆市万州区长江二桥发生重大交通事故，-辆大巴车被撞后冲破护栏坠入长江，疑有重大伤亡。目前,政府正在组织救援。据传，事故系一女司机驾驶的红色私家车桥上逆行所致。"""
    New_sentences = cut_sent(New)

    Sentences = Event_sentences + New_sentences
    Sentences_ids = ["event_{}".format(i) for i in range(len(Event_sentences))]
    Sentences_ids = Sentences_ids + ["new_sen_{}".format(i) for i in range(len(New_sentences))]
    token_dict, deny_dict, triple_dict, num_dict, com_dict = data_prepare(Sentences,Sentences_ids,cfg.words_path)

    # for event_sentence in Event_sentences:
    event_sentence = '根据警方最新通报，该司机并非逆行，而是公交车突然越过中心实线，撞向该正常行驶的小轿车！\r'
    curr_rouge_score_list = []         # 为 1-rouge,因为希望高权重句子有高rouge值，因此1-rouge需要小
    curr_deny_score_list = []          
    curr_confusion_score_list = []
    curr_exaggera_score_list = []
    curr_flag_list = []                # 标记这个值不是默认值，是计算出来的值，方便后面部分归一化
    temp_rouge_list = []
    for new_sentence in New_sentences:
        curr_rouge = rouge_score(event_sentence,new_sentence,rouge) / len(new_sentence)    # 首先计算rouge值,除以句子长度是为了避免靠句子长取得高rouge值
        print(curr_rouge)
        temp_rouge_list.append(curr_rouge)
    temp = sorted(temp_rouge_list)
    number = int(len(temp)*cfg.τ)
    rouge_τ = (temp[-number-1] + temp[-number]) / 2
    for curr_rouge in temp_rouge_list:
        if curr_rouge < rouge_τ:                                          # 如果rouge一点都不匹配，默认其他三种差异都1分，以减少计算
            curr_rouge_score_list.append(1 - curr_rouge)
            curr_deny_score_list.append(1)
            curr_confusion_score_list.append(1)
            curr_exaggera_score_list.append(1)
            curr_flag_list.append(False)
        else:
            event_id, new_id = Sentences_ids[Sentences.index(event_sentence)], Sentences_ids[Sentences.index(new_sentence)]
            curr_rouge_score_list.append(1 - curr_rouge)
            curr_deny_score_list.append(deny_score(event_id, new_id, cfg))
            curr_confusion_score_list.append(confusion_score(event_id, new_id, cfg))
            curr_exaggera_score_list.append(exaggera_score(event_id, new_id, cfg))
            curr_flag_list.append(True)
    curr_rouge_score_list = MaxMinNormalization_8(curr_rouge_score_list, curr_flag_list)
    curr_deny_score_list = MaxMinNormalization_8(curr_deny_score_list, curr_flag_list)
    curr_exaggera_score_list = MaxMinNormalization_8(curr_exaggera_score_list, curr_flag_list)
    curr_confusion_score_list = MaxMinNormalization_8(curr_confusion_score_list, curr_flag_list)
    
    curr_weight_score_list = [yuan(curr_rouge_score_list[i],curr_deny_score_list[i], \
                            curr_exaggera_score_list[i],curr_confusion_score_list[i]) for i in range(len(curr_rouge_score_list))]
    curr_weight_score_list = F.softmax(torch.tensor(curr_weight_score_list)).tolist()
    for i in range(len(New_sentences)):
        print(New_sentences[i])
        print('curr_weight_score_list:',curr_weight_score_list[i])
        print('curr_rouge_score_list:',curr_rouge_score_list[i])
        print('curr_deny_score_list:',curr_deny_score_list[i])
        print('curr_exaggera_score_list:',curr_exaggera_score_list[i])
        print('curr_confusion_score_list:',curr_confusion_score_list[i])
        print('curr_flag_list:',curr_flag_list[i])
        print('\n')

