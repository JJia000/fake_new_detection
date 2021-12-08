class Config():
    def __init__(self):
        self.data_path = "./dataset/chongqing_bus_event.csv"
        self.bert_model = './chinese-bert-wwm-ext'
        self.bert_pre_pra = './model/my_pre_bert.pth'
        self.words_path = './ChineseSemanticKB/'
        self.dict_sets = './dataset/dict_sets/'

        # 第一阶段参数
        self.alpha = 0.2              # rouge权重
        self.beta = 1 - self.alpha    # 语义余弦相似度权重
        self.k = 5                    # 取出总体事件句子的数量

        # 第二阶段参数
        self.τ = 0.5                  # rouge差异分数阈值比例，低于该值时默认差异分数为满分，高于该值时计算其他三项差异分数