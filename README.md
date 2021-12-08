# fake_new_detection

1. bert_example.py  
   对哈工大bert预训练模型在自己的数据上进行微调，并保存模型  
   
2. config.py  
   参数配置文件，所有可调参数都保存在其中  

3. event_sentence.py  
   模块1的实现，从一批真实新闻中提取指定数量的事件句子  
   
4. weight_score.py  
   模块2的实现，给定事件句子和一则新闻，输出这则新闻每个句子的权重
