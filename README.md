# fake_new_detection

1. bert_example.py  
   对哈工大bert预训练模型在自己的数据上进行微调，并保存模型  
   
2. config.py  
   参数配置文件，所有可调参数都保存在其中  

3. event_sentence.py  
   模块1的实现，从一批真实新闻中提取指定数量的事件句子汇聚成事件摘要 
   
4. weight_score.py  
   模块2的实现，给定事件句子和一则新闻，输出这则新闻每个句子的权重  
   
5. detection_model.py  
   模块3的实现，给定事件摘要嵌入和待检测新闻嵌入和待检测新闻权重，经过网络结构输出为预测真假标签  
   
6. train_fake_new.py  
   训练模型的主入口以及训练过程  
   
7. test_fake_new.py  
   测试模型的主入口，适用于已经训练过模型已保存模型的情况
