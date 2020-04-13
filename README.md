#XSS注入检测
## 1. 项目结构
```text
data 存放所有的数据  
cache 
| 中间数据
model  
| 存放训练后的模型  
|——preprocessing 存放预处理模块的结构  
src  
| main 训练和测试  
|——preprocessing 预处理模块  
|——|—— parser.py 正则化清洗数据  
|——|—— reader.py 读取数据，并划分数据

|——|—— load_data.py 生成特征空间，并将载荷数据转化成列表

|——|—— vec.py 攻击载荷生成的字符列表生成word2vec
```



# 2. 结果
```text
          		precision    recall  f1-score   support

     	 0.0       0.99      1.00      1.00     40026
     	 1.0       1.00      0.97      0.98      8128

	accuracy                           0.99     48154
   macro avg       1.00      0.98      0.99     48154
weighted avg       0.99      0.99      0.99     48154
```
