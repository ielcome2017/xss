#XSS注入检测
## 1. 项目结构
data 存放所有的数据  
model  
| 存放训练后的模型  
|——preprocessing 存放预处理模块的结构  
src  
| main 训练和测试  
|——preprocessing 预处理模块  
|——|—— feature.py 正则化清洗数据  
|——|—— feature.py 提取特征后，向量化文本  
|——|—— feature.py 提取数据集，标准化数据集  
# 2. 数据处理  
