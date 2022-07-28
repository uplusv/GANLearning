这是一个用于深度学习训练的框架，基于Pytorch，参考自basicsr和pSp的代码。

## 结构 
data: 存放和data有关的代码，以_dataset.py为结尾的文件视为Pytorch的dataset，会自动注册。  
  
models: 存放和model有关的代码，包括model的结构和训练方式，以_model.py为结尾的文件会自动注册。每种模型应该由两部分组成，结构和训练，结构继承自pytorch的nn.Module；训练部分应继承basic_model.py的BaseModel。暂定可以分成两个文件，以MLP为例，models文件下应该有MLP_arch.py（结构）和MLP_model.py（训练）。  
  
criterions: 存放loss/metric的文件夹，如果需求比较大可以分为两个。  
  
scripts: 存放运行入口。训练可以直接从train.py进入。  
  
trainig: 存放训练用代码。coach.py是训练的主逻辑，ranger.py是一个优化器。  
  
utils: 其他工具。  
  
exps: 保存实验结果的文件夹。config.yaml中保存所有实验需要的超参数。
  
## 训练逻辑
scripts/train.py -> training/coach.py -> models/xxx_model.py  
  
train.py: 负责处理option，主要是opt参数——config.yaml的地址，为实验创建文件夹。  
  
coach.py: 完整的训练过程，负责初始化model，dataset，logger，跟踪log整个训练过程。代码不需要过多改动。  
  
xxx_model.py: 每个模型独有的训练类，继承basic_model.py中的BaseModel类。optimizer和scheduler由这个类维护。feed_data(batch_data)将dataloader输出的数据转换为模型需要的形式；optimize_params()执行单步训练。