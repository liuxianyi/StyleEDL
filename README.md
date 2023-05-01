
## 运行
```bash
python main.py --tag 
```
+ `--tag` 指定保存目录的前缀，默认为cache



## 添加新的模型

1. 在`network`文件夹下新建新的文件
1. 在`network`下`__init__`中的`models`加入新的网络模型
1. 在`configs`文件夹中新建对应的配置文件


## 添加新的trainer

1. 在`trainer`文件夹中新建文件，继承`basetrainer`
1. 在`trainer`下`__init__`中的`trainers`加入新的trainer
1. 运行时，由配置文件中的`trainer`指定
