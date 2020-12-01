
import logging
from my_module.config import Config
from transformers import PreTrainedModel



class Model(PreTrainedModel):

    # 自定义模型：继承BaseModel就可以获的transformers库的一系列支持
    # 1. 模型初始化，所有的配置参数都需要从config中传入，所以在创建config时就先把参数配置准备好
    # 2. 需要实现forward方法，forward方法的所有参数名应该作为输入的dataset的每一个实例:Dict[str, Any]的key对应
    # 或与传入dataloader的每一个batch:Dict[str, Any]）的key对应，
    # 3. forward输入不仅仅是train_x, train_y也要输入（如果需要计算损失进行反向传播的话），dataset的每一个实例都是x和y的对应
    # 注意原始的trainer类不支持传入dataloader，只能通过传入dataset和collate_function进行dataloader的构建
    # 4. forward方法的返回值:Tuple必须是以下情况：
    #   (1) (loss, logits, ...) # 这个是可以计算loss的情况，...表示还可以输出其他的，比如attention的取值等等，但是训练时不会用到
    #   (2) (logits, ...) # 这个是不能计算loss的情况，...表示还可以输出其他的，比如attention的取值等等，但是训练时不会用到
    # 注意如果只返回logits，也需要加个逗号,表示返回的是元组Tuple

    '''
    # 下面3个成员变量是继承父类的，但是from_pretrain方法需要知道你这个模型的配置类
    config_class = None
    base_model_prefix = ""
    authorized_missing_keys = None
    '''
    config_class = Config
    base_model_prefix = 'customize'
    
    def __init__(self, config):
        super(Model, self).__init__(config)
    
    '''
    # 重要方法

    def save_pretrained(self, save_directory):

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
    '''
