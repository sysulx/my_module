'''
重构transformers代码，使其支持一般的模型构建与训练，不仅仅是预训练模型
自定义配置文件

'''
import logging

from transformers import PretrainedConfig

logger = logging.getLogger(__name__)

class Config(PretrainedConfig):
    # 继承PretrainedConfig是非了兼容其他transformers类 
    # 注意CONFIG_NAME='config.json'，所以自定义config，最好也是这个文件名
    # model_type是类属性，已经从PretrainedConfig继承
    # model_type = ''
    def __init__(self, **kwargs):
        # 父类的参数是默认的
        super(Config, self).__init__(**kwargs)

        # 因为在保存config的时候，为了方便只保存与默认值不同的参数，这个默认值来源于PretrainedConfig的默认值，
        # 所以如果用户传入了一个参数是PretrainedConfig类里面的，并且和默认值一样，那么模型就不会显示的保存，
        # 这就很不友好，因此这里有一个参数user_config:Dict专门用来保存用户传入的参数，不管是否与默认参数一致
        # 如果保存到硬盘再读取，这么做就不可行了！！！ 所以功能取消
        # self.user_config = {} # CANCEL

    # 重要方法在这里再列举一下
    # 注意类方法可以直接继承，并且返回的是子类的对象而不是父类的对象
    # 父类类方法中的cls被子类继承后就是代表子类，而不是父类
    # 注意下面的 pretrained_model_name_or_path自定义时只能传递本地路径文件夹，没有远程下载
    '''
    def save_pretrained(self, save_directory: str):
        # 以json格式将本对象的属性保存起来
        pass

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        # 从pretrained_model_name_or_path(目录)中读取config.json返回对象
        pass

    @classmethod
    def get_config_dict(cls, pretrained_model_name_or_path: str, **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        # 从目录中找到配置的文件config.json读取并返回，**kwargs是一方面是辅助寻找.json文件的参数，
        # 一方面是额外配置参数
        pass

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any], **kwargs) -> "PretrainedConfig":
        # 从dict对象中创建config，**kwargs表示可以加入创建额外的参数配置
        pass

    @classmethod
    def from_json_file(cls, json_file: str) -> "PretrainedConfig":
        # 从json文件的路径中创建config，注意这个文件名不一定是CONFIG_NAME
        pass

    '''
    
    # 其他方法：
    '''
    @classmethod
    def _dict_from_json_file(cls, json_file: str):
    
    def __eq__(self, other):
    
    def __repr__(self):
    
    def to_diff_dict(self) -> Dict[str, Any]:
    
    def to_dict(self) -> Dict[str, Any]:
    
    def to_json_string(self, use_diff: bool = True) -> str:
    
    def update(self, config_dict: Dict[str, Any]):
    
    '''
