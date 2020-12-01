import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

def _pad_sequence(sequences, batch_first=False, padding_length='max_length', padding_value=0.0):
    # 官方pad_sequence的修改版，支持任意长度的pad
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    max_len = max([s.size(0) for s in sequences])
    if padding_length == 'max_length':
        if batch_first:
            out_dims = (len(sequences), max_len) + trailing_dims
        else:
            out_dims = (max_len, len(sequences)) + trailing_dims
        out_tensor = sequences[0].new_full(out_dims, padding_value)
        for i, tensor in enumerate(sequences):
            length = tensor.size(0)
            # use index notation to prevent duplicate references to the tensor
            if batch_first:
                out_tensor[i, :length, ...] = tensor
            else:
                out_tensor[:length, i, ...] = tensor
    else:
        max_len = padding_length
        if batch_first:
            out_dims = (len(sequences), max_len) + trailing_dims
        else:
            out_dims = (max_len, len(sequences)) + trailing_dims
        out_tensor = sequences[0].new_full(out_dims, padding_value)
        for i, tensor in enumerate(sequences):
            length = min(tensor.size(0), max_len)
            # use index notation to prevent duplicate references to the tensor
            if batch_first:
                out_tensor[i, :length, ...] = tensor[:length]
            else:
                out_tensor[:length, i, ...] = tensor[:length]

    return out_tensor


def pad_seq(inputs, batch_first=True, padding_length='max_length', padding_value=0.0, validation_check=True):
    """
    输入一批不定长度的序列，pad到最大长度，并返回长度信息lengths和padding的mask，具体参考`torch.nn.utils.rnn.pad_sequence`
    @param inputs: List of data，每个data维度大小不一，不能直接构建tensor
    @param batch_first: 第一维是表示seq_length还是表示batch
    @param padding_length: 表示pad的长度是多少，默认是从这个batch里面获得的最大长度
    @param padding_value: 表示pad的值是多少，inputs中最好不要在padding_value出现，否则mask不正确!!!
    @param validation_check: 检查padding_value是否在inputs中出现过
    @returns padded_inputs: torch.tensor
    @returns lengths: torch.LongTensor, 表示seq的长度
    @returns mask: torch.FloatTensor, 0表示padding位置
    """
    if not isinstance(inputs[0], torch.Tensor):
        inputs = [torch.tensor(seq) for seq in inputs]
    if validation_check: # 如果自己确保不会出现，可以设置为False，减少计算
        assert all(torch.sum(input_tensor == padding_value).item() == 0 for input_tensor in inputs)
    lengths = torch.tensor([len(seq) for seq in inputs])
    padded_inputs = _pad_sequence(inputs, batch_first=batch_first, padding_length=padding_length, padding_value=padding_value)
    mask = (padded_inputs != padding_value).float()
    return padded_inputs, lengths, mask


def get_seq_last_output(seq_output, input_mask):
    """
    获取pad output（rnn 输入 pad input得到的）的最后一个step的output \n
    @param rnn_output (torch.tensor): batch_size*seq_length*output_dim \n
    @param input_mask (torch.tensor): batch_size*seq_length , 取值为0/1, 0代表是pad的位置 \n
    @returns last_output (torch.tensor): batch_size*output_dim \n
    """
    batch_size, seq_length = input_mask.size()
    last_indices = torch.sum(input_mask, dim=1, keepdims=True)-1

    # 防止存在整个seq都是pad的，如果存在这种情况，就返回第一个step的输出，即如下修改index为0
    last_indices = last_indices.masked_fill(labels==-1, 0) 

    index_position = torch.zeros((batch_size, seq_length), dtype=torch.long, device=input_mask.device)
    index_position = index_position.scatter_(1, last_indices, 1).bool()

    last_output = seq_output[index_position] # batch_size, output_dim

    return last_output

# 下面这个函数LSTM/GRU/RNN均已经测试
class DynamicRNN(nn.Module):
    """
    处理动态长度的RNN输入，\n
    返回正确的最后一个step的output \n
    默认`enforce_sorted`改为`False`，即输入的batch不需要手动排序了, \n
    这个只在高版本的torch中支持，低版本没有这个参数，必须手动排序，可以参考高版本torch源码的实现 \n
    不排序的好处就是自己不用写排序代码（实际由`pack_padded_sequence`完成排序），坏处就是不支持ONNX迁移。\n
    Note: \n 
        - pytorch高版本可用 \n 
        - `batch_first==True`!!!  \n
    """
    def __init__(self, rnn_module):
        super().__init__()
        self._rnn_module = rnn_module
    
    def forward(self, inputs, lengths=None, enforce_sorted=False, **kwargs):
        """
        @param `inputs` (torch.tensor): (batch_size, seq_length, output_dim,) \n
        @param `lengths` (torch.tensor or List): (batch_size,)  \n
        @param `enforce_sorted` (bool): 如果自己排好序，就可以设置为True，否则默认就好 \n
        """
        if lengths is None:
            batch_size, seq_len, _ = inputs.size()
            lengths = torch.full((batch_size,), fill_value = seq_len, dtype=torch.long, device=inputs.device)
        packed_inputs = pack_padded_sequence(inputs, lengths, batch_first=True, enforce_sorted=enforce_sorted)
        packed_outputs, last_hidden = self._rnn_module(packed_inputs, **kwargs) # hidden已经考虑变长的因素了
        padded_outputs, lengths_tensor = pad_packed_sequence(packed_outputs, batch_first=True)
        return padded_outputs, last_hidden # 确保和正常使用RNN无差异

