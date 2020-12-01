"""
A wrapper that unrolls the some dimensions of a tensor
into the first (batch) dimension, applies some other `Module`,
and then rolls the time dimension back up.
"""

from typing import List
import torch

class TimeDistributed(torch.nn.Module):
    """
    将前面任意k个dimension合并到一个dimension，然后输入到module里进行运算，将运算结果的 \n
    第一维再返回分解出k个dimension（包括自己是k+1个dimension），剩下的dimension不变。\n
    注意，可以是部分inputs进行TimeDistributed，使用pass_through参数；
    """

    def __init__(self, module, squash_dim_num=2, input_dim_num=None):
        super().__init__()
        self._module = module
        if input_dim_num is None:
            self.squash_dim_num = squash_dim_num

    def forward(self, *inputs, pass_through: List[str] = None, **kwargs):

        pass_through = pass_through or []

        reshaped_inputs = [self._reshape_tensor(input_tensor) for input_tensor in inputs]

        # Need some input to then get the squashed dims.
        some_input = None
        if inputs:
            some_input = inputs[-1]

        reshaped_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor) and key not in pass_through:
                if some_input is None:
                    some_input = value

                value = self._reshape_tensor(value)

            reshaped_kwargs[key] = value

        reshaped_outputs = self._module(*reshaped_inputs, **reshaped_kwargs)

        if some_input is None:
            raise RuntimeError("No input tensor to time-distribute")

        # Now get the output back into the right shape.
        # (dim0, dim1, ..., dimk, **output_size)
        new_size = some_input.size()[:self.squash_dim_num] + reshaped_outputs.size()[1:]
        outputs = reshaped_outputs.contiguous().view(new_size)

        return outputs

    def _reshape_tensor(self, input_tensor):
        input_size = input_tensor.size()
        if len(input_size) <= self.squash_dim_num:
            raise RuntimeError(f"No dimension to distribute: {input_size}")
        # Squash top k dimenstions into a single axis; result has shape
        # (reduce(*, (dim0,dim1,...dimk)), **input_size).
        squashed_shape = [-1] + list(input_size[self.squash_dim_num:])
        return input_tensor.contiguous().view(*squashed_shape)
