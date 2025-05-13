import torch

from torch import nn


class BlockSparseLinear(torch.nn.Module):
    # a simple implementation of matrix sparsity
    # for now only support Linear Layer
    def __init__(self,
                 weight,
                 bias=None,
                 selected_blocks_list=[],
                 block_dimension=256):
        super(BlockSparseLinear, self).__init__()
        self.weight = weight
        self.weight.requires_grad = False
        self.bias = bias
        self.selected_blocks_list = selected_blocks_list

        self.selected_weight = torch.empty(len(selected_blocks_list) *
                                           block_dimension,
                                           block_dimension,
                                           dtype=self.weight.data.dtype,
                                           device=self.weight.data.device)
        self.block_dimension = block_dimension

        for i in range(len(selected_blocks_list)):
            index = selected_blocks_list[i]
            self.selected_weight[
                i * block_dimension:i * block_dimension +
                block_dimension, :] = self.weight.data[
                    index[0] * block_dimension:index[0] * block_dimension +
                    block_dimension,
                    index[1] * block_dimension:index[1] * block_dimension +
                    block_dimension]
        self.selected_weight.requires_grad = True
        self.selected_weight = nn.Parameter(self.selected_weight)

        self.block_sparse_linear_function = BlockSparseLinearFunction.apply

    def forward(self, input):
        for i in range(len(self.selected_blocks_list)):
            index = self.selected_blocks_list[i]
            # self.selected_weight[i * Block_dimension: i * Block_dimension + Block_dimension, :] = self.weight.data[index[0] * Block_dimension: index[0] * Block_dimension + Block_dimension, index[1] * Block_dimension: index[1] * Block_dimension + Block_dimension]
            self.weight.data[
                index[0] *
                self.block_dimension:index[0] * self.block_dimension +
                self.block_dimension, index[1] *
                self.block_dimension:index[1] * self.block_dimension +
                self.block_dimension] = self.selected_weight[
                    i * self.block_dimension:i * self.block_dimension +
                    self.block_dimension, :]

        output = self.block_sparse_linear_function(input, self.selected_weight,
                                              self.selected_blocks_list,
                                              self.weight,
                                              self.block_dimension)
        return output


class BlockSparseLinearFunction(torch.autograd.Function):
    # only support batch size D=3 now, for batch size = 1, need to add mm. operation.
    @staticmethod
    def forward(ctx, input, selected_weight, selected_blocks_list, weight,
                block_dimension):
        input_list = []
        for index in selected_blocks_list:
            input_list.append(
                input[:, :,
                      index[1] * block_dimension:index[1] * block_dimension +
                      block_dimension])
        # save for backward may only support tensor, use others to save!
        ctx.input_list = input_list
        ctx.selected_blocks_list = selected_blocks_list
        ctx.block_dimension = block_dimension

        ctx.save_for_backward(weight)

        # output = input.mm(weight.t())
        # print("input size:",input.size())
        # print("weight size:",weight.data.size())
        output = torch.matmul(input, weight.t())

        # memory free
        del weight
        del input_list
        del matrix_index_list

        return output

    @staticmethod
    def backward(ctx, grad_output):
        weight, = ctx.saved_tensors
        input_list = ctx.selected_blocks_list
        block_dimension = ctx.block_dimension
        matrix_index_list = ctx.list2

        # Pytorch use C++ engine to check whether gradient has matched dimenstion or not
        grad_weight = torch.empty(len(input_list) * block_dimension,
                                  block_dimension,
                                  dtype=grad_output.dtype,
                                  device=grad_output.device)
        for i in range(len(input_list)):
            index = matrix_index_list[i]

            # print("index:", index)
            # print("grad_output_dim:", grad_output.size())
            # tmp = grad_output.permute(0, 2, 1)[:, index[0] * Block_dimension: index[0] * Block_dimension + Block_dimension, :]
            # print("tmp size", tmp.size())
            # print("input list[i]", input_list[i].size())
            # tmp1 = torch.matmul(tmp, input_list[i])
            # grad_weight[i * Block_dimension: i * Block_dimension + Block_dimension, :] = torch.sum(tmp1, dim=0)

            grad_weight[i * block_dimension:i * block_dimension +
                        block_dimension, :] = torch.sum(torch.matmul(
                            grad_output.permute(
                                0, 2,
                                1)[:, index[0] *
                                   block_dimension:index[0] * block_dimension +
                                   block_dimension, :], input_list[i]),
                                                        dim=0)

        grad_input = torch.matmul(grad_output, weight)

        # memory free
        del weight
        del input_list
        del matrix_index_list

        return grad_input, grad_weight, None, None
