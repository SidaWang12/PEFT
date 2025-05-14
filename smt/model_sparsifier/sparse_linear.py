import torch

from torch import nn


class BlockSparseLinear(torch.nn.Module):
    """A block-sparse implementation of a linear layer.
    
    Args:
        weight: The full weight matrix of the linear layer
        bias: Optional bias vector
        selected_blocks_list: List of (row, col) block indices to keep active
        block_dimension: Size of each block (default: 256)
    """
    def __init__(self,
                 weight,
                 bias=None,
                 selected_blocks_list=[],
                 block_dimension=256):
        super(BlockSparseLinear, self).__init__()
        # Store original parameters
        self.weight = weight
        self.weight.requires_grad = False
        self.bias = bias
        self.selected_blocks_list = selected_blocks_list
        self.block_dimension = block_dimension

        # Initialize selected weight blocks
        self._init_selected_weights()

        # The custom sparse linear operation
        self.block_sparse_linear_function = BlockSparseLinearFunction.apply

    def _init_selected_weights(self):
        """Initialize the trainable selected weight blocks."""
        num_blocks = len(self.selected_blocks_list)
        output_dim = num_blocks * self.block_dimension
        input_dim = self.block_dimension

        self.selected_weight = torch.empty(output_dim,
                                           input_dim,
                                           dtype=self.weight.data.dtype,
                                           device=self.weight.data.device)

        # Copy relevant blocks from original weights
        for i, (row_idx, col_idx) in enumerate(self.selected_blocks_list):
            row_start = row_idx * self.block_dimension
            row_end = row_start + self.block_dimension
            col_start = col_idx * self.block_dimension
            col_end = col_start + self.block_dimension

            block = self.weight.data[row_start:row_end, col_start:col_end]
            self.selected_weight[i * self.block_dimension:(i + 1) *
                                 self.block_dimension, :] = block

        self.selected_weight.requires_grad = True
        self.selected_weight = nn.Parameter(self.selected_weight)

    def _update_original_weights(self):
        """Update the original weight matrix with trained block values."""
        for i, (row_idx, col_idx) in enumerate(self.selected_blocks_list):
            row_start = row_idx * self.block_dimension
            row_end = row_start + self.block_dimension
            col_start = col_idx * self.block_dimension
            col_end = col_start + self.block_dimension

            # Get the trained block
            trained_block = self.selected_weight[i *
                                                 self.block_dimension:(i + 1) *
                                                 self.block_dimension, :]

            # Update the corresponding block in original weights
            self.weight.data[row_start:row_end,
                             col_start:col_end] = trained_block

    def forward(self, input):
        # Update original weight matrix with trained blocks
        self._update_original_weights()

        # Perform the block-sparse linear operation
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

        output = torch.matmul(input, weight.t())

        # memory free
        del weight
        del input_list
        del selected_blocks_list

        return output

    @staticmethod
    def backward(ctx, grad_output):
        weight, = ctx.saved_tensors
        input_list = ctx.input_list
        block_dimension = ctx.block_dimension
        selected_blocks_list = ctx.selected_blocks_list

        # Pytorch use C++ engine to check whether gradient has matched dimenstion or not
        grad_weight = torch.empty(len(input_list) * block_dimension,
                                  block_dimension,
                                  dtype=grad_output.dtype,
                                  device=grad_output.device)
        for i in range(len(input_list)):
            index = selected_blocks_list[i]

            grad_weight[i * block_dimension:i * block_dimension +
                        block_dimension, :] = torch.sum(torch.matmul(
                            grad_output.permute(
                                0, 2,
                                1)[:, index[0] *
                                   block_dimension:index[0] * block_dimension +
                                   block_dimension, :], input_list[i]),
                                                        dim=0)

        grad_input = torch.matmul(grad_output, weight)

        return grad_input, grad_weight, None, None, None
