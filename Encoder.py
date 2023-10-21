import torch.nn as nn
import math 


def scaled_dot_product(q, k, v, mask=None):
    # Calculate scaled dot product attention
    d_k = q.size()[-1]
    scaled = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    
    if mask is not None:
        scaled = scaled + mask
    
    # Apply softmax to obtain attention scores
    attention = torch.softmax(scaled, dim=-1)
    
    # Compute weighted sum using attention scores
    values = torch.matmul(attention, v)
    
    return attention, values

class Multihead_Attention(nn.Module):
    def __init__(self, input_dim, d_model, num_heads):
        super(Multihead_Attention, self).__init()
        
        # Initialize class attributes
        self.input_dim = input_dim
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads  
        
        # Linear layers for Q, K, and V projections
        self.qkv_layer = nn.Linear(input_dim, 3 * d_model)
        
        # Linear layer for output
        self.linear_layer = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        batch_size, sequence_length, input_dim = x.size()

        # Apply the Q, K, V projections
        qkv = self.qkv_layer(x)

        # Reshape for multi-head attention
        qkv = qkv.view(batch_size, sequence_length, self.num_heads, 3 * self.head_dim)

        # Permute dimensions for efficient computation
        qkv = qkv.permute(0, 2, 1, 3)  # Batch x num_heads x sequence_length x (3 * head_dim)

        # Split Q, K, V into separate tensors
        q, k, v = qkv.chunk(3, dim=-1)  # Each will have shape: batch x num_heads x sequence_length x head_dim

        # Apply scaled dot-product attention
        values, attention = scaled_dot_product(q, k, v, mask)

        # Reshape the values for concatenation
        values = values.view(batch_size, sequence_length, self.num_heads * self.head_dim)

        # Apply the final linear transformation
        out = self.linear_layer(values)

        return out

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_sequence_length):
        super().__init__()
        self.max_sequence_length = max_sequence_length  # Maximum sequence length (e.g., 100)
        self.d_model = d_model  # Dimension of the model (e.g., 512)

    def forward(self,x):
        # Calculate positional encodings for even indices
        even_i = torch.arange(0, self.d_model, 2).float()  # Create a tensor with even values [0, 2, 4, ...]

        denominator = torch.pow(10000, even_i / self.d_model)  # Calculate the denominator using even_i

        # Generate positional encodings for even indices using the sine function
        position = torch.arange(self.max_sequence_length).reshape(self.max_sequence_length, 1)
        even_PE = torch.sin(position / denominator)

        # Generate positional encodings for odd indices using the cosine function
        odd_PE = torch.cos(position / denominator)

        # Stack even and odd positional encodings along the last dimension
        stacked = torch.stack([even_PE, odd_PE], dim=2)

        # Flatten the stacked tensor to create the final positional encoding tensor
        PE = torch.flatten(stacked, start_dim=1, end_dim=2)

        # Add positional encodings to the input
        output = x + PE


        return output



class LayerNormalization(nn.Module):
    def __init__(self, parameters_shape, eps=1e-5):
        super(LayerNormalization, self).__init__()
        self.parameters_shape = parameters_shape
        self.eps = eps

        # Learnable scaling and shifting parameters
        self.gamma = nn.Parameter(torch.ones(parameters_shape))
        self.beta = nn.Parameter(torch.zeros(parameters_shape))

    def forward(self, inputs):
        # Calculate the mean and variance along the specified dimensions
        dims = [-(i + 1) for i in range(len(self.parameters_shape))]
        mean = inputs.mean(dim=dims, keepdim=True)
        var = ((inputs - mean) ** 2).mean(dim=dims, keepdim=True)
        
        # Calculate the standard deviation with epsilon for stability
        std = (var + self.eps).sqrt()
        
        # Normalize the input
        y = (inputs - mean) / std
        
        # Apply scaling and shifting
        out = self.gamma * y + self.beta
        
        return out


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()

        self.fc1 = nn.Linear(d_model, d_ff)  # First fully connected layer
        self.fc2 = nn.Linear(d_ff, d_model)  # Second fully connected layer
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
       
        # Apply the first linear layer followed by the activation function
        intermediate = self.relu(self.fc1(x))
        
        # Apply dropout
        intermediate = self.dropout(intermediate)
        
        # Apply the second linear layer
        out = self.fc2(intermediate)
        
        return out