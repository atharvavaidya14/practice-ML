"""
This script trains a neural network that solves the following task:
Given an input sequence XY[0-5]+ where X and Y are two given digits,
the task is to count the number of occurrences of X and Y in the remaining
substring and then calculate the difference #X - #Y.

Example:
Input: 1213211
Output: 2 (3 - 1)

This task is solved with a multi-head attention network.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

SEQ_LEN = 5
VOCAB_SIZE = 6
NUM_TRAINING_STEPS = 25000
BATCH_SIZE = 64


# This function generates data samples
def get_data_sample(batch_size=1):
    random_seq = torch.randint(
        low=0, high=VOCAB_SIZE - 1, size=[batch_size, SEQ_LEN + 2]
    )

    # Calculate the ground truth output for the random sequence and store it in 'gts'
    X = random_seq[:, 0]  # Shape is batch_size
    Y = random_seq[:, 1]
    remaining_seq = random_seq[:, 2:]  # Shape: [batch_size, SEQ_LEN]
    count_X = (remaining_seq == X.unsqueeze(1)).sum(dim=1)
    count_Y = (remaining_seq == Y.unsqueeze(1)).sum(dim=1)
    gts = count_X - count_Y
    # Ensure that GT is non-negative
    # Ensure non-negative for CrossEntropyLoss. gt can be negative for the problem. but Cross entropy expects class indices to be non-negative
    gts += SEQ_LEN
    return random_seq, gts


# Network definition
class Net(nn.Module):
    def __init__(self, num_encoding_layers=1, num_hidden=64, num_heads=4):
        super().__init__()

        self.embedding = nn.Embedding(VOCAB_SIZE, num_hidden)
        positional_encoding = torch.empty([SEQ_LEN + 2, 1])
        nn.init.normal_(positional_encoding)
        self.positional_encoding = nn.Parameter(positional_encoding, requires_grad=True)
        q = torch.empty([1, num_hidden])
        nn.init.normal_(q)
        self.q = nn.Parameter(q, requires_grad=True)
        self.encoding_layers = torch.nn.ModuleList(
            [EncodingLayer(num_hidden, num_heads) for _ in range(num_encoding_layers)]
        )
        self.decoding_layer = MultiHeadAttention(num_hidden, num_heads)
        self.c1 = nn.Conv1d(num_hidden + 1, num_hidden, 1)
        self.fc1 = nn.Linear(num_hidden, 2 * SEQ_LEN + 1)

    def forward(self, x):
        x = self.embedding(x)
        B = x.shape[0]
        # In the following lines I add a (trainable) positional encoding to the representation.
        # We need to distinguish between the initial X/Y positions and the remaining counting positions
        # this is different that just a general sequence. here, the first two position are specific and unique (ref_X, ref_Y)
        # and the rest of the sequence is a counting sequence.
        # Positional encoding wont be needed if the order of the sequence does not matter like in a classification task (counting tasks like "is the number divisible by 3?")
        positional_encoding = self.positional_encoding.unsqueeze(0)
        positional_encoding = positional_encoding.repeat([B, 1, 1])
        x = torch.cat([x, positional_encoding], axis=-1)
        x = x.transpose(1, 2)
        x = self.c1(x)
        x = x.transpose(1, 2)
        for encoding_layer in self.encoding_layers:
            x = encoding_layer(x)
        q = self.q.unsqueeze(0).repeat([B, 1, 1])
        x = self.decoding_layer(q, x, x)
        x = x.squeeze(1)
        x = self.fc1(x)
        return x


class EncodingLayer(nn.Module):
    def __init__(self, num_hidden, num_heads):
        super().__init__()

        self.att = MultiHeadAttention(embed_dim=num_hidden, num_heads=num_heads)
        self.c1 = nn.Conv1d(num_hidden, 2 * num_hidden, 1)
        self.c2 = nn.Conv1d(2 * num_hidden, num_hidden, 1)
        self.norm1 = nn.LayerNorm([num_hidden])
        self.norm2 = nn.LayerNorm([num_hidden])

    def forward(self, x):
        x = self.att(x, x, x)
        x = self.norm1(x)
        x1 = x.transpose(1, 2)
        x1 = self.c1(x1)
        x1 = F.relu(x1)
        x1 = self.c2(x1)
        x1 = F.relu(x1)
        x1 = x1.transpose(1, 2)
        x = x + x1
        x = self.norm2(x)
        return x


"""The following two classes implement Attention and Multi-Head Attention from
the paper "Attention Is All You Need" by Ashish Vaswani et al.
Q is Query, K is Key and V is Value.
Query is something that we want to find relevant information for. something like a google search.
Here, the query can be like "What information do I need to extract to solve this task? -->  counting information about specific digits"
Key is the information that is available to us.
Here, the key is "which elements are present to extract the info? --> complete sequence of the digits we have"
Value is the actual information to extract from the key
Here, value is " What actual information to extract from the sequence of digits? --> representation of the sequence [ref_x, ref_y, count_a, count_b, ...]"
In the following lines, we will implement a naive version of Multi-Head Attention"""


class Attention(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super().__init__()

        self.W_q = nn.Linear(input_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(input_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(input_dim, embed_dim, bias=False)
        self.embed_dim = embed_dim

    def forward(self, q, k, v):
        # q, k, and v are batch-first
        # First, we calculate a trainable linear projection of q, k and v.
        # Then calculate the scaled dot-product attention as described in
        # Section 3.2.1 of the paper.
        Q = self.W_q(q)
        K = self.W_k(k)
        V = self.W_v(v)
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.embed_dim**0.5)
        attention_weights = F.softmax(scores, dim=-1)
        # Apply attention weights to value
        result = torch.matmul(attention_weights, V)

        return result


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.head_dim = embed_dim // num_heads
        self.attention_heads = nn.ModuleList(
            [Attention(embed_dim, self.head_dim) for _ in range(num_heads)]
        )
        self.W_o = nn.Linear(embed_dim, embed_dim)

    def forward(self, q, k, v):
        # q, k, and v are batch-first
        # Implement multi-head attention as described in Section 3.2.2 of the paper.
        # head_outputs = []
        head_outputs = torch.tensor([], requires_grad=True)
        for attention_head in self.attention_heads:
            head_outputs = torch.cat((head_outputs, attention_head(q, k, v)), dim=-1)
        # concat_output = torch.cat(head_outputs, dim=-1)
        # Final linear projection
        result = self.W_o(head_outputs)

        return result


# Instantiate network, loss function and optimizer
net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.005, momentum=0.9)

# Train the network
for i in range(NUM_TRAINING_STEPS):
    inputs, labels = get_data_sample(BATCH_SIZE)

    optimizer.zero_grad()
    outputs = net(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    accuracy = (torch.argmax(outputs, axis=-1) == labels).float().mean()

    if i % 100 == 0:
        print(
            "[%d/%d] loss: %.3f, accuracy: %.3f"
            % (i, NUM_TRAINING_STEPS - 1, loss.item(), accuracy.item())
        )
    if i == NUM_TRAINING_STEPS - 1:
        print("Final accuracy: %.3f, expected %.3f" % (accuracy.item(), 1.0))
