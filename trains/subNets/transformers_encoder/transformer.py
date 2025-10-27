import math
import torch
import torch.nn.functional as F
from torch import nn
from .multihead_attention import MultiheadAttention
from .position_embedding import SinusoidalPositionalEmbedding

class TransformerEncoder(nn.Module):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.
    Args:
        embed_tokens (torch.nn.Embedding): input embedding
        num_heads (int): number of heads
        layers (int): number of layers
        attn_dropout (float): dropout applied on the attention weights
        relu_dropout (float): dropout applied on the first layer of the residual block
        res_dropout (float): dropout applied on the residual block
        attn_mask (bool): whether to apply mask on the attention weights
    """

    def __init__(self, embed_dim, num_heads, layers, attn_dropout=0.0, relu_dropout=0.0, res_dropout=0.0,
                 embed_dropout=0.0, attn_mask=False):
        super().__init__()
        self.dropout = embed_dropout      # Embedding dropout
        self.attn_dropout = attn_dropout
        self.embed_dim = embed_dim
        self.embed_scale = math.sqrt(embed_dim)
        self.embed_positions = SinusoidalPositionalEmbedding(embed_dim)
        
        self.attn_mask = attn_mask

        self.layers = nn.ModuleList([])
        for layer in range(layers):
            new_layer = TransformerEncoderLayer(embed_dim,
                                                num_heads=num_heads,
                                                attn_dropout=attn_dropout,
                                                relu_dropout=relu_dropout,
                                                res_dropout=res_dropout,
                                                attn_mask=attn_mask)
            self.layers.append(new_layer)

        self.register_buffer('version', torch.Tensor([2]))
        self.normalize = True
        if self.normalize:
            self.layer_norm = LayerNorm(embed_dim)

    def forward(self, x_in, x_in_k = None, x_in_v = None):
        """
        Args:
            x_in (FloatTensor): embedded input of shape `(src_len, batch, embed_dim)`
            x_in_k (FloatTensor): embedded input of shape `(src_len, batch, embed_dim)`
            x_in_v (FloatTensor): embedded input of shape `(src_len, batch, embed_dim)`
        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
        """
        # embed tokens and positions
        x = self.embed_scale * x_in
        if self.embed_positions is not None:
            x += self.embed_positions(x_in.transpose(0, 1)[:, :, 0]).transpose(0, 1)   # Add positional embedding
        x = F.dropout(x, p=self.dropout, training=self.training)

        if x_in_k is not None and x_in_v is not None:
            # embed tokens and positions    
            x_k = self.embed_scale * x_in_k
            x_v = self.embed_scale * x_in_v
            if self.embed_positions is not None:
                x_k += self.embed_positions(x_in_k.transpose(0, 1)[:, :, 0]).transpose(0, 1)   # Add positional embedding
                x_v += self.embed_positions(x_in_v.transpose(0, 1)[:, :, 0]).transpose(0, 1)   # Add positional embedding
            x_k = F.dropout(x_k, p=self.dropout, training=self.training)
            x_v = F.dropout(x_v, p=self.dropout, training=self.training)
        
        # encoder layers
        intermediates = [x]
        for layer in self.layers:
            if x_in_k is not None and x_in_v is not None:
                x = layer(x, x_k, x_v)
            else:
                x = layer(x)
            intermediates.append(x)

        if self.normalize:
            x = self.layer_norm(x)

        return x

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions())


class TransformerEncoderLayer(nn.Module):
    """Encoder layer block.
    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: `dropout -> add residual -> layernorm`. In the
    tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.encoder_normalize_before* to ``True``.
    Args:
        embed_dim: Embedding dimension
    """

    def __init__(self, embed_dim, num_heads=4, attn_dropout=0.1, relu_dropout=0.1, res_dropout=0.1,
                 attn_mask=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        self.self_attn = MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            attn_dropout=attn_dropout
        )
        self.attn_mask = attn_mask

        self.relu_dropout = relu_dropout
        self.res_dropout = res_dropout
        self.normalize_before = True

        self.fc1 = Linear(self.embed_dim, 4*self.embed_dim)   # The "Add & Norm" part in the paper
        self.fc2 = Linear(4*self.embed_dim, self.embed_dim)
        self.layer_norms = nn.ModuleList([LayerNorm(self.embed_dim) for _ in range(2)])

    def forward(self, x, x_k=None, x_v=None):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.
            x_k (Tensor): same as x
            x_v (Tensor): same as x
        Returns:
            encoded output of shape `(batch, src_len, embed_dim)`
        """
        residual = x
        x = self.maybe_layer_norm(0, x, before=True)
        mask = buffered_future_mask(x, x_k) if self.attn_mask else None
        if x_k is None and x_v is None:
            x, _ = self.self_attn(query=x, key=x, value=x, attn_mask=mask)
        else:
            x_k = self.maybe_layer_norm(0, x_k, before=True)
            x_v = self.maybe_layer_norm(0, x_v, before=True) 
            x, _ = self.self_attn(query=x, key=x_k, value=x_v, attn_mask=mask)
        x = F.dropout(x, p=self.res_dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(0, x, after=True)

        residual = x
        x = self.maybe_layer_norm(1, x, before=True)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.res_dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(1, x, after=True)
        return x

    def maybe_layer_norm(self, i, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return self.layer_norms[i](x)
        else:
            return x

def fill_with_neg_inf(t):
    """FP16-compatible function that fills a tensor with -inf."""
    return t.float().fill_(float('-inf')).type_as(t)


def buffered_future_mask(tensor, tensor2=None):
    dim1 = dim2 = tensor.size(0)
    if tensor2 is not None:
        dim2 = tensor2.size(0)
    future_mask = torch.triu(fill_with_neg_inf(torch.ones(dim1, dim2)), 1+abs(dim2-dim1))
    if tensor.is_cuda:
        future_mask = future_mask.to(tensor.device)
    return future_mask[:dim1, :dim2]


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m


def LayerNorm(embedding_dim):
    m = nn.LayerNorm(embedding_dim)
    return m


### JMT

class SequentialEncoder(nn.Sequential):
    def forward(self, x):
        for module in self._modules.values():
            x = module(x)
        return x

class TransformerEncoderBlock(nn.Module):
    def __init__(self, input_dim, num_heads, hidden_dim, num_layers):
        super(TransformerEncoderBlock, self).__init__()
        self.layers = SequentialEncoder(
            *[TransformerEncoderLayers(input_dim, num_heads, hidden_dim)
              for _ in range(num_layers)])

    def forward(self, x):
        x = self.layers(x)
        return x

class TransformerEncoderLayers(nn.Module):
    def __init__(self, input_dim, num_heads, hidden_dim):
        super(TransformerEncoderLayers, self).__init__()
        self.attention = nn.MultiheadAttention(input_dim, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.layer_norm2 = nn.LayerNorm(input_dim)

    def forward(self, x):
        # Apply self-attention
        attn_output, _ = self.attention(x, x, x)
        x = x + attn_output
        x = self.layer_norm1(x)

        # Apply feed forward network
        ff_output = self.feed_forward(x)
        x = x + ff_output
        x = self.layer_norm2(x)
        return x


# 3dJMT模型
# class MultimodalTransformer_w_JR(nn.Module):
#     def __init__(self, ecg_dim, gsr_dim, v_dim, num_heads, hidden_dim,
#                  num_layers, output_format):
#         super(MultimodalTransformer_w_JR, self).__init__()
#
#         self.output_format = output_format
#
#         # Cross attention
#         self.cross_attention_ecg = nn.MultiheadAttention(ecg_dim, num_heads)
#         self.cross_attention_gsr = nn.MultiheadAttention(gsr_dim, num_heads)
#         self.cross_attention_v = nn.MultiheadAttention(v_dim, num_heads)
#
#         if self.output_format == 'FC':
#             # Fully connected layer for the final output
#             self.out_layer1 = nn.Linear(ecg_dim*6, 1024)
#
#         elif self.output_format == 'SELF_ATTEN':
#             # Final attention module
#             self.final_visual_encoder = TransformerEncoderBlock(ecg_dim,
#                                                                 num_heads,
#                                                                 hidden_dim,
#                                                                 num_layers)
#             self.final_self_attention = nn.MultiheadAttention(ecg_dim, num_heads)
#
#         else:
#             raise NotImplementedError(self.output_format)
#
#     def forward(self, ecg_features, gsr_features, vision_features):
#         # Permute dimension from (batch, seq, feature) to (seq, batch, feature)
#         ecg_features = ecg_features.permute(1, 0, 2)
#         gsr_features = gsr_features.permute(1, 0, 2)
#         vision_features = vision_features.permute(1, 0, 2)
#
#         # Do all the cross-attention
#         cross_attention_output_ecg_gsr, _ = self.cross_attention_ecg(
#             ecg_features, gsr_features, gsr_features)
#         cross_attention_output_gsr_ecg, _ = self.cross_attention_gsr(
#             gsr_features, ecg_features, ecg_features)
#         cross_attention_output_v_ecg, _ = self.cross_attention_ecg(
#             vision_features, ecg_features, ecg_features)
#         cross_attention_output_ecg_v, _ = self.cross_attention_ecg(
#             ecg_features, vision_features, vision_features)
#         cross_attention_output_v_gsr, _ = self.cross_attention_v(
#             vision_features, gsr_features, gsr_features)
#         cross_attention_output_gsr_v, _ = self.cross_attention_gsr(
#             gsr_features, vision_features, vision_features)
#
#         if self.output_format == "SELF_ATTEN":
#             '''
#              --- [Start] Final Attention module ---
#             '''
#
#             stack_attention = torch.stack((cross_attention_output_ecg_gsr,
#                                            cross_attention_output_gsr_ecg,
#                                            cross_attention_output_v_ecg,
#                                            cross_attention_output_ecg_v,
#                                            cross_attention_output_v_gsr,
#                                            cross_attention_output_gsr_v), dim=2)
#             stack_attention = stack_attention.permute(1, 0, 2, 3)
#             stack_attention_flatten = stack_attention.flatten(0, 1).permute(1, 0, 2)
#             stack_attention_flatten = stack_attention_flatten
#             b_size = stack_attention.shape[0]
#             seq_size = stack_attention.shape[1]
#             final_encoded = self.final_visual_encoder(stack_attention_flatten)
#
#             final_attention, _ = self.final_self_attention(final_encoded,
#                                                            final_encoded,
#                                                            final_encoded)
#             final_attention = final_attention.permute(1, 0, 2)
#             final_attention_unflatten = final_attention.unflatten(0, (
#             b_size, seq_size))
#
#             final_attention_unflatten = final_attention_unflatten[:, :, -1, :]
#             # bsz, seq, feature.
#             '''
#              --- [End] Final Attention module ---
#             '''
#
#             return final_attention_unflatten # (batch, seq, d_ecg)
#
#         elif self.output_format == 'FC':
#             # Concatenate Cross-attention outputs
#             concat_attention = torch.cat((cross_attention_output_ecg_gsr,
#                                            cross_attention_output_gsr_ecg,
#                                            cross_attention_output_v_ecg,
#                                            cross_attention_output_ecg_v,
#                                            cross_attention_output_v_gsr,
#                                            cross_attention_output_gsr_v), dim=2)
#             out = self.out_layer1(concat_attention)  # bsz, seq, 1024
#
#             return out # (batch, seq, 1024)
#
#         else:
#             raise NotImplementedError(self.output_format)

# 2dJMT模型
class MultimodalTransformer_w_JR(nn.Module):
    def __init__(self, ecg_dim, gsr_dim, v_dim, num_heads, hidden_dim,
                 num_layers, output_format):
        super(MultimodalTransformer_w_JR, self).__init__()

        self.output_format = output_format
        self.ecg_dim = ecg_dim

        # Cross attention
        self.cross_attention_ecg = nn.MultiheadAttention(ecg_dim, num_heads)
        self.cross_attention_gsr = nn.MultiheadAttention(gsr_dim, num_heads)
        self.cross_attention_v = nn.MultiheadAttention(v_dim, num_heads)

        if self.output_format == 'FC':
            # Fully connected layer for the final output
            self.out_layer1 = nn.Linear(ecg_dim * 6, 1024)

        elif self.output_format == 'SELF_ATTEN':
            # Final attention module - 修改为处理2D输入
            self.final_visual_encoder = TransformerEncoderBlock(ecg_dim,
                                                                num_heads,
                                                                hidden_dim,
                                                                num_layers)
            self.final_self_attention = nn.MultiheadAttention(ecg_dim, num_heads)

        else:
            raise NotImplementedError(self.output_format)

    def forward(self, ecg_features, gsr_features, vision_features):
        """
        输入形状: (batch_size, feature_dim)
        输出形状: (batch_size, feature_dim)
        """
        # 对于2D输入，添加序列维度 (batch, feat) -> (1, batch, feat)
        ecg_features = ecg_features.unsqueeze(0)  # (1, batch, ecg_dim)
        gsr_features = gsr_features.unsqueeze(0)  # (1, batch, gsr_dim)
        vision_features = vision_features.unsqueeze(0)  # (1, batch, v_dim)

        # Do all the cross-attention
        cross_attention_output_ecg_gsr, _ = self.cross_attention_ecg(
            ecg_features, gsr_features, gsr_features)
        cross_attention_output_gsr_ecg, _ = self.cross_attention_gsr(
            gsr_features, ecg_features, ecg_features)
        cross_attention_output_v_ecg, _ = self.cross_attention_ecg(
            vision_features, ecg_features, ecg_features)
        cross_attention_output_ecg_v, _ = self.cross_attention_ecg(
            ecg_features, vision_features, vision_features)
        cross_attention_output_v_gsr, _ = self.cross_attention_v(
            vision_features, gsr_features, gsr_features)
        cross_attention_output_gsr_v, _ = self.cross_attention_gsr(
            gsr_features, vision_features, vision_features)

        if self.output_format == "SELF_ATTEN":
            '''
             --- [Start] Final Attention module ---
            '''
            # 对于2D输入，stack的维度调整
            stack_attention = torch.stack((cross_attention_output_ecg_gsr,
                                           cross_attention_output_gsr_ecg,
                                           cross_attention_output_v_ecg,
                                           cross_attention_output_ecg_v,
                                           cross_attention_output_v_gsr,
                                           cross_attention_output_gsr_v), dim=2)
            # stack_attention shape: (1, batch, 6, ecg_dim)

            # 调整维度以适应原有的处理逻辑
            stack_attention = stack_attention.permute(1, 0, 2, 3)  # (batch, 1, 6, ecg_dim)
            stack_attention_flatten = stack_attention.flatten(0, 1)  # (batch, 6, ecg_dim)
            stack_attention_flatten = stack_attention_flatten.permute(1, 0, 2)  # (6, batch, ecg_dim)

            b_size = stack_attention.shape[0]  # batch_size
            seq_size = stack_attention.shape[1]  # 1

            final_encoded = self.final_visual_encoder(stack_attention_flatten)

            final_attention, _ = self.final_self_attention(final_encoded,
                                                           final_encoded,
                                                           final_encoded)
            final_attention = final_attention.permute(1, 0, 2)  # (batch, 6, ecg_dim)

            # 由于只有1个时间步，直接取最后一个特征
            final_attention_unflatten = final_attention.unflatten(0, (b_size, seq_size))  # (batch, 1, 6, ecg_dim)

            # 取最后一个交叉注意力输出的特征 (batch, 1, ecg_dim)
            final_attention_unflatten = final_attention_unflatten[:, :, -1, :]

            # 移除序列维度 (batch, 1, ecg_dim) -> (batch, ecg_dim)
            final_output = final_attention_unflatten.squeeze(1)
            '''
             --- [End] Final Attention module ---
            '''

            return final_output

        elif self.output_format == 'FC':
            # Concatenate Cross-attention outputs
            concat_attention = torch.cat((cross_attention_output_ecg_gsr,
                                          cross_attention_output_gsr_ecg,
                                          cross_attention_output_v_ecg,
                                          cross_attention_output_ecg_v,
                                          cross_attention_output_v_gsr,
                                          cross_attention_output_gsr_v), dim=2)
            # concat_attention shape: (1, batch, ecg_dim*6)

            out = self.out_layer1(concat_attention)  # (1, batch, 1024)

            # 移除序列维度 (1, batch, 1024) -> (batch, 1024)
            out = out.squeeze(0)

            return out

        else:
            raise NotImplementedError(self.output_format)


if __name__ == '__main__':
    encoder = TransformerEncoder(300, 4, 2)
    x = torch.tensor(torch.rand(20, 2, 300))
    print(encoder(x).shape)
