# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Dict, List, Optional
import torch
import torch.nn as nn
from torch import Tensor
from fairseq import utils
from fairseq.distributed import fsdp_wrap
from fairseq.models import FairseqEncoder
from fairseq.models.transformer import TransformerConfig
from fairseq.modules import (
    FairseqDropout,
    LayerDropModuleList,
    LayerNorm,
    PositionalEmbedding,
    SinusoidalPositionalEmbedding,
    transformer_layer,
)
from fairseq.modules.checkpoint_activations import checkpoint_wrapper
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_
import dgl
from gat_model import MultiHeadGATLayer
from fairseq.models.transformer import buildHomo
import numpy as np


def module_name_fordropout(module_name: str) -> str:
    if module_name == "TransformerEncoderBase":
        return "TransformerEncoder"
    else:
        return module_name


class TransformerEncoderBase(FairseqEncoder):
    """
    Transformer encoder consisting of *cfg.encoder.layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
    """

    def __init__(self, cfg, dictionary, embed_tokens, return_fc=False):
        self.cfg = cfg
        super().__init__(dictionary)
        self.register_buffer("version", torch.Tensor([3]))
        self.dropout_module = FairseqDropout(
            cfg.dropout, module_name=module_name_fordropout(self.__class__.__name__)
        )
        self.encoder_layerdrop = cfg.encoder.layerdrop
        self.return_fc = return_fc
        embed_dim = embed_tokens.embedding_dim
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = cfg.max_source_positions
        self.embed_tokens = embed_tokens
        self.embed_scale = 1.0 if cfg.no_scale_embedding else math.sqrt(embed_dim)
        self.embed_positions = (
            PositionalEmbedding(
                cfg.max_source_positions,
                embed_dim,
                self.padding_idx,
                learned=cfg.encoder.learned_pos,
            )
            if not cfg.no_token_positional_embeddings
            else None
        )

        self.fc1 = nn.Linear(1024, 1024)
        self.fc2 = nn.Linear(1024, 1)
        self.slice_rate = 0
        self.n_graph_layers = 3
        self.graphconv_dropout = nn.Dropout(p=0.0)

        self.g, self.node_char_dict, self.token_char_dict, self.token_node_dict, self.node_token_dict, self.feat_Conv, self.node_embed_tensor = self.generate_dglgraph(
            slice_rate=self.slice_rate)

        self.graph_layers = nn.ModuleList()
        self.graph_layers.append(
            MultiHeadGATLayer(g=self.g, in_feats=int((1 - self.slice_rate) * 512),
                              out_feats=int((1 - self.slice_rate) * 512), num_heads=8))
        for i in range(self.n_graph_layers - 1):
            self.graph_layers.append(
                MultiHeadGATLayer(g=self.g, in_feats=int((1 - self.slice_rate) * 512),
                                  out_feats=int((1 - self.slice_rate) * 512), num_heads=8))

        if cfg.layernorm_embedding:
            self.layernorm_embedding = LayerNorm(embed_dim, export=cfg.export)
        else:
            self.layernorm_embedding = None

        if not cfg.adaptive_input and cfg.quant_noise.pq > 0:
            self.quant_noise = apply_quant_noise_(
                nn.Linear(embed_dim, embed_dim, bias=False),
                cfg.quant_noise.pq,
                cfg.quant_noise.pq_block_size,
            )
        else:
            self.quant_noise = None

        if self.encoder_layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.encoder_layerdrop)
        else:
            self.layers = nn.ModuleList([])
        self.layers.extend(
            [self.build_encoder_layer(cfg) for i in range(cfg.encoder.layers)]
        )
        self.num_layers = len(self.layers)
        if cfg.encoder.normalize_before:
            self.layer_norm = LayerNorm(embed_dim, export=cfg.export)
        else:
            self.layer_norm = None

    def build_encoder_layer(self, cfg):
        layer = transformer_layer.TransformerEncoderLayerBase(
            cfg, return_fc=self.return_fc
        )
        checkpoint = cfg.checkpoint_activations
        if checkpoint:
            offload_to_cpu = cfg.offload_activations
            layer = checkpoint_wrapper(layer, offload_to_cpuTransformerDecoderBase=offload_to_cpu)
        min_params_to_wrap = cfg.min_params_to_wrap if not checkpoint else 0
        layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
        return layer

    def forward_embedding(
            self, src_tokens, token_embedding: Optional[torch.Tensor] = None
    ):
        if token_embedding is None:
            token_embedding = self.embed_tokens(src_tokens)

        src_tokens_T = src_tokens.cpu().numpy()
        token_embedding_dict = self.generate_token_embedding_dict()
        transformer_feat, graphconv_feat = self.generate_gate_input(self.embed_tokens,
                                                                    token_embedding_dict)
        gating_weight = self.En_Controller(transformer_feat, graphconv_feat)
        gating_weight_dict = {}
        for index, i in enumerate(gating_weight):
            gating_weight_dict.update({index: i})
        graphconv_embed = [[token_embedding_dict[src_tokens_T[i][j]] for j in
                            range(len(src_tokens_T[0]))] for i in
                           range(len(src_tokens_T))]
        graphconv_embed = torch.from_numpy(np.array(graphconv_embed)).cuda()

        gating_weight = torch.stack(
            [torch.stack([gating_weight_dict[x] for x in row]) for row in src_tokens.tolist()], dim=0)
        token_embedding = token_embedding * gating_weight + graphconv_embed * (1 - gating_weight)

        x = embed = self.embed_scale * token_embedding
        if self.embed_positions is not None:
            x = embed + self.embed_positions(src_tokens)
        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)
        x = self.dropout_module(x)
        if self.quant_noise is not None:
            x = self.quant_noise(x)
        return x, embed

    def forward(
            self,
            src_tokens,
            src_lengths: Optional[torch.Tensor] = None,
            return_all_hiddens: bool = False,
            token_embeddings: Optional[torch.Tensor] = None,
    ):
        return self.forward_scriptable(
            src_tokens, src_lengths, return_all_hiddens, token_embeddings
        )

    def forward_scriptable(
            self,
            src_tokens,
            src_lengths: Optional[torch.Tensor] = None,
            return_all_hiddens: bool = False,
            token_embeddings: Optional[torch.Tensor] = None,
    ):
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        has_pads = src_tokens.device.type == "xla" or encoder_padding_mask.any()
        x, encoder_embedding = self.forward_embedding(src_tokens, token_embeddings)
        if has_pads:
            x = x * (1 - encoder_padding_mask.unsqueeze(-1).type_as(x))
        x = x.transpose(0, 1)
        encoder_states = []
        fc_results = []
        if return_all_hiddens:
            encoder_states.append(x)
        layer = self.layers[0]
        BT_flag = False
        NT_flag = False
        BT_version = False
        NT_version = False
        if "fb" in torch.__version__:
            BT_version = True
            NT_version = True
        else:
            if "+" in torch.__version__:
                torch_version = torch.__version__.split("+")[0]
            else:
                torch_version = torch.__version__

            torch_version = torch_version.split(".")
            int_version = (
                    int(torch_version[0]) * 1000
                    + int(torch_version[1]) * 10
                    + int(torch_version[2])
            )
            if len(torch_version) == 3:
                if int_version >= 1120:
                    BT_version = True
                if int_version >= 1131:
                    NT_version = True
            elif len(torch_version) == 4:
                if int_version >= 1130:
                    BT_version = True
                if int_version >= 1131 or (
                        int_version == 1130 and torch_version[3][3:] >= "20220613"
                ):
                    NT_version = True

        if (
                BT_version
                and x.dim() == 3
                and layer.load_to_BT
                and not layer.return_fc
                and layer.can_use_fastpath
                and not layer.training
                and not layer.ever_training
                and not layer.cfg_checkpoint_activations
        ):
            x = x.transpose(0, 1)
            if NT_version:
                if (
                        encoder_padding_mask is not None
                        and torch._nested_tensor_from_mask_left_aligned(
                    x, encoder_padding_mask.logical_not()
                )
                ):
                    if not torch.is_grad_enabled() or not x.requires_grad:
                        x = torch._nested_tensor_from_mask(
                            x, encoder_padding_mask.logical_not()
                        )
                        NT_flag = True
            BT_flag = True
        if NT_flag:
            processing_mask = None
        else:
            processing_mask = encoder_padding_mask

        encoder_padding_mask_out = processing_mask if has_pads else None
        for layer in self.layers:
            lr = layer(x, encoder_padding_mask=encoder_padding_mask_out)
            if isinstance(lr, tuple) and len(lr) == 2:
                x, fc_result = lr
            else:
                x = lr
                fc_result = None
            if return_all_hiddens and not torch.jit.is_scripting():
                assert encoder_states is not None
                encoder_states.append(x)
                fc_results.append(fc_result)
        if NT_flag:
            x = x.to_padded_tensor(0.0)
        if NT_flag or BT_flag:
            x = x.transpose(0, 1)
        if self.layer_norm is not None:
            x = self.layer_norm(x)
        src_lengths = (
            src_tokens.ne(self.padding_idx)
                .sum(dim=1, dtype=torch.int32)
                .reshape(-1, 1)
                .contiguous()
        )
        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask],  # B x T
            "encoder_embedding": [encoder_embedding],  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "fc_results": fc_results,  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [src_lengths],
        }

    @torch.jit.export
    def reorder_encoder_out(self, encoder_out: Dict[str, List[Tensor]], new_order):
        if len(encoder_out["encoder_out"]) == 0:
            new_encoder_out = []
        else:
            new_encoder_out = [encoder_out["encoder_out"][0].index_select(1, new_order)]
        if len(encoder_out["encoder_padding_mask"]) == 0:
            new_encoder_padding_mask = []
        else:
            new_encoder_padding_mask = [
                encoder_out["encoder_padding_mask"][0].index_select(0, new_order)
            ]
        if len(encoder_out["encoder_embedding"]) == 0:
            new_encoder_embedding = []
        else:
            new_encoder_embedding = [
                encoder_out["encoder_embedding"][0].index_select(0, new_order)
            ]
        if len(encoder_out["src_tokens"]) == 0:
            src_tokens = []
        else:
            src_tokens = [(encoder_out["src_tokens"][0]).index_select(0, new_order)]
        if len(encoder_out["src_lengths"]) == 0:
            src_lengths = []
        else:
            src_lengths = [(encoder_out["src_lengths"][0]).index_select(0, new_order)]
        encoder_states = encoder_out["encoder_states"]
        if len(encoder_states) > 0:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(1, new_order)

        return {
            "encoder_out": new_encoder_out,  # T x B x C
            "encoder_padding_mask": new_encoder_padding_mask,  # B x T
            "encoder_embedding": new_encoder_embedding,  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": src_tokens,  # B x T
            "src_lengths": src_lengths,  # B x 1
        }

    @torch.jit.export
    def _reorder_encoder_out(self, encoder_out: Dict[str, List[Tensor]], new_order):
        """Dummy re-order function for beamable enc-dec attention"""
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions)

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = "{}.embed_positions.weights".format(name)
            if weights_key in state_dict:
                print("deleting {0}".format(weights_key))
                del state_dict[weights_key]
            state_dict[
                "{}.embed_positions._float_tensor".format(name)
            ] = torch.FloatTensor(1)
        for i in range(self.num_layers):
            self.layers[i].upgrade_state_dict_named(
                state_dict, "{}.layers.{}".format(name, i)
            )

        version_key = "{}.version".format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) < 2:
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])
        return state_dict

    def generate_gate_input(self, transformer_embed, graphconv_dict):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        feat_A = transformer_embed(torch.tensor(range(14544)).to(device))
        feat_B = torch.tensor(np.array([graphconv_dict[i] for i in range(14544)]))
        return feat_A, feat_B

    # Gate-weighted controller
    def En_Controller(self, inp1, inp2):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        inp2 = inp2.to(device)
        x = torch.cat([inp1, inp2], dim=1)
        x = self.fc1(x)
        x = torch.sigmoid(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x

    # Generate DGL graph through dependency syntax analysis
    def generate_dglgraph(self, slice_rate):
        homo_dict = {
            "sep": " ",
            "zipPath": r"fairseq/models/transformer/dataset_encoder",
            "graphType": "homo"}
        dependency_dgl = buildHomo(homo_dict)
        g = dependency_dgl.graph
        g = dgl.add_self_loop(g)
        _, node_char_dict = dependency_dgl.homoNode()
        special_token_dict = {len(node_char_dict): '<CLS>', len(node_char_dict) + 1: '<PAD>',
                              len(node_char_dict) + 2: '<EOS>',
                              len(node_char_dict) + 3: '<UNK>'}
        node_char_dict.update(special_token_dict)
        torch.manual_seed(100)

        # Generate a mapping dictionary from tokens to characters
        def generate_token_char_dict(dictionary):
            with open(dictionary, 'r', encoding='UTF-8') as F:
                dict_list = F.readlines()
                token_char_dict = {}
                for index, i in enumerate(dict_list):
                    token_char_dict.update({index + 4: i.split(' ')[0]})
            special_token_dict = {0: '<CLS>', 1: '<PAD>', 2: '<EOS>', 3: '<UNK>'}
            token_char_dict.update(special_token_dict)
            return token_char_dict

        token_char_dict = generate_token_char_dict(
            'fairseq/models/transformer/dataset_encoder/dict.zh.txt')
        token_char_dict = dict(sorted(token_char_dict.items()))
        token_node_dict = {}
        for index1, i in enumerate(token_char_dict.items()):
            for index2, j in enumerate(node_char_dict.items()):
                if i[1] == j[1]:
                    token_node_dict.update({i[0]: j[0]})
                else:
                    pass
        token_embed_tensor = self.embed_tokens(torch.tensor(range(len(token_char_dict))))
        # Construct a mapping dictionary between tokens and embeddings
        token_embed_dict = {}
        for index, i in enumerate(token_embed_tensor):
            token_embed_dict.update({index: i})
        # Construct a mapping dictionary in DGL between node IDs and embeddings, named 'node_embed_dict_former,'
        # to prepare for training within DGL
        node_embed_dict_former = dict(zip(token_node_dict.values(), token_embed_dict.values()))
        node_embed_dict_former = dict(sorted(node_embed_dict_former.items()))
        # Combine the values of the dictionary (1D tensors) into a 2D tensor for input to graph convolution
        tensor_list = list(node_embed_dict_former.values())
        node_embed_tensor = torch.stack(tensor_list)
        feat_Conv = node_embed_tensor[:-4]
        node_token_dict = {v: k for k, v in token_node_dict.items()}
        node_token_dict = dict(sorted(node_token_dict.items()))
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        feat_Conv = feat_Conv.to(device)
        g = g.to(device)
        node_embed_tensor = node_embed_tensor.to(device)
        return g, node_char_dict, token_char_dict, token_node_dict, node_token_dict, feat_Conv, node_embed_tensor

    # Generate embeddings through graph convolution and map them to tokens
    def generate_token_embedding_dict(self):
        feat_Conv = self.feat_Conv
        g = self.g
        node_embed_tensor = self.node_embed_tensor
        node_token_dict = self.node_token_dict
        for i, graph_layer in enumerate(self.graph_layers):
            if i != 0:
                feat_Conv = self.graphconv_dropout(feat_Conv)
            feat_Conv = graph_layer(g, feat_Conv)
        res2 = feat_Conv
        res3 = torch.cat([res2, node_embed_tensor[-4:]])
        node_embed_dict_later = {}
        for index, i in enumerate(res3):
            node_embed_dict_later.update({index: i})
        token_embedding_dict = dict(zip(node_token_dict.values(), node_embed_dict_later.values()))
        token_embedding_dict = dict(sorted(token_embedding_dict.items()))
        token_embedding_dict = {k: v.cpu().detach().numpy() for k, v in token_embedding_dict.items()}
        return token_embedding_dict


class TransformerEncoder(TransformerEncoderBase):
    def __init__(self, args, dictionary, embed_tokens, return_fc=False):
        self.args = args
        super().__init__(
            TransformerConfig.from_namespace(args),
            dictionary,
            embed_tokens,
            return_fc=return_fc,
        )

    def build_encoder_layer(self, args):
        return super().build_encoder_layer(
            TransformerConfig.from_namespace(args),
        )


class En_Controller(nn.Module):
    def __init__(self):
        super(En_Controller, self).__init__()
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, inp1, inp2):
        x = torch.cat([inp1, inp2], dim=1)
        x = self.fc1(x)
        x = torch.sigmoid(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x
