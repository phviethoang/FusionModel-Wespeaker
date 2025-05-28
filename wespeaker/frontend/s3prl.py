# Copyright (c) 2024 Hongji Wang (jijijiang77@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Union, List
import contextlib
import torch
import torch.nn as nn

import s3prl
from s3prl.nn import Featurizer, S3PRLUpstream

# class S3prlFrontend(nn.Module):
#     """Speech Pretrained Representation Frontend."""

#     def __init__(self,
#                  upstream_args: dict,
#                  download_dir: str = "./s3prl_hub",
#                  multilayer_feature: bool = True,
#                  layer: int = -1,
#                  frozen: bool = False,
#                  frame_shift: int = 20,
#                  frame_length: int = 20,
#                  sample_rate: int = 16000):
#         super().__init__()

#         self.multilayer_feature = multilayer_feature
#         self.layer = layer
#         self.frozen = frozen

#         if download_dir is not None:
#             s3prl.util.download.set_dir(download_dir)

#         assert upstream_args.get("name",
#                                  None) in S3PRLUpstream.available_names()
#         self.upstream = S3PRLUpstream(
#             upstream_args.get("name"),
#             path_or_url=upstream_args.get("path_or_url", None),
#             normalize=upstream_args.get("normalize", False),
#             extra_conf=upstream_args.get("extra_conf", None),
#         )
#         if getattr(self.upstream.upstream, "model", None):
#             if getattr(self.upstream.upstream.model, "feature_grad_mult",
#                        None) is not None:
#                 self.upstream.upstream.model.feature_grad_mult = 1.0
#         self.upstream.eval()

#         #fix bug với xlsr
#         for idx, (name, param) in enumerate(self.upstream.named_parameters()):
#             print(f"Index {idx}: {name}")
#         for name, param in self.upstream.named_parameters():
#             if "quantizer" in name or "project_q" in name:
#                 param.requires_grad_(False)
#         for name, param in self.upstream.named_parameters():
#             if "final_proj" in name:
#                 param.requires_grad_(False)


#         if layer != -1:
#             layer_selections = [layer]
#             assert not multilayer_feature, \
#                 "multilayer_feature must be False if layer is specified"
#         else:
#             layer_selections = None
#         self.featurizer = Featurizer(self.upstream,
#                                      layer_selections=layer_selections)

#         assert self.featurizer.downsample_rate == sample_rate * frame_shift // 1000

#         if self.frozen:
#             for param in self.upstream.parameters():
#                 param.requires_grad_(False)
#         else:
#             for name, param in self.upstream.named_parameters():
#                 if "mask_emb" in name:
#                     param.requires_grad_(False)

#     def output_size(self):
#         return self.featurizer.output_size

#     def forward(self, input: torch.Tensor, input_lengths: torch.LongTensor):
#         with torch.no_grad() if self.frozen else contextlib.nullcontext():
#             feats, feats_lens = self.upstream(input, input_lengths)
#         if self.layer != -1:
#             layer = self.layer
#             feats, feats_lens = feats[layer], feats_lens[layer]
#             return feats, feats_lens

#         if self.multilayer_feature:
#             feats, feats_lens = self.featurizer(feats, feats_lens)
#         else:
#             feats, feats_lens = self.featurizer(feats[-1:], feats_lens[-1:])

#         return feats, feats_lens

class S3prlFrontend(nn.Module):
    """Speech Pretrained Representation Frontend."""

    def __init__(self,
                 upstream_args: dict,
                 download_dir: str = "./s3prl_hub",
                 multilayer_feature: bool = True,
                 layer: Union[int, List[int]] = -1,  # Cho phép layer là int hoặc list
                 frozen: bool = False,
                 frame_shift: int = 20,
                 frame_length: int = 20,
                 sample_rate: int = 16000):
        super().__init__()

        self.multilayer_feature = multilayer_feature
        self.layer = layer  # Có thể là int (-1) hoặc list (ví dụ: [0, 1, ..., 11])
        self.frozen = frozen

        if download_dir is not None:
            s3prl.util.download.set_dir(download_dir)

        assert upstream_args.get("name",
                                None) in S3PRLUpstream.available_names()
        self.upstream = S3PRLUpstream(
            upstream_args.get("name"),
            path_or_url=upstream_args.get("path_or_url", None),
            normalize=upstream_args.get("normalize", False),
            extra_conf=upstream_args.get("extra_conf", None),
        )
        if getattr(self.upstream.upstream, "model", None):
            if getattr(self.upstream.upstream.model, "feature_grad_mult",
                      None) is not None:
                self.upstream.upstream.model.feature_grad_mult = 1.0
        self.upstream.eval()

        # Fix bug với xlsr
        for idx, (name, param) in enumerate(self.upstream.named_parameters()):
            print(f"Index {idx}: {name}")
        for name, param in self.upstream.named_parameters():
            if "quantizer" in name or "project_q" in name:
                param.requires_grad_(False)
        for name, param in self.upstream.named_parameters():
            if "final_proj" in name:
                param.requires_grad_(False)

        # Xử lý layer_selections dựa trên self.layer
        if isinstance(self.layer, list):
            layer_selections = self.layer  # Truyền list layer (ví dụ: [0, 1, ..., 11])
            assert not multilayer_feature, \
                "multilayer_feature must be False if layer is specified as a list"
        elif self.layer != -1:
            layer_selections = [self.layer]  # Chuyển int thành list
            assert not multilayer_feature, \
                "multilayer_feature must be False if layer is specified as an int"
        else:
            layer_selections = None  # Sử dụng tất cả layer nếu layer = -1
        self.featurizer = Featurizer(self.upstream,
                                     layer_selections=layer_selections)

        assert self.featurizer.downsample_rate == sample_rate * frame_shift // 1000

        if self.frozen:
            for param in self.upstream.parameters():
                param.requires_grad_(False)
        else:
            for name, param in self.upstream.named_parameters():
                if "mask_emb" in name:
                    param.requires_grad_(False)

    def output_size(self):
        return self.featurizer.output_size

    def forward(self, input: torch.Tensor, input_lengths: torch.LongTensor):
        with torch.no_grad() if self.frozen else contextlib.nullcontext():
            feats, feats_lens = self.upstream(input, input_lengths)
        if isinstance(self.layer, list):
            # Lấy các layer từ list
            selected_feats = [feats[i] for i in self.layer]
            selected_feats_lens = [feats_lens[i] for i in self.layer]
            feats, feats_lens = self.featurizer(selected_feats, selected_feats_lens)
            return feats, feats_lens
        
        if self.layer != -1: 
            layer = self.layer
            feats, feats_lens = feats[layer], feats_lens[layer]
            return feats, feats_lens

        if self.multilayer_feature: 
            feats, feats_lens = self.featurizer(feats, feats_lens)
        else:
            feats, feats_lens = self.featurizer(feats[-1:], feats_lens[-1:])

        return feats, feats_lens
    


# # Only get layer 1-12 from Large SSL model 
# class S3prlFrontend(nn.Module):
#     def __init__(self,
#                  upstream_args: dict,
#                  download_dir: str = "./s3prl_hub",
#                  multilayer_feature: bool = True,
#                  layer: int = -1,
#                  frozen: bool = False,
#                  frame_shift: int = 20,
#                  frame_length: int = 20,
#                  sample_rate: int = 16000,
#                  max_layer: int = None):  # Thêm tham số để giới hạn forward
#         super().__init__()

#         self.multilayer_feature = multilayer_feature
#         self.layer = layer
#         self.frozen = frozen
#         self.max_layer = max_layer

#         if download_dir is not None:
#             s3prl.util.download.set_dir(download_dir)

#         assert upstream_args.get("name",
#                                 None) in S3PRLUpstream.available_names()
#         upstream = S3PRLUpstream(
#             upstream_args.get("name"),
#             path_or_url=upstream_args.get("path_or_url", None),
#             normalize=upstream_args.get("normalize", False),
#             extra_conf=upstream_args.get("extra_conf", None),
#         )
#         if self.max_layer is not None:
#             self.upstream = CustomS3PRLUpstream(upstream, max_layer=self.max_layer)
#         else:
#             self.upstream = upstream
#         if getattr(self.upstream.upstream, "model", None):
#             if getattr(self.upstream.upstream.model, "feature_grad_mult",
#                       None) is not None:
#                 self.upstream.upstream.model.feature_grad_mult = 1.0
#         self.upstream.eval()

#         # Fix bug với xlsr
#         for idx, (name, param) in enumerate(self.upstream.named_parameters()):
#             print(f"Index {idx}: {name}")
#         for name, param in self.upstream.named_parameters():
#             if "quantizer" in name or "project_q" in name:
#                 param.requires_grad_(False)
#         for name, param in self.upstream.named_parameters():
#             if "final_proj" in name:
#                 param.requires_grad_(False)

#         # Truyền layer_selections cụ thể cho 12 layer
#         layer_selections = list(range(12))  # Lấy layer 0-11
#         # Luôn đặt layer_selections=None để Featurizer xử lý tất cả layer được truyền vào
#         self.featurizer = Featurizer(self.upstream, layer_selections=layer_selections)

#         assert self.featurizer.downsample_rate == sample_rate * frame_shift // 1000

#         if self.frozen:
#             for param in self.upstream.parameters():
#                 param.requires_grad_(False)
#         else:
#             for name, param in self.upstream.named_parameters():
#                 if "mask_emb" in name:
#                     param.requires_grad_(False)

#     def output_size(self):
#         return self.featurizer.output_size

#     def forward(self, input: torch.Tensor, input_lengths: torch.LongTensor):
#         with torch.no_grad() if self.frozen else contextlib.nullcontext():
#             feats, feats_lens = self.upstream(input, input_lengths)


#         # Lấy feature từ layer 0 đến 11 (12 layer đầu tiên)
#         feats = feats[:12]  # Lấy từ layer 0 đến 11
#         feats_lens = feats_lens[:12]

#         # Truyền qua Featurizer để tổng hợp
#         feats, feats_lens = self.featurizer(feats, feats_lens)

#         return feats, feats_lens

# # Định nghĩa CustomS3PRLUpstream (nếu muốn bỏ layer 12-23 khỏi forward)
# class CustomS3PRLUpstream(nn.Module):
#     def __init__(self, upstream, max_layer=11):  
#         super().__init__()
#         self.upstream = upstream
#         self.max_layer = max_layer
#         self.hidden_sizes = self.upstream.hidden_sizes
#         self.downsample_rates = self.upstream.downsample_rates
#         self.num_layers = self.upstream.num_layers

#     def forward(self, input, input_lengths):
#         feats, feats_lens = self.upstream(input, input_lengths)
#         limited_feats = feats[:self.max_layer + 1]  # Lấy từ layer 0 đến max_layer
#         limited_feats_lens = feats_lens[:self.max_layer + 1]
#         return limited_feats, limited_feats_lens