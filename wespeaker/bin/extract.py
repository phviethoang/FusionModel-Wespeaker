# Copyright (c) 2021 Hongji Wang (jijijiang77@gmail.com)
#               2022 Chengdong Liang (liangchengdong@mail.nwpu.edu.cn)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import os

import fire
import kaldiio
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
from wespeaker.dataset.dataset import Dataset
from wespeaker.dataset.dataset_utils import apply_cmvn, spec_aug
from wespeaker.frontend import *
from wespeaker.models.speaker_model import get_speaker_model
from wespeaker.utils.checkpoint import load_checkpoint
from wespeaker.utils.utils import parse_config_or_kwargs, validate_path
from wespeaker.models.fusion import CrossAttentionFusion

# # BASE MODEL
# def extract(config='conf/config.yaml', **kwargs):
#     # parse configs first
#     configs = parse_config_or_kwargs(config, **kwargs)

#     model_path = configs['model_path']
#     embed_ark = configs['embed_ark']
#     batch_size = configs.get('batch_size', 1)
#     num_workers = configs.get('num_workers', 1)

#     # Since the input length is not fixed, we set the built-in cudnn
#     # auto-tuner to False
#     torch.backends.cudnn.benchmark = False

#     test_conf = copy.deepcopy(configs['dataset_args'])
#     # model: frontend (optional) => speaker model
#     model = get_speaker_model(configs['model'])(**configs['model_args'])
#     frontend_type = test_conf.get('frontend', 'fbank')
#     if frontend_type != 'fbank':
#         frontend_args = frontend_type + "_args"
#         print('Initializing frontend model (this could take some time) ...')
#         frontend = frontend_class_dict[frontend_type](
#             **test_conf[frontend_args], sample_rate=test_conf['resample_rate'])
#         model.add_module("frontend", frontend)
#     print('Loading checkpoint ...')
#     load_checkpoint(model, model_path)
#     print('Finished !!! Start extracting ...')
#     device = torch.device("cuda")
#     model.to(device).eval()

#     # test_configs
#     # test_conf = copy.deepcopy(configs['dataset_args'])
#     test_conf['speed_perturb'] = False
#     if 'fbank_args' in test_conf:
#         test_conf['fbank_args']['dither'] = 0.0
#     test_conf['spec_aug'] = False
#     test_conf['shuffle'] = False
#     test_conf['aug_prob'] = configs.get('aug_prob', 0.0)
#     test_conf['filter'] = False

#     dataset = Dataset(configs['data_type'],
#                       configs['data_list'],
#                       test_conf,
#                       spk2id_dict={},
#                       whole_utt=(batch_size == 1),
#                       reverb_lmdb_file=configs.get('reverb_data', None),
#                       noise_lmdb_file=configs.get('noise_data', None),
#                       repeat_dataset=False)
#     dataloader = DataLoader(dataset,
#                             shuffle=False,
#                             batch_size=batch_size,
#                             num_workers=num_workers,
#                             prefetch_factor=4)

#     validate_path(embed_ark)
#     embed_ark = os.path.abspath(embed_ark)
#     embed_scp = embed_ark[:-3] + "scp"

#     with torch.no_grad():
#         with kaldiio.WriteHelper('ark,scp:' + embed_ark + "," +
#                                  embed_scp) as writer:
#             for _, batch in tqdm(enumerate(dataloader)):
#                 utts = batch['key']
#                 if frontend_type == 'fbank':
#                     features = batch['feat']
#                     features = features.float().to(device)  # (B,T,F)
#                 else:  # 's3prl'
#                     wavs = batch['wav']  # (B,1,W)
#                     wavs = wavs.squeeze(1).float().to(device)  # (B,W)
#                     wavs_len = torch.LongTensor([wavs.shape[1]]).repeat(
#                         wavs.shape[0]).to(device)  # (B)
#                     features, _ = model.frontend(wavs, wavs_len)

#                 # apply cmvn
#                 if test_conf.get('cmvn', True):
#                     features = apply_cmvn(features,
#                                           **test_conf.get('cmvn_args', {}))
#                 # spec augmentation
#                 if test_conf.get('spec_aug', False):
#                     features = spec_aug(features, **test_conf['spec_aug_args'])

#                 # Forward through model
#                 outputs = model(features)  # embed or (embed_a, embed_b)
#                 embeds = outputs[-1] if isinstance(outputs, tuple) else outputs
#                 embeds = embeds.cpu().detach().numpy()  # (B,F)

#                 for i, utt in enumerate(utts):
#                     embed = embeds[i]
#                     writer(utt, embed)

# # FUSION MODEL
# def extract(config='conf/config.yaml', **kwargs):
#     # Parse configs first
#     configs = parse_config_or_kwargs(config, **kwargs)

#     model_path = configs['model_path']
#     embed_ark = configs['embed_ark']
#     batch_size = configs.get('batch_size', 1)
#     num_workers = configs.get('num_workers', 1)

#     # Since the input length is not fixed, we set the built-in cudnn
#     # auto-tuner to False
#     torch.backends.cudnn.benchmark = False

#     # Test configs
#     test_conf = copy.deepcopy(configs['dataset_args'])
#     test_conf['speed_perturb'] = False
#     if 'fbank_args' in test_conf:
#         test_conf['fbank_args']['dither'] = 0.0
#     test_conf['spec_aug'] = False
#     test_conf['shuffle'] = False
#     test_conf['aug_prob'] = configs.get('aug_prob', 0.0)
#     test_conf['filter'] = False

#     # Dataset and DataLoader
#     dataset = Dataset(configs['data_type'],
#                       configs['data_list'],
#                       test_conf,
#                       spk2id_dict={},
#                       whole_utt=(batch_size == 1),
#                       reverb_lmdb_file=configs.get('reverb_data', None),
#                       noise_lmdb_file=configs.get('noise_data', None),
#                       repeat_dataset=False)
#     dataloader = DataLoader(dataset,
#                             shuffle=False,
#                             batch_size=batch_size,
#                             num_workers=num_workers,
#                             prefetch_factor=4)

#     # Model: frontend (optional) => speaker model
#     model = get_speaker_model(configs['model'])(**configs['model_args'])
#     frontend_type = test_conf.get('frontend', 'fbank')
#     use_fusion = test_conf.get('use_fusion', False)  # Lấy use_fusion từ config
#     frontend_output_dim = 0
#     if frontend_type != 'fbank':
#         frontend_args = frontend_type + "_args"
#         print('Initializing frontend model (this could take some time) ...')
#         frontend = frontend_class_dict[frontend_type](
#             **test_conf[frontend_args], sample_rate=test_conf['resample_rate'])
#         frontend_output_dim = frontend.output_size()
#         model.add_module("frontend", frontend)
#     # Tính input_dim cho fusion_projection
#     fbank_dim = test_conf['fbank_args'].get('num_mel_bins', 80)
#     if use_fusion:
#         input_dim = fbank_dim + frontend_output_dim
#         fusion_projection = torch.nn.Linear(input_dim, configs['model_args']['feat_dim'])
#         model.add_module("fusion_projection", fusion_projection)
#     else:
#         input_dim = fbank_dim if frontend_type == "fbank" else frontend_output_dim
#     print('Loading checkpoint ...')
#     load_checkpoint(model, model_path)
#     print('Finished !!! Start extracting ...')
#     device = torch.device("cuda")
#     model.to(device).eval()
#     # Kiểm tra fusion_projection sau khi tải checkpoint
#     if use_fusion:
#         if hasattr(model, 'fusion_projection'):
#             print("Fusion_projection layer is loaded into the model.")
#         else:
#             raise AttributeError("Fusion_projection layer is NOT loaded into the model. Check if it was saved in the checkpoint.")
    
#     # Extract embeddings
#     validate_path(embed_ark)
#     embed_ark = os.path.abspath(embed_ark)
#     embed_scp = embed_ark[:-3] + "scp"

#     with torch.no_grad():
#         with kaldiio.WriteHelper('ark,scp:' + embed_ark + "," +
#                                  embed_scp) as writer:
#             for _, batch in tqdm(enumerate(dataloader)):
#                 utts = batch['key']

#                 # Trích xuất cả fbank và frontend features nếu use_fusion=True
#                 if use_fusion:
#                     # Fbank features
#                     fbank_features = batch['feat'].float().to(device)  # (B,T_fbank,F)

#                     # Frontend features (WavLM)
#                     wavs = batch['wav']  # (B,1,W)
#                     wavs = wavs.squeeze(1).float().to(device)  # (B,W)
#                     wavs_len = torch.LongTensor([wavs.shape[1]]).repeat(
#                         wavs.shape[0]).to(device)  # (B)
#                     frontend_features, _ = model.frontend(wavs, wavs_len)  # (B,T_frontend,F')

#                     # Đồng bộ số frame bằng nội suy (giống run_epoch)
#                     T_fbank = fbank_features.shape[1]
#                     T_frontend = frontend_features.shape[1]
#                     if T_fbank != T_frontend:
#                         if T_frontend < T_fbank:
#                             frontend_features = F.interpolate(
#                                 frontend_features.transpose(1, 2),
#                                 size=T_fbank,
#                                 mode='linear',
#                                 align_corners=False
#                             ).transpose(1, 2)
#                         else:
#                             fbank_features = F.interpolate(
#                                 fbank_features.transpose(1, 2),
#                                 size=T_frontend,
#                                 mode='linear',
#                                 align_corners=False
#                             ).transpose(1, 2)

#                     # Fusion: Concatenate và project
#                     features = torch.cat((fbank_features, frontend_features), dim=-1)  # (B,T,F+F')
#                     features = model.fusion_projection(features)  # (B,T,feat_dim)
#                 else:
#                     # Không dùng fusion, chỉ lấy một loại feature
#                     if frontend_type == 'fbank':
#                         features = batch['feat'].float().to(device)  # (B,T,F)
#                     else:  # 's3prl'
#                         wavs = batch['wav']  # (B,1,W)
#                         wavs = wavs.squeeze(1).float().to(device)  # (B,W)
#                         wavs_len = torch.LongTensor([wavs.shape[1]]).repeat(
#                             wavs.shape[0]).to(device)  # (B)
#                         features, _ = model.frontend(wavs, wavs_len)

#                 # Apply cmvn
#                 if test_conf.get('cmvn', True):
#                     features = apply_cmvn(features, **test_conf.get('cmvn_args', {}))

#                 # Spec augmentation (tắt trong test)
#                 if test_conf.get('spec_aug', False):
#                     features = spec_aug(features, **test_conf['spec_aug_args'])

#                 # Forward through model
#                 outputs = model(features)  # embed or (embed_a, embed_b)
#                 embeds = outputs[-1] if isinstance(outputs, tuple) else outputs
#                 embeds = embeds.cpu().detach().numpy()  # (B,F)

#                 for i, utt in enumerate(utts):
#                     embed = embeds[i]
#                     writer(utt, embed)



#CROSS-ATTENTION FUSION MODEL

def extract(config='conf/config.yaml', **kwargs):
    # Parse configs first
    configs = parse_config_or_kwargs(config, **kwargs)

    model_path = configs['model_path']
    embed_ark = configs['embed_ark']
    batch_size = configs.get('batch_size', 1)
    num_workers = configs.get('num_workers', 1)

    # Since the input length is not fixed, we set the built-in cudnn
    # auto-tuner to False
    torch.backends.cudnn.benchmark = False

    # Test configs
    test_conf = copy.deepcopy(configs['dataset_args'])
    test_conf['speed_perturb'] = False
    if 'fbank_args' in test_conf:
        test_conf['fbank_args']['dither'] = 0.0
    test_conf['spec_aug'] = False
    test_conf['shuffle'] = False
    test_conf['aug_prob'] = configs.get('aug_prob', 0.0)
    test_conf['filter'] = False

    # Dataset and DataLoader
    dataset = Dataset(configs['data_type'],
                      configs['data_list'],
                      test_conf,
                      spk2id_dict={},
                      whole_utt=(batch_size == 1),
                      reverb_lmdb_file=configs.get('reverb_data', None),
                      noise_lmdb_file=configs.get('noise_data', None),
                      repeat_dataset=False)
    dataloader = DataLoader(dataset,
                            shuffle=False,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            prefetch_factor=4)

    # Model: frontend (optional) => speaker model
    model = get_speaker_model(configs['model'])(**configs['model_args'])
    frontend_type = test_conf.get('frontend', 'fbank')
    use_fusion = test_conf.get('use_fusion', False)  # Lấy use_fusion từ config
    
    frontend_output_dim = 0
    if frontend_type != 'fbank':
        frontend_args = frontend_type + "_args"
        print('Initializing frontend model (this could take some time) ...')
        frontend = frontend_class_dict[frontend_type](
            **test_conf[frontend_args], sample_rate=test_conf['resample_rate'])
        frontend_output_dim = frontend.output_size()
        model.add_module("frontend", frontend)
    # Tính input_dim cho fusion_projection
    fbank_dim = test_conf['fbank_args'].get('num_mel_bins', 80)
    if use_fusion:
        input_dim = fbank_dim + frontend_output_dim
        fusion_module = CrossAttentionFusion(
            fbank_dim=fbank_dim,
            wavlm_dim=frontend_output_dim,
            d_model=configs['model_args']['feat_dim'],  # Ví dụ: 192
            num_heads=1,
            dropout=0.0
        )
        model.add_module("fusion_module", fusion_module)
    else:
        input_dim = fbank_dim if frontend_type == "fbank" else frontend_output_dim
    print('Loading checkpoint ...')
    load_checkpoint(model, model_path)
    print('Finished !!! Start extracting ...')
    device = torch.device("cuda")
    model.to(device).eval()
    # Kiểm tra fusion_module sau khi tải checkpoint
    if use_fusion:
        if hasattr(model, 'fusion_module'):
            print("Fusion_module layer is loaded into the model.")
        else:
            raise AttributeError("Fusion_module layer is NOT loaded into the model. Check if it was saved in the checkpoint.")
    
    # Extract embeddings
    validate_path(embed_ark)
    embed_ark = os.path.abspath(embed_ark)
    embed_scp = embed_ark[:-3] + "scp"

    with torch.no_grad():
        with kaldiio.WriteHelper('ark,scp:' + embed_ark + "," +
                                 embed_scp) as writer:
            for _, batch in tqdm(enumerate(dataloader)):
                utts = batch['key']

                # Trích xuất cả fbank và frontend features nếu use_fusion=True
                if use_fusion:
                    # Fbank features
                    fbank_features = batch['feat'].float().to(device)  # (B,T_fbank,F)

                    # Frontend features (WavLM)
                    wavs = batch['wav']  # (B,1,W)
                    wavs = wavs.squeeze(1).float().to(device)  # (B,W)
                    wavs_len = torch.LongTensor([wavs.shape[1]]).repeat(
                        wavs.shape[0]).to(device)  # (B)
                    frontend_features, _ = model.frontend(wavs, wavs_len)  # (B,T_frontend,F')

                    # Đồng bộ số frame bằng nội suy (giống run_epoch)
                    T_fbank = fbank_features.shape[1]
                    T_frontend = frontend_features.shape[1]
                    if T_fbank != T_frontend:
                        if T_frontend < T_fbank:
                            frontend_features = F.interpolate(
                                frontend_features.transpose(1, 2),
                                size=T_fbank,
                                mode='linear',
                                align_corners=False
                            ).transpose(1, 2)
                        else:
                            fbank_features = F.interpolate(
                                fbank_features.transpose(1, 2),
                                size=T_frontend,
                                mode='linear',
                                align_corners=False
                            ).transpose(1, 2)

                    # Áp dụng cross-attention
                    features = model.fusion_module(fbank_features, frontend_features)  # (B, T, d_model)
                else:
                    # Không dùng fusion, chỉ lấy một loại feature
                    if frontend_type == 'fbank':
                        features = batch['feat'].float().to(device)  # (B,T,F)
                    else:  # 's3prl'
                        wavs = batch['wav']  # (B,1,W)
                        wavs = wavs.squeeze(1).float().to(device)  # (B,W)
                        wavs_len = torch.LongTensor([wavs.shape[1]]).repeat(
                            wavs.shape[0]).to(device)  # (B)
                        features, _ = model.frontend(wavs, wavs_len)

                # Apply cmvn
                if test_conf.get('cmvn', True):
                    features = apply_cmvn(features, **test_conf.get('cmvn_args', {}))

                # Spec augmentation (tắt trong test)
                if test_conf.get('spec_aug', False):
                    features = spec_aug(features, **test_conf['spec_aug_args'])

                # Forward through model
                outputs = model(features)  # embed or (embed_a, embed_b)
                embeds = outputs[-1] if isinstance(outputs, tuple) else outputs
                embeds = embeds.cpu().detach().numpy()  # (B,F)

                for i, utt in enumerate(utts):
                    embed = embeds[i]
                    writer(utt, embed)


if __name__ == '__main__':
    fire.Fire(extract)
