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

import tableprint as tp
import torch.nn.functional as F
import torch
import torchnet as tnt
from wespeaker.dataset.dataset_utils import apply_cmvn, spec_aug



#CROSS ATTENTION FUSION MODEL
def run_epoch(dataloader, epoch_iter, model, criterion, optimizer, scheduler,
              margin_scheduler, epoch, logger, scaler, device, configs):
    model.train()
    loss_meter = tnt.meter.AverageValueMeter()
    acc_meter = tnt.meter.ClassErrorMeter(accuracy=True)

    frontend_type = configs['dataset_args'].get('frontend', 'fbank')
    use_fusion = configs['dataset_args'].get('use_fusion', False)
    
    for i, batch in enumerate(dataloader):
        cur_iter = (epoch - 1) * epoch_iter + i
        scheduler.step(cur_iter)
        margin_scheduler.step(cur_iter)

        utts = batch['key']
        targets = batch['label']
        targets = targets.long().to(device)  # (B)

        # Lấy cả fbank và frontend model features
        fbank_features = batch['feat'].float().to(device)  # (B,T_fbank,F)

        if frontend_type != 'fbank':
            wavs = batch['wav']  # (B,1,W)
            wavs = wavs.squeeze(1).float().to(device)  # (B,W)
            wavs_len = torch.LongTensor([wavs.shape[1]]).repeat(
                wavs.shape[0]).to(device)  # (B)
            with torch.amp.autocast("cuda", enabled=configs['enable_amp']):
                frontend_features, _ = model.module.frontend(wavs, wavs_len)  # (B,T_frontend,F')
        else:
            frontend_features = torch.zeros_like(fbank_features)  # (B,T_fbank,F')

        # Đồng bộ số frame (T) bằng nội suy
        if use_fusion and frontend_type != 'fbank':
            T_fbank = fbank_features.shape[1]
            T_frontend = frontend_features.shape[1]
            if T_fbank != T_frontend:
                # Nội suy frontend_features để khớp với T_fbank
                if T_frontend < T_fbank:
                    frontend_features = F.interpolate(
                        frontend_features.transpose(1, 2),  # (B,F',T_frontend)
                        size=T_fbank,
                        mode='linear',  # Nội suy tuyến tính
                        align_corners=False
                    ).transpose(1, 2)  # (B,T_fbank,F')
                else:
                    # Nội suy fbank_features để khớp với T_frontend
                    fbank_features = F.interpolate(
                        fbank_features.transpose(1, 2),  # (B,F,T_fbank)
                        size=T_frontend,
                        mode='linear',
                        align_corners=False
                    ).transpose(1, 2)  # (B,T_frontend,F)

       # Fusion: Sử dụng cross-attention hai chiều thay vì concatenate và project
        if use_fusion:
            with torch.amp.autocast("cuda", enabled=configs['enable_amp']):
                features = model.module.fusion_module(fbank_features, frontend_features)  # (B,T,d_model)
        else:
            features = fbank_features if frontend_type == 'fbank' else frontend_features
        # Kiểm tra NaN/Inf trong features
        if torch.isnan(features).any() or torch.isinf(features).any():
            logger.warning(f"NaN/Inf detected in features at batch {i+1}, utts: {utts}")
            continue

        with torch.amp.autocast("cuda", enabled=configs['enable_amp']):
            if configs['dataset_args'].get('cmvn', True):
                features = apply_cmvn(
                    features, **configs['dataset_args'].get('cmvn_args', {}))
            if configs['dataset_args'].get('spec_aug', False):
                features = spec_aug(features,
                                    **configs['dataset_args']['spec_aug_args'])

            outputs = model(features)  # (embed_a,embed_b) in most cases
            embeds = outputs[-1] if isinstance(outputs, tuple) else outputs
            outputs = model.module.projection(embeds, targets)
            if isinstance(outputs, tuple):
                outputs, loss = outputs
            else:
                loss = criterion(outputs, targets)
        # Kiểm tra NaN/Inf trong loss
        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning(f"NaN/Inf detected in loss at batch {i+1}, utts: {utts}")
            continue
        
        loss_meter.add(loss.item())
        acc_meter.add(outputs.cpu().detach().numpy(), targets.cpu().numpy())

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if (i + 1) % configs['log_batch_interval'] == 0:
            logger.info(
                tp.row((epoch, i + 1, scheduler.get_lr(),
                        margin_scheduler.get_margin()) +
                       (loss_meter.value()[0], acc_meter.value()[0]),
                       width=10,
                       style='grid'))

        if (i + 1) == epoch_iter:
            break

    logger.info(
        tp.row(
            (epoch, i + 1, scheduler.get_lr(), margin_scheduler.get_margin()) +
            (loss_meter.value()[0], acc_meter.value()[0]),
            width=10,
            style='grid'))





# # FUSION MODEL
# def run_epoch(dataloader, epoch_iter, model, criterion, optimizer, scheduler,
#               margin_scheduler, epoch, logger, scaler, device, configs):
#     model.train()
#     loss_meter = tnt.meter.AverageValueMeter()
#     acc_meter = tnt.meter.ClassErrorMeter(accuracy=True)

#     frontend_type = configs['dataset_args'].get('frontend', 'fbank')
#     use_fusion = configs['dataset_args'].get('use_fusion', False)
    
#     for i, batch in enumerate(dataloader):
#         cur_iter = (epoch - 1) * epoch_iter + i
#         scheduler.step(cur_iter)
#         margin_scheduler.step(cur_iter)

#         utts = batch['key']
#         targets = batch['label']
#         targets = targets.long().to(device)  # (B)

#         # Lấy cả fbank và frontend model features
#         fbank_features = batch['feat'].float().to(device)  # (B,T_fbank,F)

#         if frontend_type != 'fbank':
#             wavs = batch['wav']  # (B,1,W)
#             wavs = wavs.squeeze(1).float().to(device)  # (B,W)
#             wavs_len = torch.LongTensor([wavs.shape[1]]).repeat(
#                 wavs.shape[0]).to(device)  # (B)
#             with torch.amp.autocast("cuda", enabled=configs['enable_amp']):
#                 frontend_features, _ = model.module.frontend(wavs, wavs_len)  # (B,T_frontend,F')
#         else:
#             frontend_features = torch.zeros_like(fbank_features)  # (B,T_fbank,F')

#         # Đồng bộ số frame (T) bằng nội suy
#         if use_fusion and frontend_type != 'fbank':
#             T_fbank = fbank_features.shape[1]
#             T_frontend = frontend_features.shape[1]
#             if T_fbank != T_frontend:
#                 # Nội suy frontend_features để khớp với T_fbank
#                 if T_frontend < T_fbank:
#                     frontend_features = F.interpolate(
#                         frontend_features.transpose(1, 2),  # (B,F',T_frontend)
#                         size=T_fbank,
#                         mode='linear',  # Nội suy tuyến tính
#                         align_corners=False
#                     ).transpose(1, 2)  # (B,T_fbank,F')
#                 else:
#                     # Nội suy fbank_features để khớp với T_frontend
#                     fbank_features = F.interpolate(
#                         fbank_features.transpose(1, 2),  # (B,F,T_fbank)
#                         size=T_frontend,
#                         mode='linear',
#                         align_corners=False
#                     ).transpose(1, 2)  # (B,T_frontend,F)

#         # Fusion: Concatenate và project
#         if use_fusion:
#             features = torch.cat((fbank_features, frontend_features), dim=-1)  # (B,T,F+F')
#             with torch.amp.autocast("cuda", enabled=configs['enable_amp']):
#                 features = model.module.fusion_projection(features)  # (B,T,feat_dim)
#         else:
#             features = fbank_features if frontend_type == 'fbank' else frontend_features

#         with torch.amp.autocast("cuda", enabled=configs['enable_amp']):
#             if configs['dataset_args'].get('cmvn', True):
#                 features = apply_cmvn(
#                     features, **configs['dataset_args'].get('cmvn_args', {}))
#             if configs['dataset_args'].get('spec_aug', False):
#                 features = spec_aug(features,
#                                     **configs['dataset_args']['spec_aug_args'])

#             outputs = model(features)  # (embed_a,embed_b) in most cases
#             embeds = outputs[-1] if isinstance(outputs, tuple) else outputs
#             outputs = model.module.projection(embeds, targets)
#             if isinstance(outputs, tuple):
#                 outputs, loss = outputs
#             else:
#                 loss = criterion(outputs, targets)

#         loss_meter.add(loss.item())
#         acc_meter.add(outputs.cpu().detach().numpy(), targets.cpu().numpy())

#         optimizer.zero_grad()
#         scaler.scale(loss).backward()
#         scaler.step(optimizer)
#         scaler.update()

#         if (i + 1) % configs['log_batch_interval'] == 0:
#             logger.info(
#                 tp.row((epoch, i + 1, scheduler.get_lr(),
#                         margin_scheduler.get_margin()) +
#                        (loss_meter.value()[0], acc_meter.value()[0]),
#                        width=10,
#                        style='grid'))

#         if (i + 1) == epoch_iter:
#             break

#     logger.info(
#         tp.row(
#             (epoch, i + 1, scheduler.get_lr(), margin_scheduler.get_margin()) +
#             (loss_meter.value()[0], acc_meter.value()[0]),
#             width=10,
#             style='grid'))



# #BASE MODEL
# def run_epoch(dataloader, epoch_iter, model, criterion, optimizer, scheduler,
#               margin_scheduler, epoch, logger, scaler, device, configs):
#     model.train()
#     # By default use average pooling
#     loss_meter = tnt.meter.AverageValueMeter()
#     acc_meter = tnt.meter.ClassErrorMeter(accuracy=True)

#     frontend_type = configs['dataset_args'].get('frontend', 'fbank')
#     for i, batch in enumerate(dataloader):
#         cur_iter = (epoch - 1) * epoch_iter + i
#         scheduler.step(cur_iter)
#         margin_scheduler.step(cur_iter)

#         utts = batch['key']
#         targets = batch['label']
#         targets = targets.long().to(device)  # (B)
#         if frontend_type == 'fbank':
#             features = batch['feat']  # (B,T,F)
#             features = features.float().to(device)
#         else:  # 's3prl'
#             wavs = batch['wav']  # (B,1,W)
#             wavs = wavs.squeeze(1).float().to(device)  # (B,W)
#             wavs_len = torch.LongTensor([wavs.shape[1]]).repeat(
#                 wavs.shape[0]).to(device)  # (B)
#             with torch.amp.autocast("cuda",enabled=configs['enable_amp']):
#                 features, _ = model.module.frontend(wavs, wavs_len)

#         with torch.amp.autocast("cuda",enabled=configs['enable_amp']):
#             # apply cmvn
#             if configs['dataset_args'].get('cmvn', True):
#                 features = apply_cmvn(
#                     features, **configs['dataset_args'].get('cmvn_args', {}))
#             # spec augmentation
#             if configs['dataset_args'].get('spec_aug', False):
#                 features = spec_aug(features,
#                                     **configs['dataset_args']['spec_aug_args'])

#             outputs = model(features)  # (embed_a,embed_b) in most cases
#             embeds = outputs[-1] if isinstance(outputs, tuple) else outputs
#             outputs = model.module.projection(embeds, targets)
#             if isinstance(outputs, tuple):
#                 outputs, loss = outputs
#             else:
#                 loss = criterion(outputs, targets)

#         # loss, acc
#         loss_meter.add(loss.item())
#         acc_meter.add(outputs.cpu().detach().numpy(), targets.cpu().numpy())

#         # updata the model
#         optimizer.zero_grad()
#         # scaler does nothing here if enable_amp=False
#         scaler.scale(loss).backward()
#         scaler.step(optimizer)
#         scaler.update()

#         # log
#         if (i + 1) % configs['log_batch_interval'] == 0:
#             logger.info(
#                 tp.row((epoch, i + 1, scheduler.get_lr(),
#                         margin_scheduler.get_margin()) +
#                        (loss_meter.value()[0], acc_meter.value()[0]),
#                        width=10,
#                        style='grid'))

#         if (i + 1) == epoch_iter:
#             break

#     logger.info(
#         tp.row(
#             (epoch, i + 1, scheduler.get_lr(), margin_scheduler.get_margin()) +
#             (loss_meter.value()[0], acc_meter.value()[0]),
#             width=10,
#             style='grid'))
