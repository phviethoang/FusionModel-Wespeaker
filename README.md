# Fusion Model with Wespeaker Framework

## Hướng dẫn sử dụng
- [x] Bước 1: Clone repo về rồi chạy lệnh `pip install wespeaker`
- [x] Bước 2: Chuẩn bị dataset và các file bổ sung hỗ trợ training (Edit các file trong folder `local`)
- [x] Bước 3: Chạy file `run.sh`

## Model CrossAttentionFusion
- [x] config: `conf/ecapa_tdnn_fusion.yaml`
- [x] exp_dir: `exp/Voxceleb1-ECAPA_TDNN_GLOB_c1024-ASTP-emb192-Fusion-1HEAD-WavLM_Large_frozen-num_frms150-aug0.6-spTrue-saFalse-ArcMargin_intertopk_subcenter-SGD-epoch150`
- [x] Các file đã sửa đổi so với framework gốc: `CrossAttentionFusion.py` `train.py` `extract.py` `executor.py` `dataset.py` `processor.py`
