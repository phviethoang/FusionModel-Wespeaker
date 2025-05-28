#!/bin/bash

# Copyright 2022 Hongji Wang (jijijiang77@gmail.com)
#           2022 Chengdong Liang (liangchengdong@mail.nwpu.edu.cn)

. ./path.sh || exit 1

# multi-node + multi-gpus:
#   bash run.sh --stage 3 --stop-stage 3 --HOST_NODE_ADDR "xxx.xxx.xxx.xxx:port" --num_nodes num_node

stage=-1
stop_stage=-1

HOST_NODE_ADDR="localhost:29600"
num_nodes=1
job_id=2026

data=/home4/datpt/hoangpv
data_type="shard"  # shard/raw

config=conf/ecapa_tdnn_fusion.yaml
exp_dir=exp/Voxceleb1-ECAPA_TDNN_GLOB_c1024-ASTP-emb192-Fusion-1HEAD-WavLM_Large_frozen-num_frms150-aug0.6-spTrue-saFalse-ArcMargin_intertopk_subcenter-SGD-epoch150
gpus="[0]"
num_avg=10
checkpoint=${checkpoint:-""} # Nếu checkpoint chưa được gán, gán giá trị rỗng
# checkpoint=${exp_dir}/models/model_20.pt #Trường hợp đã có checkpoint

trials="voxceleb1-O.kaldi" #vietnam-celeb-e-cleaned.kaldi vietnam-celeb-h-cleaned.kaldi"
score_norm_method="asnorm"  # asnorm/snorm
top_n=300


. tools/parse_options.sh || exit 1

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Prepare datasets ..."
  ./local/prepare_data.sh --stage 4 --stop_stage 4 --data ${data}
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "Covert train and test data to ${data_type}..."
  for dset in voxceleb1 voxceleb1_test; do
    if [ $data_type == "shard" ]; then
      python tools/make_shard_list.py --num_utts_per_shard 1000 \
          --num_threads 16 \
          --prefix shards \
          --shuffle \
          ${data}/$dset/wav.scp ${data}/$dset/utt2spk \
          ${data}/$dset/shards ${data}/$dset/shard.list
    else
      python tools/make_raw_list.py ${data}/$dset/wav.scp \
          ${data}/$dset/utt2spk ${data}/$dset/raw.list
    fi
  done
  # # Convert all musan data to LMDB
  # python tools/make_lmdb.py ${data}/musan/wav.scp ${data}/musan/lmdb
  # # Convert all rirs data to LMDB
  # python tools/make_lmdb.py ${data}/rirs/wav.scp ${data}/rirs/lmdb
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "Start training ..."
  # mkdir -p ${exp_dir}/models
  # cp /home4/datpt/hoangpv/wespeaker/examples/vnceleb/v2/exp/Voxceleb1-ECAPA_TDNN_GLOB_c1024-ASTP-emb192-Fusion_fbank80_WavLM_Large_frozen-num_frms150-aug0.6-spTrue-saFalse-ArcMargin_intertopk_subcenter-SGD-epoch150/models/avg_model.pt ${exp_dir}/models/model_0.pt
  num_gpus=$(echo $gpus | awk -F ',' '{print NF}')
  echo "$0: num_nodes is $num_nodes, proc_per_node is $num_gpus"
  nice -n 10 torchrun --nnodes=$num_nodes --nproc_per_node=$num_gpus \
           --rdzv_id=$job_id --rdzv_backend="c10d" --rdzv_endpoint=$HOST_NODE_ADDR \
    wespeaker/bin/train.py --config $config \
      --exp_dir ${exp_dir} \
      --gpus $gpus \
      --num_avg ${num_avg} \
      --data_type "${data_type}" \
      --train_data ${data}/voxceleb1/${data_type}.list \
      --train_label ${data}/voxceleb1/utt2spk \
      --reverb_data ${data}/rirs/lmdb \
      --noise_data ${data}/musan/lmdb \
      ${checkpoint:+--checkpoint "$checkpoint"}
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  echo "Do model average ..."
  avg_model=$exp_dir/models/avg_model.pt
  python wespeaker/bin/average_model.py \
    --dst_model $avg_model \
    --src_path $exp_dir/models \
    --num ${num_avg}

  model_path=$avg_model

  echo "Extract embeddings ..."
  local/extract_vox.sh \
    --exp_dir $exp_dir --model_path $model_path \
    --nj 2 --gpus $gpus --data_type $data_type --data ${data}
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  echo "Score ..."
  local/score.sh \
    --stage 1 --stop-stage 2 \
    --data ${data} \
    --exp_dir $exp_dir \
    --trials "$trials"
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  echo "Score norm ..."
  local/score_norm.sh \
    --stage 1 --stop-stage 3 \
    --score_norm_method $score_norm_method \
    --cohort_set voxceleb1 \
    --top_n $top_n \
    --data ${data} \
    --exp_dir $exp_dir \
    --trials "$trials"
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
  echo "Score calibration ..."
  local/score_calibration.sh \
    --stage 1 --stop-stage 5 \
    --score_norm_method $score_norm_method \
    --calibration_trial "voxceleb1_cali.kaldi" \
    --cohort_set voxceleb1 \
    --top_n $top_n \
    --data ${data} \
    --exp_dir $exp_dir \
    --trials "$trials"
fi

if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
  echo "Export the best model ..."
  python wespeaker/bin/export_jit.py \
    --config $exp_dir/config.yaml \
    --checkpoint $exp_dir/models/avg_model.pt \
    --output_file $exp_dir/models/final.zip
fi

if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ]; then
  echo "Large margin fine-tuning ..."
  lm_exp_dir=${exp_dir}-LM
  mkdir -p ${lm_exp_dir}/models
  # Use the pre-trained average model to initialize the LM training
  cp ${exp_dir}/models/avg_model.pt ${lm_exp_dir}/models/model_0.pt
  bash run_gemini.sh --stage 3 --stop_stage 3 \
      --data ${data} \
      --data_type ${data_type} \
      --config ${lm_config} \
      --exp_dir ${lm_exp_dir} \
      --gpus $gpus \
      --num_avg 1 \
      --checkpoint ${lm_exp_dir}/models/model_0.pt \
      --trials "$trials" \
      --score_norm_method ${score_norm_method} \
      --top_n ${top_n}
fi
