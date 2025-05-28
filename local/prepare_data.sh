#!/bin/bash

# Copyright (c) 2022 Hongji Wang (jijijiang77@gmail.com)
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

stage=-1
stop_stage=-1
data=/home4/datpt/hoangpv

. tools/parse_options.sh || exit 1

data=`realpath ${data}`
download_dir=${data}/download_data
rawdata_dir=${data}/raw_data
noisedata_dir=/home3/thanhpv/speaker_verification/slt/ECAPA-TDNN/voxceleb_trainer/data_augment

#Stage 1->3: Không thực hiện
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Download musan.tar.gz, rirs_noises.zip"
  echo "This may take a long time. Thus we recommand you to download all archives above in your own way first."

  ./local/download_data.sh --download_dir ${download_dir}
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "Decompress all archives ..."
  echo "This could take some time ..."

  # for archive in musan.tar.gz rirs_noises.zip; do
  #   [ ! -f ${download_dir}/$archive ] && echo "Archive $archive not exists !!!" && exit 1
  # done
  # [ ! -d ${rawdata_dir} ] && mkdir -p ${rawdata_dir}

  # if [ ! -d ${rawdata_dir}/musan ]; then
  #   tar -xzvf ${download_dir}/musan.tar.gz -C ${rawdata_dir}
  # fi

  # if [ ! -d ${rawdata_dir}/RIRS_NOISES ]; then
  #   unzip ${download_dir}/rirs_noises.zip -d ${rawdata_dir}
  # fi

  if [ ! -d ${rawdata_dir}/voxceleb1 ]; then
    mkdir -p ${rawdata_dir}/voxceleb1/test # ${rawdata_dir}/voxceleb1/dev
    unzip ${download_dir}/vox1_test_wav.zip -d ${rawdata_dir}/voxceleb1/test
    # unzip ${download_dir}/vox1_dev_wav.zip -d ${rawdata_dir}/voxceleb1/dev
  fi

  # if [ ! -d ${rawdata_dir}/voxceleb2_m4a ]; then
  #   mkdir -p ${rawdata_dir}/voxceleb2_m4a
  #   unzip ${download_dir}/vox2_aac.zip -d ${rawdata_dir}/voxceleb2_m4a
  # fi

  echo "Decompress success !!!"
fi

#Không cần convert vì vn-celeb đã ở .wav sẵn
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3]; then
  echo "Convert voxceleb2 wav format from m4a to wav using ffmpeg."
  echo "This could also take some time ..."

  if [ ! -d ${rawdata_dir}/voxceleb2_wav ]; then
    ./local/m4a2wav.pl ${rawdata_dir}/voxceleb2_m4a dev ${rawdata_dir}/voxceleb2_wav
    # Here we use 8 parallel jobs
    cat ${rawdata_dir}/voxceleb2_wav/dev/m4a2wav_dev.sh | xargs -P 8 -i sh -c "{}"
  fi

  echo "Convert m4a2wav success !!!"
fi

# Bắt đầu tạo file bổ sung ở bước Stage 4 này
# Chuẩn bị data vnceleb
# if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
#   echo "Prepare wav.scp for each dataset ..."
#   export LC_ALL=C # kaldi config

#   mkdir -p  ${data}/vn-celeb ${data}/vn-celeb_test
#   # musan
#   find ${noisedata_dir}/musan_split -name "*.wav" | awk -F"/" '{print $(NF-1)"/"$NF,$0}' >${data}/musan/wav.scp
#   # rirs
#   find ${noisedata_dir}/RIRS_NOISES/simulated_rirs -name "*.wav" | awk -F"/" '{print $(NF-1)"/"$NF,$0}' >${data}/rirs/wav.scp
#   # vn-celeb_test
#   find /home4/thanhpv/vn-celeb/data_test -name "*.wav" | awk -F"/" '{print $(NF-1)"/"$NF,$0}' | sort >${data}/vn-celeb_test/wav.scp
#   awk '{print $1}' ${data}/vn-celeb_test/wav.scp | awk -F "/" '{print $0,$1}' >${data}/vn-celeb_test/utt2spk
#   ./tools/utt2spk_to_spk2utt.pl ${data}/vn-celeb_test/utt2spk >${data}/vn-celeb_test/spk2utt
#   if [ ! -d ${data}/vn-celeb_test/trials ]; then
#     echo "Download trials for vn-celeb_test ..."
#     mkdir -p ${data}/vn-celeb_test/trials

#     cp ${data}/vietnam-celeb-t.txt ${data}/vn-celeb_test/trials/vietnam-celeb-t.txt
#     cp ${data}/vietnam-celeb-h-cleaned.txt ${data}/vn-celeb_test/trials/vietnam-celeb-h-cleaned.txt
#     cp ${data}/vietnam-celeb-e-cleaned.txt ${data}/vn-celeb_test/trials/vietnam-celeb-e-cleaned.txt
    
#     # transform them into kaldi trial format
#     # awk '{if($1==0)label="nontarget";else{label="target"}; print $2,$3,label}' ${data}/vn-celeb/trials/vietnam-celeb-t.txt >${data}/vn-celeb/trials/vietnam-celeb-t.kaldi
#     awk '{if($1==0)label="nontarget";else{label="target"}; print $2,$3,label}' ${data}/vn-celeb_test/trials/vietnam-celeb-h-cleaned.txt >${data}/vn-celeb_test/trials/vietnam-celeb-h-cleaned.kaldi
#     awk '{if($1==0)label="nontarget";else{label="target"}; print $2,$3,label}' ${data}/vn-celeb_test/trials/vietnam-celeb-e-cleaned.txt >${data}/vn-celeb_test/trials/vietnam-celeb-e-cleaned.kaldi
#   fi
#   # vn-celeb
#   find /home4/thanhpv/vn-celeb/data -name "*.wav" | awk -F"/" '{print $(NF-1)"/"$NF,$0}' | sort >${data}/vn-celeb/wav.scp
#   awk '{print $1}' ${data}/vn-celeb/wav.scp | awk -F "/" '{print $0,$1}' >${data}/vn-celeb/utt2spk
#   ./tools/utt2spk_to_spk2utt.pl ${data}/vn-celeb/utt2spk >${data}/vn-celeb/spk2utt
  

#   echo "Success !!!"
# fi

#Chuẩn bị data voxceleb
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  echo "Prepare wav.scp for each dataset ..."
  export LC_ALL=C # kaldi config

  mkdir -p  ${data}/voxceleb1 ${data}/voxceleb1_test
  # # musan
  # find ${noisedata_dir}/musan_split -name "*.wav" | awk -F"/" '{print $(NF-1)"/"$NF,$0}' >${data}/musan/wav.scp
  # # rirs
  # find ${noisedata_dir}/RIRS_NOISES/simulated_rirs -name "*.wav" | awk -F"/" '{print $(NF-1)"/"$NF,$0}' >${data}/rirs/wav.scp
  # voxceleb1_test
  find /home4/datpt/hoangpv/raw_data/voxceleb1/test/wav -name "*.wav" | awk -F"/" '{print $(NF-2)"/"$(NF-1)"/"$NF,$0}' | sort >${data}/voxceleb1_test/wav.scp
  awk '{print $1}' ${data}/voxceleb1_test/wav.scp | awk -F "/" '{print $0,$1}' >${data}/voxceleb1_test/utt2spk
  ./tools/utt2spk_to_spk2utt.pl ${data}/voxceleb1_test/utt2spk >${data}/voxceleb1_test/spk2utt
  if [ ! -d ${data}/voxceleb1_test/trials ]; then
    echo "Download trials for voxceleb1_test ..."
    mkdir -p ${data}/voxceleb1_test/trials

    cp ${data}/voxceleb1-O.txt ${data}/voxceleb1_test/trials/voxceleb1-O.txt
    
    # transform them into kaldi trial format
    awk '{if($1==0)label="nontarget";else{label="target"}; print $2,$3,label}' ${data}/voxceleb1_test/trials/voxceleb1-O.txt >${data}/voxceleb1_test/trials/voxceleb1-O.kaldi
  fi
  # voxceleb1
  find /home4/vuhl/hoanxt/SASVdata/english_data/vox_celeb/wav -name "*.wav" | awk -F"/" '{print $(NF-2)"/"$(NF-1)"/"$NF,$0}' | sort >${data}/voxceleb1/wav.scp
  awk '{print $1}' ${data}/voxceleb1/wav.scp | awk -F "/" '{print $0,$1}' >${data}/voxceleb1/utt2spk
  ./tools/utt2spk_to_spk2utt.pl ${data}/voxceleb1/utt2spk >${data}/voxceleb1/spk2utt
  echo "Success !!!"
fi