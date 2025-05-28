# Copyright (c) 2022 Zhengyang Chen (chenzhengyang117@gmail.com)
#               2022 Hongji Wang (jijijiang77@gmail.com)
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

import os

import fire
import numpy as np

from wespeaker.utils.score_metrics import (compute_pmiss_pfa_rbst, compute_eer,
                                           compute_c_norm)


def compute_metrics(scores_file, p_target=0.01, c_miss=1, c_fa=1):
    scores = []
    labels = []

    with open(scores_file) as readlines:
        for line in readlines:
            tokens = line.strip().split()
            # assert len(tokens) == 4
            scores.append(float(tokens[2]))
            labels.append(tokens[3] == 'target')

    scores = np.hstack(scores)
    labels = np.hstack(labels)

    fnr, fpr = compute_pmiss_pfa_rbst(scores, labels)
    eer, thres = compute_eer(fnr, fpr, scores)

    min_dcf = compute_c_norm(fnr,
                             fpr,
                             p_target=p_target,
                             c_miss=c_miss,
                             c_fa=c_fa)
    # Tính precision và recall tại ngưỡng EER
    sorted_ndx = np.argsort(scores)
    sorted_scores = scores[sorted_ndx]
    sorted_labels = labels[sorted_ndx]
    threshold_idx = np.argmin(np.abs(fnr - fpr))  # Tìm chỉ số ngưỡng EER
    threshold = sorted_scores[threshold_idx]

    # Tính TP, FP, FN
    total_target = np.sum(labels == 1)
    total_nontarget = np.sum(labels == 0)
    fnr_at_eer = fnr[threshold_idx]  # FNR tại EER
    fpr_at_eer = fpr[threshold_idx]  # FPR tại EER
    tpr_at_eer = 1 - fnr_at_eer  # TPR = 1 - FNR
    fn = fnr[threshold_idx] * total_target  # False Negatives
    fp = fpr[threshold_idx] * total_nontarget  # False Positives
    tp = total_target - fn  # True Positives

    # Tính precision và recall
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

 
    print("---- {} -----".format(os.path.basename(scores_file)))
    print("EER = {0:.3f}%".format(100 * eer))
    print("Threshold at EER: {0:.3f}".format(thres))
    print("FNR at EER: {0:.3f}".format(fnr_at_eer*100))
    print("FPR at EER: {0:.3f}".format(fpr_at_eer*100))
    print("TPR at EER: {0:.3f}".format(tpr_at_eer*100))
    print("Precision at EER: {0:.3f}".format(precision*100))
    print("Recall at EER: {0:.3f}".format(recall*100))
    print("minDCF (p_target:{} c_miss:{} c_fa:{}) = {:.3f}".format(p_target, c_miss, c_fa, min_dcf))

def main(p_target=0.01, c_miss=1, c_fa=1, *scores_files):
    for scores_file in scores_files:
        compute_metrics(scores_file, p_target, c_miss, c_fa)


if __name__ == "__main__":
    fire.Fire(main)
