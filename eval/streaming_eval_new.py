'''
Streaming evaluation
Given real-time tracking outputs,
it pairs them with the ground truth.

Note that this script does not need to run in real-time
'''

import argparse, pickle
from os.path import join, isfile
import numpy as np
from tqdm import tqdm
import sys
import os

# the line below is for running in both the current directory 
# and the repo's root directory


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_root', default='/home/test4/code/EventBenchmark/lib/pytracking/pytracking/tracking_results_rt_raw',type=str,
        help='raw result root')
    parser.add_argument('--target_root', default='/home/test4/code/EventBenchmark/lib/pytracking/pytracking/tracking_results_rt_final',type=str,
        help='target result root')
    parser.add_argument('--gt_root',default='/home/test4/code/EventBenchmark/data/EventSOT/EventSOT500/EventSOT500/anno_t/', type=str)
    # parser.add_argument('--fps', type=float, default=30)
    # parser.add_argument('--eta', type=float, default=0, help='eta >= -1')
    parser.add_argument('--overwrite', action='store_true', default=False)

    args = parser.parse_args()
    return args

def find_last_pred(gt_t, pred_raw):
    pred_timestamps = pred_raw['out_timestamps']
    pred_timestamps[0] = 0
    gt_t = gt_t*1e6
    # print(gt_t, pred_timestamps[-1])
    # assert abs(gt_t - pred_timestamps[-1]) < 100  # time unit:s
    last_pred_idx = np.searchsorted(pred_timestamps, gt_t)-1
    pred_results = pred_raw['results_raw']
    pred_last_result = pred_results[last_pred_idx]
    pred_last_time = pred_timestamps[last_pred_idx]
    # print(gt_t, pred_last_time)
    return pred_last_result

def main():
    args = parse_args()
    trackers = os.listdir(args.raw_root)
    gt_path = args.gt_root
    raw_result_path = args.raw_root
    save_path = args.target_root
    
    gt_list = os.listdir(gt_path)
    gt_list = [os.path.join(gt_path, i) for i in os.listdir(gt_path) if i.endswith('.txt')]
    for tracker in trackers:
        for param in os.listdir(os.path.join(raw_result_path, tracker)):
            raw_result_list_dir = os.path.join(raw_result_path,tracker,param)
            raw_result_list = os.listdir(raw_result_list_dir)
            for sequence in tqdm(raw_result_list):
                sequence_name = sequence.split('.')[0]
                raw_result = pickle.load(open(os.path.join(raw_result_list_dir, sequence), 'rb'))
                gt_anno_t_path = os.path.join(gt_path, sequence_name+'.txt')
                gt_anno_t = np.loadtxt(gt_anno_t_path, delimiter=' ')
                pred_final = []
                for line in gt_anno_t:
                    gt_t = line[0]
                    pred_label = find_last_pred(gt_t, raw_result)
                    pred_bbox = pred_label
                    pred_final.append(pred_bbox)
                save_dir = '{}/{}/{}'.format(save_path,tracker,param)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                np.savetxt('{}/{}/{}/{}.txt'.format(save_path,tracker,param,sequence_name),pred_final,fmt='%d',delimiter='\t')
    
        mismatch = 0
        fps_a=[]
if __name__ == '__main__':
    main()
