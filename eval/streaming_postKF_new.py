'''
Streaming postKF-forecaste evaluation
Given real-time tracking outputs, it estimate the intermediate result using Kalman Filter
and pairs them with the ground truth.

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
class PostKalmanFilter(object):
    def __init__(self, A = None, B = None, H = None, Q = None, R = None, P = None, x0 = None):
        if(A is None or H is None):
            raise ValueError("Set proper system dynamics.")
        self.n = A.shape[1]
        self.m = H.shape[0]
        self.A = A
        self.H = H
        self.B = 0 if B is None else B
        self.Q = np.eye(self.n) if Q is None else Q
        self.R = np.eye(self.m) if R is None else R
        self.P = np.eye(self.n) if P is None else P
        self.x = np.zeros((self.n, 1)) if x0 is None else x0

    def predict(self, current_A):
        self.x = np.dot(current_A, self.x)
        self.P = np.dot(np.dot(current_A, self.P), current_A.T) + self.Q
        
    def update(self, z):
        y = z - np.dot(self.H, self.x)
        S = self.R + np.dot(self.H, np.dot(self.P, self.H.T))
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        I = np.eye(self.n)
        self.P = np.dot(np.dot(I - np.dot(K, self.H), self.P), 
        	(I - np.dot(K, self.H)).T) + np.dot(np.dot(K, self.R), K.T)
    def estimate(self,current_A):
        pred_bbox = np.dot(self.H, np.dot(current_A, self.x))
        return pred_bbox

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_root', default='/home/test4/code/EventBenchmark/lib/pytracking/pytracking/tracking_results_rt_raw',type=str,
        help='raw result root')
    parser.add_argument('--target_root', default='/home/test4/code/EventBenchmark/lib/pytracking/pytracking/results_rt_postKF_500_w2ms',type=str,
        help='target result root')
    parser.add_argument('--gt_root',default='/home/test4/code/EventBenchmark/data/EventSOT/EventSOT500/EventSOT500/anno_t/', type=str)
    # parser.add_argument('--fps', type=float, default=30)
    # parser.add_argument('--eta', type=float, default=0, help='eta >= -1')
    parser.add_argument('--overwrite', action='store_true', default=False)

    args = parser.parse_args()
    return args

def find_last_pred(gt_t, pred_raw, t0):
    pred_timestamps = pred_raw['out_timestamps']
    pred_timestamps[0] = 0
    in_timestamps = pred_raw['in_timestamps']
    in_timestamps[0] = t0*1e6
    gt_t = gt_t*1e6
    # print(gt_t, pred_timestamps[-1])
    # assert abs(gt_t - pred_timestamps[-1]) < 100  # time unit:s
    last_pred_idx = np.searchsorted(pred_timestamps, gt_t)-1
    pred_results = pred_raw['results_raw']
    pred_last_result = pred_results[last_pred_idx]
    in_time = in_timestamps[last_pred_idx]

    # print(gt_t, pred_last_time)
    return in_time,pred_last_result

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
                # KF for every sequence
                kf_A = np.eye(12)
                kf_H = np.eye(4)
                kf_H = np.concatenate([kf_H, np.zeros((4,8))],axis=1)
                # print('kf_H.shape', kf_H.shape)
                kf_Q = np.eye(12) # 
                kf_Q[4:8,4:8] = np.eye(4)*1
                kf_R = np.eye(4)*1
                # kf_R[2:,2:] = np.eye(2)*1 #
                kf_P_init = np.eye(12)*1
                kf_x0 = np.array(raw_result['results_raw'][0]).reshape(4,1)
                kf_x0 = np.concatenate([kf_x0, np.zeros((8,1))],axis=0)

                kf = PostKalmanFilter(A = kf_A, H = kf_H, Q = kf_Q, R = kf_R , P=kf_P_init, x0=kf_x0)

                pred_final = []
                last_time = gt_anno_t[0][0] # the first gt_t
                ii = 0

                for line in gt_anno_t:
                    gt_t = line[0]
                    in_time, pred_label = find_last_pred(gt_t, raw_result,gt_anno_t[0][0])
                    while ii < len(gt_anno_t) and gt_anno_t[ii][0]*1e6 <= in_time:
                        ii += 1
                    pred_time = gt_anno_t[ii-1][0]

                    if pred_time>last_time:
                        #dt = hj - hj-1
                        dth = (pred_time - last_time)
                        # kf_A[:4,4:] = np.eye(4)*dth
                        kf_A[:4,4:8] = np.eye(4)*1
                        kf_A[:4,8:] = np.eye(4)*1*1*0.5
                        kf_A[4:8,8:] = np.eye(4)*1
                        current_A = kf_A

                        kf.Q = kf_Q*1

                        kf.predict(current_A)
                        kf.update(np.array(pred_label).reshape(4,1))
                        
                        last_time = pred_time

                    if last_time>gt_anno_t[0][0]:
                        #dt = i - hj
                        dt = (gt_t - last_time)/dth
                        # kf_A[:4,4:] = np.eye(4)*dt
                        kf_A[:4,4:8] = np.eye(4)*dt
                        kf_A[:4,8:] = np.eye(4)*dt*dt*0.5
                        kf_A[4:8,8:] = np.eye(4)*dt
                        current_A = kf_A
                        pred_bbox = kf.estimate(current_A)
                    else:
                        pred_bbox = np.array(raw_result['results_raw'][0]).reshape(4,1)
                    pred_bbox = pred_bbox.reshape(-1,4)
                    if pred_bbox[0,2]<=0:
                        pred_bbox[0,2]=1
                    if pred_bbox[0,3]<=0:
                        pred_bbox[0,3]=1
                    pred_final.append(pred_bbox)
                
                save_dir = '{}/{}/{}'.format(save_path,tracker,param)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                pred_final = np.stack(pred_final,0).reshape(-1,4)
                np.savetxt('{}/{}/{}/{}_500_w2ms.txt'.format(save_path,tracker,param,sequence_name),pred_final,fmt='%d',delimiter='\t')
    
        mismatch = 0
        fps_a=[]
if __name__ == '__main__':
    main()