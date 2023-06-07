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
import importlib
env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)

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
    return in_time,pred_last_result,last_pred_idx

def stream_eval(gt_anno_t:list, raw_result:dict, datasetname, args):
    # KF for every sequence
    kf_A = np.eye(8)
    kf_H = np.eye(4)
    kf_H = np.concatenate([kf_H, np.zeros((4,4))],axis=1)
    # print('kf_H.shape', kf_H.shape)
    kf_Q = np.eye(8)
    if datasetname=='esot2s':
        kf_R = np.eye(4)*1
        kf_P_init = np.eye(8)*1
    elif datasetname=='esot500s':
        kf_R = np.eye(4)*1
        kf_P_init = np.eye(8)*1
    
    # kf_R[2:,2:] = np.eye(2)*1 #
    kf_x0 = np.array(raw_result['results_raw'][0]).reshape(4,1)
    kf_x0 = np.concatenate([kf_x0, np.zeros((4,1))],axis=0)
    if args.ref_point == 'left_top':
        kf = PostKalmanFilter(A = kf_A, H = kf_H, Q = kf_Q, R = kf_R , P=kf_P_init, x0=kf_x0)
    elif args.ref_point == 'centor':
        kf_x0[0,0] = kf_x0[0,0] +kf_x0[2,0]/2
        kf_x0[1,0] = kf_x0[1,0] +kf_x0[3,0]/2
        kf = PostKalmanFilter(A = kf_A, H = kf_H, Q = kf_Q, R = kf_R , P=kf_P_init, x0=kf_x0)
    else:
        print("Invalid ref_point")
        return
    pred_final = []
    last_time = gt_anno_t[0][0] # the first gt_t
    last_idx = 0
    ii = 0
    # if datasetname=='esot2s':
    #     step_factor = 20
    # elif datasetname=='esot500s':
    #     step_factor = 100
    if datasetname=='esot2s':
        step_factor = 20
        #
        step_factor = 100
    elif datasetname=='esot500s':
        step_factor = 50

    for line in gt_anno_t:
        gt_t = line[0]
        in_time, pred_label, pred_idx = find_last_pred(gt_t, raw_result,gt_anno_t[0][0])
        # find the gt time near before the in_time. but in the streaming setting, it should be the in_time
        # while ii < len(gt_anno_t) and gt_anno_t[ii][0]*1e6 <= in_time:
        #     ii += 1
        # pred_time = gt_anno_t[ii-1][0]

        pred_time = in_time/1e6

        if pred_time>last_time:
            #dt = hj - hj-1
            dth = (pred_time - last_time)*step_factor   ###
            kf_A[:4,4:] = np.eye(4)*dth
            current_A = kf_A
            if datasetname=='esot2s':
                # update everytime the tracker outputs a result
                for idx in range(last_idx+1, pred_idx+1):
                    dth = (raw_result['in_timestamps'][idx] - raw_result['in_timestamps'][idx-1])/1e6*step_factor   ###
                    kf_A[:4,4:] = np.eye(4)*dth
                    current_A = kf_A
                    kf.Q = kf_Q*dth*dth
                    # kf.Q = kf_Q*1
                    kf.predict(current_A)
                    if args.ref_point == 'left_top':
                        kf.update(np.array(raw_result['results_raw'][idx]).reshape(4,1))
                    elif args.ref_point == 'centor':
                        pred_label = np.array(raw_result['results_raw'][idx]).reshape(4,1)
                        pred_label[0,0] = pred_label[0,0] + pred_label[2,0]/2
                        pred_label[1,0] = pred_label[1,0] + pred_label[3,0]/2
                        kf.update(pred_label)

            elif datasetname=='esot500s':
                kf.Q = kf_Q*dth*dth
                # kf.Q = kf_Q*0.1
                kf.predict(current_A)

                if args.ref_point == 'left_top':
                    kf.update(np.array(pred_label).reshape(4,1))
                elif args.ref_point == 'centor':
                    pred_label = np.array(pred_label).reshape(4,1)
                    pred_label[0,0] = pred_label[0,0] + pred_label[2,0]/2
                    pred_label[1,0] = pred_label[1,0] + pred_label[3,0]/2
                    kf.update(pred_label)
            last_time = pred_time
            last_idx = pred_idx

        if last_time>gt_anno_t[0][0]:
            #dt = i - hj
            dt = (gt_t - last_time)*step_factor
            # dt = 0.1*step_factor
            # dt = 0.016*step_factor
            # dt = 0.009*step_factor
            kf_A[:4,4:] = np.eye(4)*dt
            current_A = kf_A
            if args.ref_point == 'left_top':
                pred_bbox = kf.estimate(current_A)
            elif args.ref_point == 'centor':
                pred_bbox = kf.estimate(current_A)
                pred_bbox[0,0] = pred_bbox[0,0] - pred_bbox[2,0]/2
                pred_bbox[1,0] = pred_bbox[1,0] - pred_bbox[3,0]/2
        else:
            pred_bbox = np.array(raw_result['results_raw'][0]).reshape(4,1)
        pred_bbox = pred_bbox.reshape(-1,4)
        if pred_bbox[0,2]<=0:
            pred_bbox[0,2]=1
        if pred_bbox[0,3]<=0:
            pred_bbox[0,3]=1
        pred_final.append(pred_bbox)
    pred_final = np.stack(pred_final,0).reshape(-1,4)
    return pred_final

def eval_sequence_stream(sequence, tracker, stream_setting, args):

    print("Stream Evaluation: {} {} {}".format(sequence.name, tracker.name, stream_setting.id))
    tracker_name = tracker.name
    param = tracker.parameter_name
    datasetname = sequence.dataset
    gt_anno_t = sequence.ground_truth_t
    save_dir = os.path.join(tracker.results_dir_rt_final,str(stream_setting.id)+'-postKF')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # if os.path.exists(os.path.join(save_dir, sequence.name+'.txt')):
    #     print('Already exists. Skipped. ')
    #     return

    raw_result = pickle.load(open(os.path.join(tracker.results_dir_rt, str(stream_setting.id), sequence.name+'.pkl'), 'rb'))
    assert raw_result['stream_setting'] == stream_setting.id
    pred_final = stream_eval(gt_anno_t, raw_result, datasetname, args)

    
    np.savetxt('{}/{}.txt'.format(save_dir,sequence.name),pred_final,fmt='%d',delimiter='\t')

def run_streaming_eval_kf(experiment_module: str, experiment_name: str, args, debug=0, threads=0, ):
    expr_module = importlib.import_module('pytracking.experiments.{}'.format(experiment_module))
    expr_func = getattr(expr_module, experiment_name)
    trackers, dataset, stream_setting = expr_func()
    print('Running:  {}  {}'.format(experiment_module, experiment_name))
    for seq in dataset:
        for tracker_info in trackers:
                eval_sequence_stream(seq, tracker_info, stream_setting,args)

def main():
    # args = parse_args()

    parser = argparse.ArgumentParser(description='Run tracker on sequence or dataset.')
    parser.add_argument('--experiment_module',default='exp_streaming', type=str, help='Name of experiment module in the experiments/ folder.')
    parser.add_argument('--experiment_name',default='streaming_sotas_s19', type=str, help='Name of the experiment function.')
    parser.add_argument('--ref_point',default='left_top', type=str, help='left_top, or centor')
    args = parser.parse_args()

    run_streaming_eval_kf(args.experiment_module, args.experiment_name,args)
if __name__ == '__main__':
    main()