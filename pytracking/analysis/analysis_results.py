
import os
import sys
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [14, 8]

sys.path.append('/home/test4/code/EventBenchmark/lib/pytracking')
from pytracking.analysis.plot_results import plot_results, print_results, print_per_sequence_results
from pytracking.evaluation import get_dataset, trackerlist

# Realtime
dataset_name = 'esot2_5_20'
trackers = []
trackers.extend(trackerlist(name='atom', parameter_name='default',
                            run_ids=0, display_name='atom'))
trackers.extend(trackerlist(name='dimp', parameter_name='dimp18', 
                            run_ids=0, display_name='dimp18'))
trackers.extend(trackerlist(name='dimp', parameter_name='prdimp18', 
                            run_ids=0, display_name='prdimp18'))
trackers.extend(trackerlist(name='kys', parameter_name='default', 
                            run_ids=0, display_name='kys'))
trackers.extend(trackerlist(name='rts', parameter_name='rts50', 
                            run_ids=0, display_name='rts50'))
trackers.extend(trackerlist(name='keep_track', parameter_name='default', 
                            run_ids=0, display_name='keep_track'))
# trackers.extend(trackerlist(name='atom', parameter_name='esot500',
#                             run_ids=0, display_name='atom'))
dataset = get_dataset(dataset_name)
print_results(trackers, dataset, dataset_name, merge_results=False, plot_types=('success', 'prec', 'norm_prec'),force_evaluation=True, exclude_invalid_frames=True)