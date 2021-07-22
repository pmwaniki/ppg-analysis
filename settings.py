import os

Fs=80

home_dir=os.path.expanduser("~")
data_dir=os.path.join(home_dir,"data/ppg")
try:
    base_dir=os.path.dirname(os.path.realpath(__file__))
except:
    pass


weights_dir=os.path.join(data_dir,"results/weights")
checkpoint_dir=os.path.join(home_dir,"data/logs/ppg/")
output_dir=os.path.join(data_dir,"results/outputs")



segment_length=10
segment_slide=2