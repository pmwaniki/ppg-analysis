import os
from enum import Enum

Fs=80
myhost = os.uname()[1]

# mongo_host= "10.0.8.140" if myhost == "kenbo-pc0060" else "127.0.0.1"
mongo_host="127.0.0.1"
mongo_user='root'
mongo_password="example"
data_dir="/home/pmwaniki/data/ppg"
try:
    base_dir=os.path.dirname(os.path.realpath(__file__))
except:
    pass

if (os.uname()[1]=='kenbo-cen07') | (os.uname()[1]=='kenbo-cen05'):
    data_dir="/home/local/KWTRP/pmwaniki/data/ppg"
weights_dir=os.path.join(data_dir,"results/weights")
checkpoint_dir="/home/pmwaniki/data/logs/ppg/"
# if os.uname()[1]=="kenbo-pc0177-d":
#     checkpoint_dir="/home/pmwaniki/checkpoints/contrastive"
# elif os.uname()[1]=="kenbo-pc0139":
#     checkpoint_dir="/home/pmwaniki/checkpoints/contrastive"

class Mongo(Enum):
    host='127.0.0.1'
    user=mongo_user
    password=mongo_password


segment_length=10
segment_slide=2