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

clinical_predictors=[
'Weight-Height Z score',
'MUAC (Mid-upper arm circumference) cm',
       'Temperature (degrees celsius)',
       'Respiratory rate- RR (per minute)', 'Oxygen saturation',
       'Heart rate(HR) ', 'Length of illness (days)', 'Fever','Cough',
'Difficulty breathing', 'Diarrhoea','Vomiting ','Difficulty feeding',
'Convulsions','Jaundice', 'Visible severe wasting', 'Stridor',
       'Obstructed breathing ', 'Absent breathing(apnoea)',
       'Central cyanosis', 'Indrawing', 'Grunting', 'Acidotic breathing',
       'Wheeze', 'Flaring of alae nasi', 'Head nodding/bobbing',
       'Peripheral Pulse', 'Cap refill', 'Pallor / Anaemia',


]