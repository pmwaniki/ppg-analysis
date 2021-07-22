import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch



from datasets.signals import gaus_noise,permute
from datasets.loaders import load_file
from settings import data_dir,output_dir

filename=os.path.join(data_dir,"segments/triage-M721-0.joblib")
ppg = load_file(filename)
red, infrared = np.array(ppg["red"]), np.array(ppg["infrared"])
sd=0.005
noise=np.random.normal(0,sd,len(red))

plt.plot(red)
plt.show()

fig,axs=plt.subplots(3,1)
# fig.patch.set_visible(False)
for i,sig,lab in zip(range(3),[red,noise,red+noise],['Signal','Gaussian noise (sd=0.005)','Signal + Gaussian noise']):
    axs[i].plot(sig,color='black', linestyle='solid', linewidth=1.0)
    axs[i].set_xlabel(lab)
    axs[i].set_xticklabels([])
    axs[i].set_xticks([])
    if i==1:
        axs[i].set_ylim([-0.05,0.05])
    axs[i].set_yticklabels([])
    axs[i].set_yticks([])

fig.savefig(os.path.join(output_dir,"Gaussian noise.png"))
fig.show()


# Signal slicing and permutation
l=len(red)
n_segments=4
l_segment=l//n_segments
i_segments=[i*l_segment for i in range(n_segments)]
order_segments = np.array([1,3,2,0])
red_segments=[red[i:i+l_segment] for i in i_segments]
red_new = [red_segments[i] for i in order_segments]
red_new=np.concatenate(red_new)


fig,axs=plt.subplots(2,1)
# fig.patch.set_visible(False)
for i,sig,lab in zip(range(2),[red,red_new],['Signal','Permuted signal']):
    axs[i].plot(sig,color='black', linestyle='solid', linewidth=1.0)
    axs[i].set_xlabel(lab)
    axs[i].set_xticklabels([])
    axs[i].set_xticks([])

    axs[i].set_yticklabels([])
    axs[i].set_yticks([])
    #line segments
    ys=axs[i].get_ylim()
    for s in i_segments[1:]:
        axs[i].plot([s,s],ys,color='black', linestyle='dashed', linewidth=1.0)

xyA1 = (100, 0.232)
xyB1 = (700,0.247)

xyA2 = (300, 0.232)
xyB2 = (100,0.247)

xyA3 = (700, 0.232)
xyB3 = (300,0.247)

xyA4 = (500, 0.232)
xyB4 = (500,0.247)
con1 = ConnectionPatch(xyA=xyA1, coordsA=axs[0].transData,
                      xyB=xyB1, coordsB=axs[1].transData,arrowstyle='->')
con2 = ConnectionPatch(xyA=xyA2, coordsA=axs[0].transData,
                      xyB=xyB2, coordsB=axs[1].transData,arrowstyle='->')
con3 = ConnectionPatch(xyA=xyA3, coordsA=axs[0].transData,
                      xyB=xyB3, coordsB=axs[1].transData,arrowstyle='->')
con4 = ConnectionPatch(xyA=xyA4, coordsA=axs[0].transData,
                      xyB=xyB4, coordsB=axs[1].transData,arrowstyle='->')
fig.add_artist(con1)
fig.add_artist(con2)
fig.add_artist(con3)
fig.add_artist(con4)
fig.savefig(os.path.join(output_dir,"Signal permutation.png"))
fig.show()

