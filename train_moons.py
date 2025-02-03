import os
os.chdir("/home/fe/belova/projects/bridges/aligned_diffusion_bridges/") 
import sys
sys.path.append(os.getcwd())

from reproducibility.reproducibility import *

AlignExperiment.run("--dataset=moon  --h_dim=64  --n_layers=2  --n_epochs=1  \
                    --reg_weight=1.  --timestep_emb_dim=32  --in_dim=2 --out_dim=2  \
                    --diffusivity_schedule=fbb  --max_diffusivity=1.0 --H=.5 --K=0 \
                    --use_drift_in_doobs=True  --activation=silu").save("moon_silu")

moon = AlignExperiment.load("moon_silu")


plot_marginals(moon.get_marginals(), alpha=.2)

#print(.shape)
samples = moon.sample(samples_num=500, trials_num=7)
print('samples shape',samples.shape)
if len(samples.shape) == 3:
    plot_multiple_marginals(samples, skip_step=4) 
else:
    plot_multiple_marginals(samples[:,:,:,0], skip_step=4)     
export_fig("fig_sb_align_moon_traj")