#!/usr/bin/env python
import sys
import matplotlib.pyplot as plt
import numpy as np
import scienceplots
import os 
import re

plt.style.use(['science','no-latex', 'nature'])

#file = f"rdf_Li_F6P_[Li1r_FPo].csv"
file = sys.argv[1]
left_part, _, right_part = file.partition('_[')
right_part, _, end_part = right_part.rpartition('].csv')

data = np.loadtxt(file, delimiter=';')

fig, ax1 = plt.subplots(figsize=(5,5*0.618))
ax2 = ax1.twinx()
# 绘制第一个图形
ax1.plot(data[:,0], data[:,1], label=f'{right_part}')
ax1.tick_params(axis='y')
ax1.legend()
ax1.set_xlabel('Distance (angstrom)')
ax1.set_ylabel('Radial Distribution Function')

ax2.plot(data[:,0], data[:,2], label='CN', linestyle='--')
ax2.set_ylabel('Coordination Number')
ax2.tick_params(axis='y')
ax2.set_ylim(-5, 50)

plt.xlim(100, 1000)
# plt.title('RDF'%hydrogens)
os.makedirs('res', exist_ok=True)
plt.savefig(f"res/res_{file.split('.')[0]}.png", dpi=300, bbox_inches = 'tight')

