#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib import colormaps

from scipy.stats import gaussian_kde
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
import sys
import scienceplots

# 设置绘图风格
plt.style.use(['science', 'no-latex', 'nature'])

# 从命令行参数获取数据文件名
dimer = sys.argv[1] #'test_data.xvg'
data = np.loadtxt(dimer)

# 提取自变量和因变量
x = data[:, 0]  # 第一列数据作为自变量
y1 = data[:, 2]  # 第三列数据作为因变量

# 创建线性回归模型并拟合数据
model = LinearRegression()
model.fit(x.reshape(-1, 1), y1)
y_pred = model.predict(x.reshape(-1, 1))

# 计算决定系数（R-squared）
r_squared = model.score(x.reshape(-1, 1), y1)

# 计算均方根误差（RMSE）
rmse_eann = np.sqrt(mean_squared_error(y1, y_pred))

# 使用高斯核密度估计（Gaussian KDE）来计算数据点的相对密度
xy = np.vstack([x, y1])
z = gaussian_kde(xy)(xy)
z = preprocessing.maxabs_scale(z, axis=0, copy=True)

# 创建图表
fig, ax = plt.subplots()

# 绘制散点图，颜色表示相对密度
plt.scatter(x, y1, c=z, s=5, edgecolor='none', cmap=colormaps['summer'])

# 添加一条斜率为1的参考线
ax.axline((0, 0), slope=1, linewidth=1.0, color="k", alpha=0.4)
plt.plot(x, y_pred, linewidth=1.0, color='#279DC4', label='Regression line', alpha=0.4)

# 添加文本标签显示RMSE和R-squared
ax.text(0.98, 0.05, 'R$^2$ = %.2f\nRMSD = %.2f' % (r_squared, rmse_eann, ),
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax.transAxes)

plt.title(f"Bonding Energy {dimer.split('.')[0]} (kJ/mol)")
plt.ylabel("PhyNEO Force Field")
plt.xlabel("MP2")
plt.tick_params(axis='both', which='both', bottom=True, top=False, left=True, right=False, labelbottom=True, labelleft=True)
plt.savefig(f"{dimer.split('.')[0]}.png", dpi=600, bbox_inches='tight')