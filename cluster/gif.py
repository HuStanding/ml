# -*- coding: utf-8 -*-
# @Author: huzhu
# @Date:   2019-11-12 17:39:15
# @Last Modified by:   huzhu
# @Last Modified time: 2019-11-12 18:04:18

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig, ax = plt.subplots()
xdata, ydata = [], []
ln, = plt.plot([], [], 'ro',animated=True)

def init():
    ax.set_xlim(-np.pi,np.pi)
    ax.set_ylim(-1, 1)
    return ln,

def update(frame):
    xdata.append(frame)
    ydata.append(np.sin(frame))
    ln.set_data(xdata, ydata)
    return ln,

anim = animation.FuncAnimation(fig, update, frames=np.linspace(-np.pi,np.pi, 90),interval=10,
                    init_func=init,blit=True)
plt.show()
#anim.save('test_animation.gif',writer='pillow')