# =============================================================================
# Federal University of Rio Grande do Sul (UFRGS)
# Connectionist Artificial Intelligence Laboratory (LIAC)
# Renato de Pontes Pereira - rppereira@inf.ufrgs.br
# =============================================================================
# Copyright (c) 2013 Renato de Pontes Pereira, renato.ppontes at gmail dot com
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy 
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights 
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell 
# copies of the Software, and to permit persons to whom the Software is 
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in 
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# =============================================================================

'''
This is a full Ratslam implementation in python. This implementation is based 
on Milford's original implementation [1]_ in matlab, and Christine Lee's python 
implementation [2]_. The original data movies can also be found in [1]_.

This file is the only dependent of OpenCV, which is used to open and convert 
the movie files. Thus, you can change only this file to use other media API.

.. [1] https://wiki.qut.edu.au/display/cyphy/RatSLAM+MATLAB
.. [2] https://github.com/coxlab/ratslam-python
'''

import cv2
import numpy as np
from matplotlib import pyplot as plot
import mpl_toolkits.mplot3d.axes3d as p3

import ratslam

if __name__ == '__main__':
    # Change this line to open other movies
    data = r'/projects/rg_vip_class/neuro/stlucia_testloop.avi'

    video = cv2.VideoCapture(data)
    slam = ratslam.Ratslam()
    fps = video.get(cv2.CAP_PROP_FPS)
    time_per_frame = 1/fps
    
    loop = 0
    _, frame = video.read()
    while True:
        loop += 1

        # RUN A RATSLAM ITERATION ==================================
        _, frame = video.read()
        if frame is None: break

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        slam.digest(img, time_per_frame)
        # ==========================================================

        # Plot each 50 frames
        if loop%10 != 0:
            continue

        # PLOT THE CURRENT RESULTS =================================
        b, g, r = cv2.split(frame)
        rgb_frame = cv2.merge([r, g, b])

        plot.clf()

        # RAW IMAGE -------------------
        ax = plot.subplot(2, 2, 1)
        plot.title('RAW IMAGE')
        plot.imshow(rgb_frame)
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        # -----------------------------

        # RAW ODOMETRY ----------------
        plot.subplot(2, 2, 2)
        plot.title('RAW ODOMETRY')
        plot.plot(slam.odometry[0], slam.odometry[1])
        plot.plot(slam.odometry[0][-1], slam.odometry[1][-1], 'ko')
        #------------------------------

        # POSE CELL ACTIVATION --------
        #ax = plot.subplot(2, 2, 3, projection='3d')
        #plot.title('POSE CELL ACTIVATION')
        #x, y, th = slam.pc
        #ax.plot(x, y, 'x')
        #ax.plot3D([0, 60], [y[-1], y[-1]], [th[-1], th[-1]], 'K')
        #ax.plot3D([x[-1], x[-1]], [0, 60], [th[-1], th[-1]], 'K')
        #ax.plot3D([x[-1], x[-1]], [y[-1], y[-1]], [0, 36], 'K')
        #ax.plot3D([x[-1]], [y[-1]], [th[-1]], 'mo')
        #ax.grid()
        #ax.axis([0, 60, 0, 60]);
        #ax.set_zlim(0, 36)
        # -----------------------------

        #plt.plot(sim.trange(), sim.data[inp_p].T[0], c="k", label="Input")
        #plt.plot(sim.trange(), sim.data[pre_p].T[0], c="b", label="Pre")
        #plt.plot(sim.trange(), sim.data[post_p].T[0], c="r", label="Post")
        #plt.ylabel("Dimension 1")
        #plt.legend(loc="best")


        plot.subplot(2, 3, 3)
        plot.title("X Probes")
        plot.plot(slam.pose_cells.simulator.trange(), slam.pose_cells.simulator.data[slam.pose_cells.pre_probe].T[0], c="k", label="Pre")
        plot.plot(slam.pose_cells.simulator.trange(), slam.pose_cells.simulator.data[slam.pose_cells.post_probe].T[0], c="b", label="Post")
        plot.plot(slam.pose_cells.simulator.trange(), slam.pose_cells.simulator.data[slam.pose_cells.error_probe].T[0], c="r", label="Error")
        plot.legend(loc="best")

        #plot.subplot(2, 3, 3)
        #plot.title("Y Probes")
        #plot.plot(slam.pose_cells.simulator.trange(), slam.pose_cells.simulator.data[slam.pose_cells.pre_probe].T[1], c="k", label="Pre")
        #plot.plot(slam.pose_cells.simulator.trange(), slam.pose_cells.simulator.data[slam.pose_cells.post_probe].T[1], c="b", label="Post")
        #plot.plot(slam.pose_cells.simulator.trange(), slam.pose_cells.simulator.data[slam.pose_cells.error_probe].T[1], c="r", label="Error")
        #plot.legend(loc="best")

        #plot.subplot(2, 3, 3)
        #plot.title("TH Probes")
        #plot.plot(slam.pose_cells.simulator.trange(), slam.pose_cells.simulator.data[slam.pose_cells.pre_probe].T[2], c="k", label="Pre")
        #plot.plot(slam.pose_cells.simulator.trange(), slam.pose_cells.simulator.data[slam.pose_cells.post_probe].T[2], c="b", label="Post")
        #plot.plot(slam.pose_cells.simulator.trange(), slam.pose_cells.simulator.data[slam.pose_cells.error_probe].T[2], c="r", label="Error")
        #plot.legend(loc="best")

        # EXPERIENCE MAP --------------
        plot.subplot(2, 2, 4)
        plot.title('EXPERIENCE MAP')
        xs = []
        ys = []
        for exp in slam.experience_map.exps:
            xs.append(exp.x_m)
            ys.append(exp.y_m)

        plot.plot(xs, ys, 'bo')
        plot.plot(slam.experience_map.current_exp.x_m,
                  slam.experience_map.current_exp.y_m, 'ko')
        # -----------------------------

        plot.tight_layout()
        # plot.savefig('C:\\Users\\Renato\\Desktop\\results\\forgif\\' + '%04d.jpg'%loop)
        plot.pause(0.1)
        # ==========================================================

    print('DONE!')
    print('n_ templates:', len(slam.view_cells.cells))
    print('n_ experiences:', len(slam.experience_map.exps))
    plot.show()
