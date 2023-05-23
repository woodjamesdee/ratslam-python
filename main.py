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
import shutil

import ratslam

if __name__ == '__main__':
    # Change this line to open other movies
    data = r'/projects/rg_vip_class/neuro/stlucia_testloop.avi'

    video = cv2.VideoCapture(data)
    slam = ratslam.Ratslam()
    fps = video.get(cv2.CAP_PROP_FPS)
    time_per_frame = 1/fps

    do_plotting = True
    first_time = True
    video_loops = 0
    
    loop = 0
    _, frame = video.read()
    try:
        while True:
            loop += 1

            # RUN A RATSLAM ITERATION ==================================
            _, frame = video.read()
            if frame is None:
                break

            img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            slam.digest(img, time_per_frame)

            if first_time:
                slam.nengo_pose_cells.save("weights_initial.npy")
                first_time = False

            # ==========================================================

            if not do_plotting:
                continue

            # Plot each 50 frames
            if loop%50 != 0:
                continue

            # PLOT THE CURRENT RESULTS =================================
            b, g, r = cv2.split(frame)
            rgb_frame = cv2.merge([r, g, b])

            plot.clf()

            # RAW IMAGE -------------------
            ax = plot.subplot(3, 3, 1)
            plot.title('RAW IMAGE')
            plot.imshow(rgb_frame)
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
            # -----------------------------

            # RAW ODOMETRY ----------------
            plot.subplot(3, 3, 2)
            plot.title('RAW ODOMETRY')
            plot.plot(slam.odometry[0], slam.odometry[1])
            plot.plot(slam.odometry[0][-1], slam.odometry[1][-1], 'ko')
            #------------------------------

            # POSE CELL ACTIVATION --------
            ax = plot.subplot(3, 3, 3, projection='3d')
            plot.title('POSE CELL ACTIVATION')
            x, y, th = slam.pc
            ax.plot(x, y, 'x')
            ax.plot3D([0, 60], [y[-1], y[-1]], [th[-1], th[-1]], 'k')
            ax.plot3D([x[-1], x[-1]], [0, 60], [th[-1], th[-1]], 'k')
            ax.plot3D([x[-1], x[-1]], [y[-1], y[-1]], [0, 36], 'k')
            ax.plot3D([x[-1]], [y[-1]], [th[-1]], 'mo')
            ax.grid()
            ax.axis([0, 60, 0, 60]);
            ax.set_zlim(0, 36)
            # -----------------------------

            # EXPERIENCE MAP --------------
            plot.subplot(3, 3, 4)
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

            plot.subplot(3, 3, 5)
            plot.title("X Probing")
            #plot.plot(slam.pose_cells.simulator.trange(), slam.pose_cells.simulator.data[slam.pose_cells.pre_probe].T[0], c="k", label="Pre")
            #plot.plot(slam.pose_cells.simulator.trange(), slam.pose_cells.simulator.data[slam.pose_cells.post_probe].T[0], c="b", label="Post")
            #plot.plot(slam.pose_cells.simulator.trange(), slam.pose_cells.simulator.data[slam.pose_cells.error_probe].T[0], c="r", label="Error")
            #plot.plot(slam.pose_cells.simulator.trange(), slam.pose_cells.simulator.data[slam.pose_cells.active_probe].T[0], c="g", label="Active", linewidth=3)
            #plot.legend(loc="best")
            plot.plot(slam.nengo_pose_cells.simulator.trange(), slam.nengo_pose_cells.simulator.data[slam.nengo_pose_cells.post_probe].T[0], c="b", label="Post")
            plot.plot(slam.nengo_pose_cells.simulator.trange(), slam.nengo_pose_cells.simulator.data[slam.nengo_pose_cells.active_probe].T[0], c="g", label="Active", linewidth=3)

            plot.subplot(3, 3, 6)
            plot.title("Y Probing")
            #plot.plot(slam.pose_cells.simulator.trange(), slam.pose_cells.simulator.data[slam.pose_cells.pre_probe].T[0], c="k", label="Pre")
            #plot.plot(slam.pose_cells.simulator.trange(), slam.pose_cells.simulator.data[slam.pose_cells.post_probe].T[1], c="b", label="Post")
            #plot.plot(slam.pose_cells.simulator.trange(), slam.pose_cells.simulator.data[slam.pose_cells.error_probe].T[0], c="r", label="Error")
            #plot.plot(slam.pose_cells.simulator.trange(), slam.pose_cells.simulator.data[slam.pose_cells.active_probe].T[1], c="g", label="Active", linewidth=3)
            #plot.legend(loc="best")
            plot.plot(slam.nengo_pose_cells.simulator.trange(), slam.nengo_pose_cells.simulator.data[slam.nengo_pose_cells.post_probe].T[1], c="b", label="Post")
            plot.plot(slam.nengo_pose_cells.simulator.trange(), slam.nengo_pose_cells.simulator.data[slam.nengo_pose_cells.active_probe].T[1], c="g", label="Active", linewidth=3)

            plot.subplot(3, 3, 7)
            plot.title("Th Probing")
            #plot.plot(slam.pose_cells.simulator.trange(), slam.pose_cells.simulator.data[slam.pose_cells.pre_probe].T[0], c="k", label="Pre")
            #plot.plot(slam.pose_cells.simulator.trange(), slam.pose_cells.simulator.data[slam.pose_cells.post_probe].T[2], c="b", label="Post")
            #plot.plot(slam.pose_cells.simulator.trange(), slam.pose_cells.simulator.data[slam.pose_cells.error_probe].T[0], c="r", label="Error")
            #plot.plot(slam.pose_cells.simulator.trange(), slam.pose_cells.simulator.data[slam.pose_cells.active_probe].T[2], c="g", label="Active", linewidth=3)
            #plot.legend(loc="best")
            plot.plot(slam.nengo_pose_cells.simulator.trange(), slam.nengo_pose_cells.simulator.data[slam.nengo_pose_cells.post_probe].T[2], c="b", label="Post")
            plot.plot(slam.nengo_pose_cells.simulator.trange(), slam.nengo_pose_cells.simulator.data[slam.nengo_pose_cells.active_probe].T[2], c="g", label="Active", linewidth=3)

            plot.tight_layout()
            # plot.savefig('C:\\Users\\Renato\\Desktop\\results\\forgif\\' + '%04d.jpg'%loop)
            plot.pause(0.1)
            # ==========================================================

    finally:
        #try:
        #    shutil.copyfile("weights.npy", "weights.npy.prev")
        #except:
        #    print("No prev to copy")
        #np.save("weights.npy", slam.pose_cells.simulator.data[slam.pose_cells.pre_post_connection].weights)
        #np.save("weights.npy", slam.pose_cells.simulator.data[slam.pose_cells.weights_probe][-1])
        #print("Saved weights to file")
        #try:
        #    old_weights = np.load("weights.npy.prev")
        #    print(old_weights)
        #    diff = slam.pose_cells.simulator.data[slam.pose_cells.pre_post_connection].weights - old_weights
        #    abs_avg = np.average(np.absolute(diff))
        #    print(f"Avg of diff: {abs_avg}")
        #except Exception as e:
        #    print("Couldn't calculate diff")
        #    print(e)
        slam.nengo_pose_cells.save("weights_complete.npy")
        print('DONE!')
        #print('n_ templates:', len(slam.view_cells.cells))
        #print('n_ experiences:', len(slam.experience_map.exps))
        if do_plotting:
            plot.show()