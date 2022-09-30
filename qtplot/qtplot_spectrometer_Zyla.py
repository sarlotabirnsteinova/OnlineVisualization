#!/gpfs/exfel/sw/software/xfel_anaconda3/1.1/bin/python
"""
online device
    - visualise spectrometer image, normalised mean over horizontal dimension of ROI, running mean of mean  (Zyla/5 now)
    - one bridge
TODO 
    - Zyla number as parameter ?
    - ROI not working well --> redo
    - dummy version of rescale x-axis according to energy vs. pixel value --> redo   

Parameters
----------
run_flat : int
    Run number with flat-fields.
roi_values : list
    Values for ROI of an image.
rot_angle : int 
    Angle to rotate an image.
Returns
-------
plot
    updating image and two line plots
"""
from PyQt5 import QtGui
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot, QTimer
from PyQt5.QtWidgets import QCheckBox
import pyqtgraph as pg
from pyqtgraph.dockarea import *
import sys, argparse
import time
import datetime
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from karabo_bridge import Client
from collections import deque
import pandas as pd
from dffc_functions_online import read_pca_info_all
from scipy import ndimage, misc


# ................
parser = argparse.ArgumentParser(description=""" 
This script does Principal Component Analysis on flat-field data, reads dark data and outputs relevant information needed for PCA flat-field reconstruction""")
# data dir & data proposal
parser.add_argument('-rf', type=int, dest='run_flat', default=20, help="Run number with flat-fields.")
# parser.add_argument('-l', nargs='+', dest='roi_values', default=[1000, 1300, 200 , 2000],help="Values for ROI of an image.")
parser.add_argument('-l', nargs='+', dest='roi_values', default=None,help="Values for ROI of an image.")
parser.add_argument('-a', type=int, dest='rot_angle', default=0, help="Angle to rotate Zyla image.")

# parameters
args = parser.parse_args()
run_number_flat = args.run_flat
angle = args.rot_angle
eV0 = 978
deV = 0.0238
# ...............
interfaceHPVX = 'tcp://10.253.0.51:45432'  # Zyla 5 (only)
# interfaceHPVX = 'tcp://10.253.0.52:54456'  # Shimadzu 1 
# interfaceSlave = 'tcp://10.253.0.52:54333' # KOHZU bridge, Shimadzu 2 

# read darkflat info for normalisation 
norm_info = read_pca_info_all(f"norm_info_Zyla5_flat_r{run_number_flat:04d}.h5")
mf = norm_info['mean_flat']
md = norm_info['mean_flat']
if args.roi_values is not None:
    roi = args.roi_values
else:
    roi = [0, mf.shape[0]+10,100, 2200] 
# # fuck
# md = np.zeros(imageData.shape)
# mf = np.ones(imageData.shape)

# x_axis = np.linspace((roi[2]-eV0)*deV,(roi[3]-eV0)*deV,roi[3]-roi[2])

## device, key pairs
device_name = {'zyla5':'SPB_EXP_ZYLA/CAM/5:output','zyla3':'SPB_EXP_ZYLA/CAM/3:output','zyla4':'SPB_EXP_ZYLA/CAM/4:output','shimadzu2':'SPB_EHD_HPVX2_2/CAM/CAMERA:output','shimadzu1':'SPB_EHD_HPVX2_1/CAM/CAMERA:output','shimadzu1del':'SPB_EHD_HPVX2_1/TSYS/STBY','shimadzu2del':'SPB_EHD_HPVX2_2/TSYS/STBY','dg1':'SPB_EXP_SYS/DG/DG1','dg2':'SPB_EXP_SYS/DG/DG2'}
device_prop = {'camera': 'data.image.data', 'motor': 'actualPosition.value','delayG':'G.delay.value','delayA':'A.delay.value', 'delayCam': 'actualDelay.value'}
           


class bridgeClientShimadzu(QThread):
    newDataSignal =  pyqtSignal(object)
    def __init__(self, interface):
        super().__init__()
        self.client = Client(interface, timeout = 5)
        try: 
            data, meta = self.client.next()
            print( data.keys() )
        except TimeoutError as e:
            pass
    def __del__(self):
        self.wait()
    def run(self):
        print('starting queue thread')
        while True:
            try: 
                data, meta = self.client.next()
                self.newDataSignal.emit([data, meta])
            except TimeoutError as e:
                #print('bridge timeout...')
                data = None
            time.sleep(1)


class imageView(QtGui.QMainWindow):
    def __init__(self, parent=None, title=None):
        super().__init__(parent)
        
       # parameters & variables
        self.del_que = deque(maxlen=10000)
        #
        self.df = pd.DataFrame(columns=['cam_mean'],dtype=np.float64)
        self.max_len = 10000


        ## gui init
        self.widget = QtGui.QWidget()
        self.area = DockArea()
        self.setCentralWidget(self.area)
        self.resize(800,1200)
        self.setWindowTitle('Shimadzu Camera View')

        self.d0 = Dock("", size=(100, 50))
        self.d1 = Dock("Zyla 5", size=(800,400))
        self.d11 = Dock("spectrometer", size=(800,400), closable=True)
        self.d22 = Dock("spectrometer running mean", size=(800,400), closable=True)

        self.area.addDock(self.d0, 'left')
        self.area.addDock(self.d1,'bottom',self.d0)
        self.area.addDock(self.d11,'bottom',self.d1)
        self.area.addDock(self.d22,'bottom',self.d11)


        # buttons
        w0 = pg.LayoutWidget()
        btnClear = QtGui.QPushButton('clear df')
        # btnSave = QtGui.QPushButton('save csv')

        w0.addWidget(btnClear, row=0, col=0)
        # w0.addWidget(btnSave, row=0, col=1)

        btnClear.clicked.connect(self.clearDF)
        # btnSave.clicked.connect(self.saveDF)
        self.d0.addWidget(w0)
        # 
        self.imv1 = pg.ImageView()  # moving
        self.d1.addWidget(self.imv1)
        #
        self.scat_w1 = pg.PlotWidget()
        self.d11.addWidget(self.scat_w1)
        #
        self.scat_w2 = pg.PlotWidget()
        self.d22.addWidget(self.scat_w2)
        #
        self.plot_config = {'autoRange': False, 
                                'autoLevels': False, 
                                'autoHistogramRange': False,
                                 'axes' : {'x':0, 'y':1}
                                 }
        self.imv_config = {'autoRange': False, 
                            'autoLevels': False, 
                            'autoHistogramRange': False,
                            'axes' : {'x':0, 'y':1}
                            }
        self.show()

        #shimadzu bridge info
        self.clientThread = bridgeClientShimadzu(interfaceHPVX)
        self.clientThread.start()
        self.clientThread.finished.connect(self.threadFin)
        self.clientThread.newDataSignal.connect(self.update)


    def updatePlot(self, data):
            imageData = None
            cam_mean = None

            # getting data from ZMQ
            if (device_name['zyla5'] in data.keys() ): 
                    imageData = np.array(data[device_name['zyla5']][device_prop['camera']]*1.)     # *1. makes a copy 
                    #.....        
            
            # update graph & update DF
            if imageData is not None:


                # # calculate spectrum of flat
                # mf_norm = mf #- md
                # mf_roi = mf_norm[roi[0]:roi[1],roi[2]:roi[3]]
                # mf_mean = np.mean(mf_roi,axis=(0))
                # # normalise img
                # image_norm = (imageData - md)#/mf 
                image_norm = imageData
                # rotate if necessary
                if angle != 0:
                    image_norm = self.rotate_img(image_norm,angle)
                # ROI
                img_roi = image_norm[roi[0]:roi[1],roi[2]:roi[3]]
                # calculate mean along axis 0r    roi[0]:roi[1],roi[2]:roi[3]
                img_mean = np.mean(img_roi,axis=(0))

                # spec_norm = img_mean / mf_mean
                spec_norm = img_mean
                # flip?

                # plot current img & mean 
                self.imv1.setImage(imageData[roi[0]:roi[1],roi[2]:roi[3]].T)
                # print(img_mean.shape) 
                # self.scat_plot1.addPoints(range(img_mean.shape[0]),img_mean, size=4, pen=pg.mkPen(None), brush=pg.mkBrush('w'))
                # self.scat_plot1.clear()
                self.scat_w1.clear()
                # self.scat_w1.plot(x_axis,spec_norm)#, size=4, pen=pg.mkPen(None), brush=pg.mkBrush('w'))
                self.scat_w1.plot(range(spec_norm.shape[0]),spec_norm)
                # update DF & plot running mean (flipped ...)
                self.updateDF(spec_norm)
                d =  self.df['cam_mean'].mean()
                dfliped = self.flip(d)
                self.scat_w2.clear()
                # self.scat_w2.plot(x_axis,list(dfliped))#, size=4, pen=pg.mkPen(None), brush=pg.mkBrush('w'))
                self.scat_w2.plot(range(spec_norm.shape[0]),list(dfliped))
                # calculate pixel --> eV   

            
            QtGui.QApplication.processEvents()

    def update(self, d):
        """ called by bridgeClientShimadzu when bridge receives Shimadzu data
        d = data, meta
        """
        data, meta = d
        self.updatePlot(data)   

    
    def getDataFromQueue(self):
        data, meta = self.queue.pop()
        return data, meta
    
    def updateDF(self, cam1_meanBuff):
        newData = {'cam_mean':cam1_meanBuff}
        newIdx = self.df.shape[0]
        self.df.loc[newIdx] = newData

    def flip(self,d):
        dm = np.mean(d)
        dflip = dm - (d - dm)
        return dflip

    def rotate_img(self,img,angle):
        img_rot = ndimage.rotate(img, angle, reshape=False)
        return img_rot

    def threadFin(self):
        pass
    
    def closeEvent(self,event):
        # stop thread here... weird on macos
        event.accept()

    # button fun
    def clearDF(self):
        self.df = pd.DataFrame(columns=['cam_mean'],dtype=np.float64)
        print('DataFrame cleared.')

    # def saveDF(self):
    #     outName = 'DF_delayBuffer2_{}.h5'.format(datetime.datetime.now().strftime('%Y%m%dT%H%M'))
    #     print('Saving as {}'.format(outName))
    #     # self.df.to_csv(outName)
    #     with h5py.File(outName, "w") as f:
    #         f.create_dataset('meanBuff', data = self.df.cam1_meanBuff.to_numpy())
    #         f.create_dataset('dels', data = self.df.delay.to_numpy())


def main():
    app = QtGui.QApplication(sys.argv)
    form = imageView()
    form.show()
    # add timer to allow for catching ctrl+c :
    timer = QTimer()
    timer.timeout.connect(lambda: None)
    timer.start(100)
    # app.exec_()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
