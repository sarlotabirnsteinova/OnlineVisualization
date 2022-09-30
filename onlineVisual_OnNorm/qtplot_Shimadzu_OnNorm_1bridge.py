#!/gpfs/exfel/sw/software/xfel_anaconda3/1.1/bin/python
"""
online device
    - visualise raw and normalised image from Shimadzu 1 & 2
    - one bridge
    - now it should work also if just one camera is running
TODO 
    - add parameters as input 
    - parameter to change interface, if needed
    - make 2 bridges as that
    - make it to take parameter for which camera

Parameters
----------
run_flat : int
    Run number with flat-fields.

Returns
-------
plot
    updating stacks of images for both cameras now, raw and normalised, and combined view  
"""
from PyQt5 import QtGui
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot, QTimer
from PyQt5.QtWidgets import QCheckBox
import pyqtgraph as pg
from pyqtgraph.dockarea import *
import sys
import time
import datetime
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from karabo_bridge import Client
from collections import deque
from threading import Thread
import pandas as pd
from dffc_functions_online import *
import ADMMCTF


# interface = 'tcp://10.253.0.52:54456'  # for Shimadzu bridge
interface = 'tcp://10.253.0.52:54333' # KOHZU bridge 

## device, key pairs
device_name = {'zyla5':'SPB_EXP_ZYLA/CAM/5:output','zyla3':'SPB_EXP_ZYLA/CAM/3:output','zyla4':'SPB_EXP_ZYLA/CAM/4:output','shimadzu2':'SPB_EHD_HPVX2_2/CAM/CAMERA:output','shimadzu1':'SPB_EHD_HPVX2_1/CAM/CAMERA:output','shimadzu1del':'SPB_EHD_HPVX2_1/TSYS/STBY','shimadzu2del':'SPB_EHD_HPVX2_2/TSYS/STBY','dg1':'SPB_EXP_SYS/DG/DG1','dg2':'SPB_EXP_SYS/DG/DG2'}
device_prop = {'camera': 'data.image.data', 'motor': 'actualPosition.value','delayG':'G.delay.value','delayA':'A.delay.value', 'delayCam': 'actualDelay.value'}

# parameters
rank = 20
# parameters OnNorm            
flat_run = 40
# pca_info_cam1, pca_info_cam2 = read_pca_info_bothCameras('pca_info_both_flat_run'+ str(flat_run) + "_rank" +  str(rank) + '.h5')
# or from different h5 files
flat_run_cam1, flat_run_cam2 = 40, 40
pca_info_cam1 = read_pca_info_all("pca_info_cam1" + "_flat_run" + str(flat_run_cam1) + "_rank" + str(rank) + ".hf5")
pca_info_cam2 = read_pca_info_all("pca_info_cam2" + "_flat_run" + str(flat_run_cam2) + "_rank" + str(rank) + ".hf5")

ds_parameter = (2,4)
w0_last = True


class bridgeClientShimadzu(QThread):
    newDataSignal =  pyqtSignal(object)
    def __init__(self, interface):
        super().__init__()
        self.client = Client(interface, timeout = 5)
        try: 
            data, meta = self.client.next()
            #print( data.keys() )
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
            time.sleep(0.1)


class imageView(QtGui.QMainWindow):
    def __init__(self, parent=None, title=None):
        super().__init__(parent)

        ## gui init
        self.widget = QtGui.QWidget()
        self.area = DockArea()
        self.setCentralWidget(self.area)
        self.resize(1600,1200)
        self.setWindowTitle('Shimadzu Camera View')

        self.d0 = Dock("", size=(50, 100))
        self.d1 = Dock("Shimadzu 1: raw", size=(250,400))
        self.d11 = Dock("Shimadzu 1: normalised", size=(250,400), closable=True)
        self.d2 = Dock("Shimadzu 2: raw", size=(250,400))
        self.d22 = Dock("Shimadzu 2: normalised", size=(250,400), closable=True)
        self.d3 = Dock("Shimadzu combined: raw", size=(250,400))
        self.d33 = Dock("Shimadzu combined: normalised", size=(250,400), closable=True)
    

        self.area.addDock(self.d0, 'left')
        self.area.addDock(self.d1,'bottom',self.d0)
        self.area.addDock(self.d11,'right',self.d1)
        self.area.addDock(self.d2,'bottom',self.d1)
        self.area.addDock(self.d22,'bottom',self.d11)
        self.area.addDock(self.d3,'bottom',self.d2)
        self.area.addDock(self.d33,'bottom',self.d22)


        # Shimadzu 1 
        self.imv1 = pg.ImageView()  
        self.d1.addWidget(self.imv1)
        self.imv11 = pg.ImageView()  
        self.d11.addWidget(self.imv11)
        # Shimadzu 2
        self.imv2 = pg.ImageView()  
        self.d2.addWidget(self.imv2)
        self.imv22 = pg.ImageView()  
        self.d22.addWidget(self.imv22)
        # Shimadzu combined
        self.imv3 = pg.ImageView()  
        self.d3.addWidget(self.imv3)
        self.imv33 = pg.ImageView()  
        self.d33.addWidget(self.imv33)

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
        self.clientThread = bridgeClientShimadzu(interface)
        self.clientThread.start()
        self.clientThread.finished.connect(self.threadFin)
        self.clientThread.newDataSignal.connect(self.update)
  

    def update(self, dataBridge):
        """ called by bridgeClientShimadzu when bridge receives Shimadzu data
        d = data, meta
        """
        dataHPVX, meta = dataBridge
        imgs_Shim1 = None
        imgs_Shim2 = None
        # imgs_Shim1_corrected = None
        # imgs_Shim2_corrected = None

        # raw data
        if device_name['shimadzu1'] in dataHPVX.keys(): # 
            imgs_Shim1 = np.array(dataHPVX[device_name['shimadzu1']][device_prop['camera']]*1.)   # SHimadzu 1   # *1. makes a copy                     
            self.imv1.setImage(imgs_Shim1)    
        if device_name['shimadzu2'] in dataHPVX.keys():   # 
            imgs_Shim2 = np.array( dataHPVX[device_name['shimadzu2']][device_prop['camera']]*1. )   # SHimadzu 2
            self.imv2.setImage(imgs_Shim2)    
        # if imgs_Shim1 is not None and imgs_Shim2 is not None:
        #     imgs_combined = self.combine_Shimadzus(imgs_Shim1,imgs_Shim2)
        #     if imgs_combined is not None:
        #         self.imv3.setImage(imgs_combined)    

        # normalised
        if imgs_Shim1 is not None:
            imgs_Shim1_corrected = dffc_correct(imgs_Shim1, pca_info_cam1, ds_parameter, x0_last=w0_last)
            self.imv11.setImage(imgs_Shim1_corrected)    
        if imgs_Shim2 is not None:
            imgs_Shim2_corrected = dffc_correct(imgs_Shim2, pca_info_cam1, ds_parameter, x0_last=w0_last)
            self.imv22.setImage(imgs_Shim2_corrected)  
        # if imgs_Shim1 is not None and imgs_Shim2 is not None:
        #     imgs_combined_corrected = self.combine_Shimadzus(imgs_Shim1_corrected,imgs_Shim2_corrected)
        #     if imgs_combined is not None:
        #         self.imv33.setImage(imgs_combined_corrected)

        QtGui.QApplication.processEvents()

    def combine_Shimadzus(self,Sh1=None,Sh2=None):
        """
        combine two arrays into one long array along 0 axis
        :param Sh1: first frame from first camera, frams on even indexes of a new array
        :param Sh2: frames from this camera on odd indexes of a new array
        :returns a combined array along 0 axis    
        """
        if Sh1 is not  None and Sh2 is not None:
            if len(Sh1.shape)==3 and len(Sh2.shape)==3:  
                tid_both = Sh1.shape[0]+Sh2.shape[0]
                combined = np.zeros((tid_both,Sh1.shape[1],Sh1.shape[2]))
                for t in range(0,tid_both):
                    t_ = t//2
                    if t%2==0:
                        combined[t,:,:] = Sh1[t_,:,:]
                    else:
                        combined[t,:,:] = Sh2[t_,:,:]
                return combined
            else:
                return None
        else:
            return None


    def threadFin(self):
        pass
    
    def closeEvent(self,event):
        # stop thread here... weird on macos
        event.accept()
    
    def setTranspose(self,state):
        self.transpose = state


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