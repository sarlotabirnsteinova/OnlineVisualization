#!/gpfs/exfel/sw/software/xfel_anaconda3/1.1/bin/python
"""
online device
    
    images raw & combined, if saved to dataframe: possible to image flat-field corrected & phase retrieved  
    combines data from 2 bridges

TODO 
    - redo to take parameters !!
    - change combine cameras as 'inter' mode or 'one_first' or 'two_first'


Parameters
----------
None 
Returns
-------
plot
    updating 9 images: raw 2 Shimadzus and combined (interleaved mode), saved normalised& phase retrieved imgs from dataframe 
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


interfaceHPVX =  'tcp://10.253.0.52:54333' # KOHZU bridge, Shimadzu 2
interfaceSlave = 'tcp://10.253.0.52:54456'  # Shimadzu 1

## device, key pairs
device_name = {'zyla5':'SPB_EXP_ZYLA/CAM/5:output','zyla3':'SPB_EXP_ZYLA/CAM/3:output','zyla4':'SPB_EXP_ZYLA/CAM/4:output','shimadzu2':'SPB_EHD_HPVX2_2/CAM/CAMERA:output','shimadzu1':'SPB_EHD_HPVX2_1/CAM/CAMERA:output','shimadzu1del':'SPB_EHD_HPVX2_1/TSYS/STBY','shimadzu2del':'SPB_EHD_HPVX2_2/TSYS/STBY','dg1':'SPB_EXP_SYS/DG/DG1','dg2':'SPB_EXP_SYS/DG/DG2'}
device_prop = {'camera': 'data.image.data', 'motor': 'actualPosition.value','delayG':'G.delay.value','delayA':'A.delay.value', 'delayCam': 'actualDelay.value'}


# parameters
rank = 20
# parameters OnNorm            
flat_run = 40
pca_info_path = "pca_info_both" + "_flat_run" + str(flat_run) + "_rank" + str(rank) + ".h5"
# pca_info_cam1, pca_info_cam2 = read_pca_info_bothCameras(pca_info_path)
# or from different h5 files
flat_run_cam1, flat_run_cam2 = 40, 40
pca_info_cam1 = read_pca_info_all("pca_info_cam1" + "_flat_run" + str(flat_run) + "_rank" + str(rank) + ".hf5")
pca_info_cam2 = read_pca_info_all("pca_info_cam2" + "_flat_run" + str(flat_run) + "_rank" + str(rank) + ".hf5")

ds_parameter = (2,4)
w0_last = True
# parameters PhaseReco.................
# Physical Parameters
E = 9.3  # keV
wvl = 12.4/E*1e-10
pxs = 3.2e-6  # pixelsize
DOF = pxs**2/wvl
D = DOF*100
betaoverdelta = 5e-1
# ADMM-TV setting
niter = 200  # number of iterations
eps = 1e-3
# stopping threshold
tau = 5e-5  # connection strength
eta = 0.02*tau  # regularization strength
phys = 0  # flag for the physical constraints 




class bridgeClientShimadzu(QThread):
    newDataSignal =  pyqtSignal(object)
    def __init__(self, interface):
        super().__init__()
        self.client = Client(interface, timeout = 5)
        try: 
            data, meta = self.client.next()
            # print( data.keys() )
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
            time.sleep(0.01)


class imageView(QtGui.QMainWindow):
    def __init__(self, parent=None, title=None):
        super().__init__(parent)
        # parameters & variables
        self.df1 = pd.DataFrame(columns=['normalised', 'phase'],dtype=object)
        self.df2 = pd.DataFrame(columns=['normalised', 'phase'],dtype=object)
        self.df3 = pd.DataFrame(columns=['normalised', 'phase'],dtype=object)
        self.max_len = 10

        self.idx_viewProc = 0

        self.current1 = None
        self.current2 = None

        ## gui init
        self.widget = QtGui.QWidget()
        self.area = DockArea()
        self.setCentralWidget(self.area)
        self.resize(1600,1200)
        self.setWindowTitle('Shimadzu Camera View')

        self.d0 = Dock("", size=( 50, 100))
        self.d1 = Dock("Shimadzu 1: raw", size=(250,400))
        self.d11 = Dock("Shimadzu 1: normalised", size=(250,400), closable=True)
        self.d111 = Dock("Shimadzu 1: phase", size=(250,400), closable=True)
        self.d2 = Dock("Shimadzu 2: raw", size=(250,400))
        self.d22 = Dock("Shimadzu 2: normalised", size=(250,400), closable=True)
        self.d222 = Dock("Shimadzu 2: phase", size=(250,400), closable=True)
        self.d3 = Dock("Shimadzu combined: raw", size=(250,400))
        self.d33 = Dock("Shimadzu combined: normalised", size=(250,400), closable=True)
        self.d333 = Dock("Shimadzu combined: phase", size=(250,400), closable=True)


        self.area.addDock(self.d0, 'left')
        self.area.addDock(self.d1,'bottom',self.d0)
        self.area.addDock(self.d11,'right',self.d1)
        self.area.addDock(self.d111,'right',self.d11)
        self.area.addDock(self.d2,'bottom',self.d1)
        self.area.addDock(self.d22,'bottom',self.d11)
        self.area.addDock(self.d222,'bottom',self.d111)
        self.area.addDock(self.d3,'bottom',self.d2)
        self.area.addDock(self.d33,'bottom',self.d22)
        self.area.addDock(self.d333,'bottom',self.d222)

        # buttons
        w0 = pg.LayoutWidget()
        # 3 buttons for saving to df, viewing prev. or next. from df 
        btnSave = QtGui.QPushButton('Save current data')
        btnViewPrev = QtGui.QPushButton('View Prev.')
        btnViewNext = QtGui.QPushButton('View Next')

        w0.addWidget(btnSave, row=0, col=0)
        w0.addWidget(btnViewPrev, row=0, col=1)
        w0.addWidget(btnViewNext, row=0, col=2)

        btnSave.clicked.connect(self.saveDF)
        btnViewPrev.clicked.connect(self.viewPrev)
        btnViewNext.clicked.connect(self.viewNext)
        self.d0.addWidget(w0)
        # Shimadzu 1 
        self.imv1 = pg.ImageView()  # moving
        self.d1.addWidget(self.imv1)
        self.imv11 = pg.ImageView()  # moving
        self.d11.addWidget(self.imv11)
        self.imv111 = pg.ImageView()  # moving
        self.d111.addWidget(self.imv111)
        # Shimadzu 2
        self.imv2 = pg.ImageView()  
        self.d2.addWidget(self.imv2)
        self.imv22 = pg.ImageView()  
        self.d22.addWidget(self.imv22)
        self.imv222 = pg.ImageView()  
        self.d222.addWidget(self.imv222)
        # Shimadzu combined
        self.imv3 = pg.ImageView()  
        self.d3.addWidget(self.imv3)
        self.imv33 = pg.ImageView()  
        self.d33.addWidget(self.imv33)
        self.imv333 = pg.ImageView()  
        self.d333.addWidget(self.imv333)

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

        # setup zmq data reciever for other data
        self.queue = deque(maxlen=2)    # save just last data from XTD Shimadzu bridge .... may cause problems
        self.acquire = Thread(target=request, args=(self.queue, interfaceSlave))
        self.acquire.daemon = True
        self.acquire.start()



    def updatePlot(self, dataHPVX, dataSlave):
        """
        plot updating after data from the main interface are emitted - 'interfaceHPVX'
        """
        imgs_Shim1, imgs_Shim2 = None, None
        # raw data
        if device_name['shimadzu2'] in dataHPVX.keys(): # 
            imgs_Shim1 = np.array(dataHPVX[device_name['shimadzu2']][device_prop['camera']]*1.)            
            self.imv1.setImage(imgs_Shim1)  
            self.current1 = imgs_Shim1 
        if dataSlave is not None:   # 
            imgs_Shim2 = np.array( dataSlave[device_name['shimadzu1']][device_prop['camera']]*1. )  
            self.imv2.setImage(imgs_Shim2)
            self.current2 = imgs_Shim2
        # print(imgs_Shim1 is not None and imgs_Shim2 is not None)
        if imgs_Shim1 is not None and imgs_Shim2 is not None:
            imgs_combined = self.combine_Shimadzus(Sh1=imgs_Shim1, Sh2=imgs_Shim2)
            # print(type(imgs_combined))
            if imgs_combined is not None:
                self.imv3.setImage(imgs_combined)   
        QtGui.QApplication.processEvents()

    def update(self, d):
        """ called by bridgeClientShimadzu when bridge receives Shimadzu data
        d = data, meta
        """
        data, meta = d
        slaveData = None
        if len(self.queue)>0:
            slaveData, slaveMeta = self.getDataFromQueue()
        self.updatePlot(data, slaveData)  

    def getDataFromQueue(self):
        data, meta = self.queue.pop()
        return data, meta

    # normalis. & phase retr.
    def normalise_imgs(self, imgs_Shim1 = None, imgs_Shim2=None):
        """
        normalise images using dynamic flat-field correction algorithm
        """
        imgs_Shim1_normalised, imgs_Shim2_normalised = None, None
        if imgs_Shim1 is not  None:
            imgs_Shim1_normalised = dffc_correct(imgs_Shim1, pca_info_cam1, ds_parameter, x0_last=w0_last)
        if imgs_Shim2 is not  None:
            imgs_Shim2_normalised = dffc_correct(imgs_Shim2, pca_info_cam1, ds_parameter, x0_last=w0_last)
        return imgs_Shim1_normalised, imgs_Shim2_normalised 

    def phase_retrive_imgs(self, imgs_Shim1_normalised=None, imgs_Shim2_normalised=None):
        """
        using ADMMCTF for phase retrieval
        """
        imgs_Shim1_phase, imgs_Shim2_phase = None, None
        if imgs_Shim1_normalised is not  None:
            tid1, n, m = imgs_Shim1_normalised.shape
            imgs_Shim1_phase = np.zeros(imgs_Shim1_normalised.shape)
            for tid1_i in range(0,tid1,100):
                img = imgs_Shim1_normalised[tid1_i,:,:]
                mask =  np.zeros(img.shape)
                mask[:,:] = 1
                ks = ADMMCTF.kernel_grad().shape[0]-1  # size of the gradient kernel
                # Padding image
                b = np.pad(img, [ks, ks], mode='edge')
                # FPSF(Fourier transformed of the PSF)
                FPSF = []
                imgs_Shim1_phase[tid1_i,:,:] = ADMMCTF.admm_ctf_betaoverdelta( b, niter, eps, eta, tau, phys, mask, wvl, D, pxs, betaoverdelta, FPSF)
        if imgs_Shim2_normalised is not  None:
            tid2, n, m = imgs_Shim2_normalised.shape
            imgs_Shim2_phase = np.zeros(imgs_Shim2_normalised.shape)
            for tid2_i in range(0,tid2,100):
                img = imgs_Shim2_normalised[tid2_i,:,:]
                mask =  np.zeros(img.shape)
                mask[:,:] = 1
                ks = ADMMCTF.kernel_grad().shape[0]-1  # size of the gradient kernel
                # Padding image
                b = np.pad(img, [ks, ks], mode='edge')
                # FPSF(Fourier transformed of the PSF)
                FPSF = []
                imgs_Shim2_phase[tid2_i,:,:] = ADMMCTF.admm_ctf_betaoverdelta( b, niter, eps, eta, tau, phys, mask, wvl, D, pxs, betaoverdelta, FPSF)
        return imgs_Shim1_phase, imgs_Shim2_phase

    def combine_Shimadzus(self, Sh1=None ,Sh2=None):
        """
        combine two arrays into one 
        - firstframe from first camera, then second etc.  
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

    # button fun
    def saveDF(self):
        """
        save currently imaged data for postprocessing - flat-field normalisation, phase retrieval 
        """
        print("Saving and processing current data...")        
        imgs1_normalised, imgs2_normalised = self.normalise_imgs(imgs_Shim1 = self.current1, imgs_Shim2 = self.current2)
        imgs1_phase, imgs2_phase = self.phase_retrive_imgs(imgs_Shim1_normalised=imgs1_normalised, imgs_Shim2_normalised=imgs2_normalised)

        if imgs1_normalised is not None:
            newIdx1 = self.df1.shape[0]
            self.df1.loc[newIdx1] = {"normalised": imgs1_normalised, 'phase': imgs1_phase}
            if newIdx1 > self.max_len:   
                self.df1 = self.df1[int(self.max_len/2):].copy()

        if imgs2_normalised is not None:
            newIdx2 = self.df2.shape[0]
            self.df2.loc[newIdx2] = {"normalised": imgs2_normalised, 'phase': imgs2_phase}
            if newIdx2 > self.max_len:  
                self.df2 = self.df2[int(self.max_len/2):].copy()
        
        if imgs1_normalised is not None or imgs2_normalised is not None:
            imgs3_normalised = self.combine_Shimadzus(Sh1=imgs1_normalised, Sh2=imgs2_normalised)
            imgs3_phase = self.combine_Shimadzus(Sh1=imgs1_phase, Sh2=imgs2_phase)
            newIdx3 = self.df3.shape[0]
            self.df3.loc[newIdx3] = {"normalised": imgs3_normalised, 'phase': imgs3_phase}
            if newIdx3 > self.max_len:   
                self.df3 = self.df3[int(self.max_len/2):].copy()        
        print('Saving finished.')

    def viewPrev(self):
        """
        view previousl data from the saved ones
        """
        if self.idx_viewProc > 0:
            self.idx_viewProc -= 1
        self.viewProc()
        print('Index of viewed processed data: ', self.idx_viewProc)

    def viewNext(self):
        """
        view next data from the saved ones
        """
        if self.idx_viewProc != self.df1.shape[0]-1:
            self.idx_viewProc += 1
        else: 
            self.idx_viewProc = 0
        self.viewProc()
        print('Index of viewed processed data: ', self.idx_viewProc)

    #.................
    def viewProc(self):
        """
        mage processed data 
        """
        if self.df1.shape[0] > 0:
            if self.df1.normalised[self.idx_viewProc]  is not None:
                img1 = np.array(self.df1.normalised[self.idx_viewProc])
                self.imv11.setImage(img1)
            if self.df1.phase[self.idx_viewProc] is not None:
                img2 = np.array(self.df1.phase[self.idx_viewProc])
                self.imv111.setImage(img2)  
        if self.df2.shape[0] > 0:
            if self.df2.normalised[self.idx_viewProc]  is not None:
                img2n = np.array(self.df2.normalised[self.idx_viewProc])
                self.imv22.setImage(img2n)    
            if self.df2.phase[self.idx_viewProc] is not None:
                img2p = np.array(self.df2.phase[self.idx_viewProc])
                self.imv222.setImage()  
        if self.df3.shape[0] > 0:
            if self.df3.normalised[self.idx_viewProc]  is not None:     
                img3n = np.array(self.df3.normalised[self.idx_viewProc])      
                self.imv33.setImage(img3n)    
            if self.df3.phase[self.idx_viewProc] is not None:
                img3p = np.array(self.df3.phase[self.idx_viewProc])
                self.imv333.setImage(img3p)


def request(queue, interface):
    ''' data requester to run on another thread, this keeps requesting data in the background and filling up the ques from the right, while data is removed from the left...
    '''
    client = Client(interface)
    while True:
        data = client.next()
        queue.append(data)


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
