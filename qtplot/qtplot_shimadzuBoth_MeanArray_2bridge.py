#!/gpfs/exfel/sw/software/xfel_anaconda3/1.1/bin/python
"""
online device
    plots an image: means over frames from 1&2 Shimadzu updating as raws 
    combines data from 2 bridges
TODO 
    - add clear button

Parameters
----------
None
Returns
-------
plot
    updating image, scatter plot with number of updates  
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



interfaceHPVX = 'tcp://10.253.0.52:54456'  # for Shimadzu 2 ?
interfaceSlave = 'tcp://10.253.0.52:54333' # KOHZU bridge ... delays & Shimadzu 1

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
            time.sleep(0.01)


class imageView(QtGui.QMainWindow):
    def __init__(self, parent=None, title=None):
        super().__init__(parent)
        
       # parameters & variables
        self.del_que = deque(maxlen=10000)
        #
        self.df = pd.DataFrame(columns=['delay', 'cam1_meanBuff','cam1'],dtype=np.float64)
        self.df.cam1 = self.df.cam1.astype(object)
        self.max_len = 1000
        self.delay = 0

        ## gui init
        self.widget = QtGui.QWidget()
        self.area = DockArea()
        self.setCentralWidget(self.area)
        self.resize(800,1200)
        self.setWindowTitle('Shimadzu Camera View')

        self.d0 = Dock("", size=(100, 50))
        self.d1 = Dock("Delay: Shimadzu 1", size=(800,800))
        self.d11 = Dock("delay", size=(800,400), closable=True)

        self.area.addDock(self.d0, 'left')
        self.area.addDock(self.d1,'bottom',self.d0)
        self.area.addDock(self.d11,'bottom',self.d1)

        # buttons
        w0 = pg.LayoutWidget()
        btnHalf = QtGui.QPushButton('reduce df')
        btnSave = QtGui.QPushButton('save csv')

        w0.addWidget(btnHalf, row=0, col=0)
        w0.addWidget(btnSave, row=0, col=1)

        btnHalf.clicked.connect(self.halfDF)
        btnSave.clicked.connect(self.saveDF)
        self.d0.addWidget(w0)
        # 
        self.imv1 = pg.ImageView()  # moving
        self.d1.addWidget(self.imv1)
        #
        self.scat_w1 = pg.PlotWidget()
        self.scat_plot1 = pg.ScatterPlotItem(size=10, pen=pg.mkPen(None), brush=pg.mkBrush(255,255,255,120))
        self.scat_w1.addItem(self.scat_plot1)
        self.scat_w1.showGrid(x=True,y=True)
        self.d11.addWidget(self.scat_w1)
        #
        self.show()
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
        self.queue = deque(maxlen=1)    # save just last data from XTD Shimadzu bridge .... may cause problems
        self.acquire = Thread(target=request, args=(self.queue, interfaceSlave))
        self.acquire.daemon = True
        self.acquire.start()



    def updatePlot(self, dataHPVX, dataSlave):
            delay = None
            cam = None
            cam1Buff = None
            camBuff = None
            cam2Buff = None


            if (device_name['shimadzu1'] in dataHPVX.keys() ): 
                imageData1 = np.array(dataHPVX[device_name['shimadzu1']][device_prop['camera']]*1.)     # *1. makes a copy 
                if len(imageData1.shape)==3:
                    cam1Buff = np.mean(imageData1,axis=(1,2))
                else:
                    cam1Buff = np.mean(imageData1,axis=(1)) # bullshit, but does not crash
                self.delay += 1
                delay = self.delay

            if dataSlave is not None and cam1Buff is not None:  
                imageData2 = dataSlave[device_name['shimadzu2']][device_prop['camera']]*1.
                cam2Buff = np.mean(imageData2,axis=(1,2))
                camBuff = np.append(cam2Buff,cam1Buff,axis=0)
            
            if delay is not None and cam2Buff is not None:
                self.updateDF(delay,cam,camBuff)   ################################3
                # calculate
                df_sorted = self.df.sort_values(by='delay', ascending=False)
                d = df_sorted['cam1_meanBuff'].to_numpy()
                l, w = (d.shape[0],d[0].shape[0])
                data_arr = np.concatenate(d).reshape(l,w)
                # plot
                self.imv1.setImage(data_arr[1:,:].T)  # ommit first noisy frame
                # calculate delays
                mp = list(df_sorted['delay'].to_numpy())
                idx = np.arange(len(mp))
                # plot delays
                self.scat_plot1.clear()
                self.scat_plot1.addPoints(idx,mp, size=8, pen=pg.mkPen(None), brush=pg.mkBrush('w'))

               
            QtGui.QApplication.processEvents()
            # print(self.imv.roi.pos(), self.imv.roi.size()) # access to ROI coords

    def update(self, d):
        """ called by bridgeClientShimadzu when bridge receives Shimadzu data
        d = data, meta
        """
        data, meta = d
        slaveData = None
        slaveTrainId = None
        if len(self.queue)>0:
            slaveData, slaveMeta = self.getDataFromQueue()

            #     print("trainId difference = 0, for SPB trainId: ", trainId)
        self.updatePlot(data, slaveData)   # with data - SPB, and slaveData - XTD or None, if XTD is late or missing?

    
    def getDataFromQueue(self):

        data, meta = self.queue.pop()
        return data, meta
    
    def updateDF(self, delay, cam1, cam1_meanBuff):
        newData = {"delay": delay, 'cam1': cam1, 'cam1_meanBuff':cam1_meanBuff}
        newIdx = self.df.shape[0]
        self.df.loc[newIdx] = newData
        if self.df.shape[0] > self.max_len:   # if too long DataFrame
            self.df = self.df[int(self.max_len//2):].copy()


    def threadFin(self):
        pass
    
    def closeEvent(self,event):
        # stop thread here... weird on macos
        event.accept()

    # button fun
    def halfDF(self):
        if self.df.shape[0] > self.max_len:   # if too long DataFrame
            self.df = self.df[int(self.max_len//2):].copy()
        print('DataFrame halved in size.')

    def saveDF(self):
        outName = 'DF_delayBuffer1_{}.h5'.format(datetime.datetime.now().strftime('%Y%m%dT%H%M'))
        print('Saving as {}'.format(outName))
        # self.df.to_csv(outName)
        with h5py.File(outName, "w") as f:
            f.create_dataset('meanBuff', data = self.df.cam1_meanBuff.to_numpy())
            f.create_dataset('dels', data = self.df.delay.to_numpy())


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
