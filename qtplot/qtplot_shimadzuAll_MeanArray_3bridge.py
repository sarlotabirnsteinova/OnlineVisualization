#!/gpfs/exfel/sw/software/xfel_anaconda3/1.1/bin/python
"""
online device
    plots an image: means over frames from all 3 Shimadzus updating as raws 
    combines data from 3 bridges
TODO 
    - test the script

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


interfaceSlave2 = 'tcp://10.253.0.52:54435'  # for Shimadzu 3 ?
interfaceHPVX = 'tcp://10.253.0.52:54456'  # for Shimadzu 1 ?
interfaceSlave = 'tcp://10.253.0.52:54333' # KOHZU bridge ... delays & Shimadzu 1

## device, key pairs
device_name = {'zyla5':'SPB_EXP_ZYLA/CAM/5:output','zyla3':'SPB_EXP_ZYLA/CAM/3:output','zyla4':'SPB_EXP_ZYLA/CAM/4:output','shimadzu2':'SPB_EHD_HPVX2_2/CAM/CAMERA:output','shimadzu1':'SPB_EHD_HPVX2_1/CAM/CAMERA:output','shimadzu3':'SPB_EHD_HPVX2_3/CAM/CAMERA:output','shimadzu1del':'SPB_EHD_HPVX2_1/TSYS/STBY','shimadzu2del':'SPB_EHD_HPVX2_2/TSYS/STBY','dg1':'SPB_EXP_SYS/DG/DG1','dg2':'SPB_EXP_SYS/DG/DG2'}
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
        self.df = pd.DataFrame(columns=['delay', 'cam_meanBuff'],dtype=np.float64)
        self.max_len = 1000
        self.delay = 0

        ## gui init
        self.widget = QtGui.QWidget()
        self.area = DockArea()
        self.setCentralWidget(self.area)
        self.resize(800,1200)
        self.setWindowTitle('Shimadzu Camera View')

        self.d0 = Dock("", size=(100, 50))
        self.d1 = Dock("", size=(800,800)) # image
        self.d11 = Dock("", size=(800,400), closable=True) # how many updates

        self.area.addDock(self.d0, 'left')
        self.area.addDock(self.d1,'bottom',self.d0)
        self.area.addDock(self.d11,'bottom',self.d1)

        # buttons
        w0 = pg.LayoutWidget()
        btnHalf = QtGui.QPushButton('clear df')

        w0.addWidget(btnHalf, row=0, col=0)

        btnHalf.clicked.connect(self.clearDF)
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
        self.queue = deque(maxlen=1)
        self.queue2 = deque(maxlen=1) 
        self.acquire = Thread(target=request, args=(self.queue,self.queue2, interfaceSlave, interfaceSlave2))
        self.acquire.daemon = True
        self.acquire.start()



    def updatePlot(self, dataHPVX, dataSlave, dataSlave2):
            delay = None
            camBuff = None
            cam1Buff = None
            cam2Buff = None
            cam3Buff = None


            if (device_name['shimadzu1'] in dataHPVX.keys() ): 
                print('shim1 here')
                imageData1 = np.array(dataHPVX[device_name['shimadzu1']][device_prop['camera']]*1.)     # *1. makes a copy 
                if len(imageData1.shape)==3:
                    cam1Buff = np.mean(imageData1,axis=(1,2))
                self.delay += 1
                delay = self.delay

            if dataSlave is not None and cam1Buff is not None:  
                print('shim2 here')
                imageData2 = np.array(dataSlave[device_name['shimadzu2']][device_prop['camera']]*1.)
                cam2Buff = np.mean(imageData2,axis=(1,2))
                camBuff = np.append(cam1Buff,cam2Buff,axis=0)


            if dataSlave2 is not None and cam1Buff is not None:  
                print('shim3 here')
                imageData3 = np.array(dataSlave2[device_name['shimadzu3']][device_prop['camera']]*1.)
                cam3Buff = np.mean(imageData3,axis=(1,2))
                camBuff = np.append(camBuff,cam3Buff,axis=0)

            if delay is not None and cam2Buff is not None:
                self.updateDF(delay,camBuff)   ################################
                # calculate
                df_sorted = self.df.sort_values(by='delay', ascending=False) # stupid but OK
                d = df_sorted['cam_meanBuff'].to_numpy()
                l, w = (d.shape[0],d[0].shape[0])
                data_arr = np.concatenate(d).reshape(l,w)
                # plot
                self.imv1.setImage(data_arr[:,:].T)  
                # calculate delays
                mp = list(df_sorted['delay'].to_numpy())
                idx = np.arange(len(mp))
                # plot delays
                self.scat_plot1.clear()
                self.scat_plot1.addPoints(idx,mp, size=8, pen=pg.mkPen(None), brush=pg.mkBrush('w'))

               
            QtGui.QApplication.processEvents()

    def update(self, d):
        """ called by bridgeClientShimadzu when bridge receives Shimadzu data
        d = data, meta
        """
        data, meta = d
        slaveData = None
        slaveData2 = None
        if len(self.queue)>0:
            slaveData, slaveMeta = self.getDataFromQueue()
        if len(self.queue2)>0:
            slaveData2, slaveMeta2 = self.getDataFromQueue2()
            #     print("trainId difference = 0, for SPB trainId: ", trainId)
        self.updatePlot(data, slaveData, slaveData2)   

    
    def getDataFromQueue(self):
        data, meta = self.queue.pop()
        return data, meta

    def getDataFromQueue2(self):
        data, meta = self.queue2.pop()
        return data, meta
    
    def updateDF(self, delay, meanBuff):
        newData = {'delay': delay, 'cam_meanBuff':meanBuff}
        newIdx = self.df.shape[0]
        self.df.loc[newIdx] = newData



    def threadFin(self):
        pass
    
    def closeEvent(self,event):
        # stop thread here... weird on macos
        event.accept()

    # button fun
    def clearDF(self):
        self.df = pd.DataFrame(columns=['delay','cam_meanBuff'],dtype=np.float64)
        print('DataFrame cleared.')


def request(queue,queue2, interface,interface2):
    ''' data requester to run on another thread, this keeps requesting data in the background and filling up the ques from the right, while data is removed from the left...
    '''
    client = Client(interface)
    client2 = Client(interface2)
    while True:
        data = client.next()
        data2 = client2.next()
        queue.append(data)
        queue2.append(data2)


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
