#!/gpfs/exfel/sw/software/xfel_anaconda3/1.1/bin/python
"""
online device
    mean over frame vs. delay
    combines data from 2 bridges

TODO 
    - add clear plot button
    - add parameter which Zyla ?  

Parameters
----------
None
 
Returns
-------
plot
    updating scatter plot

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



interfaceHPVX = 'tcp://10.253.0.51:45452'  # Zyla 4 
interfaceSlave = 'tcp://10.253.0.52:54333' # KOHZU bridge delays

## device, key pairs
deviceId_name = {'zyla5':'SPB_EXP_ZYLA/CAM/5:output','zyla3':'SPB_EXP_ZYLA/CAM/3:output','zyla4':'SPB_EXP_ZYLA/CAM/4:output','shimadzu2':'SPB_EHD_HPVX2_2/CAM/CAMERA:output','dg1':'SPB_EXP_SYS/DG/DG1','dg2':'SPB_EXP_SYS/DG/DG2'}
deviceId_property = {'camera': 'data.image.data', 'motor': 'actualPosition.value','delayG':'G.delay.value','delayA':'A.delay.value'}          


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
            time.sleep(0.5)


class imageView(QtGui.QMainWindow):
    def __init__(self, parent=None, title=None):
        super().__init__(parent)
        # parameters & variables
        self.del1_que = deque(maxlen=10000)
        self.cam_que = deque(maxlen=10000)
        # self.del2_que = deque(maxlen=10000)
        #
        self.df1 = pd.DataFrame(columns=['delay1', 'cam1_mean'],dtype=np.float64)
        # self.df2 = pd.DataFrame(columns=['delay2', 'cam2_meanBuff'],dtype=np.float64)
        self.max_len = 200000


        ## gui init
        self.widget = QtGui.QWidget()
        self.area = DockArea()
        self.setCentralWidget(self.area)
        self.resize(800,1200)
        self.setWindowTitle('Shimadzu Camera View')

        self.d0 = Dock("", size=(100, 50))
        self.d1 = Dock("Delay: Zyla", size=(800,800))
        self.d11 = Dock("delay 1", size=(800,400), closable=True)
        # self.d2 = Dock("Delay: Shimadzu 2", size=(800,800))
        # self.d22 = Dock("delay 2", size=(800,400), closable=True)

        self.area.addDock(self.d0, 'left')
        self.area.addDock(self.d1,'bottom',self.d0)
        self.area.addDock(self.d11,'bottom',self.d1)
        # self.area.addDock(self.d2,'right',self.d1)
        # self.area.addDock(self.d22,'right',self.d11)

        # buttons
        w0 = pg.LayoutWidget()
        # btnHalf = QtGui.QPushButton('reduce df')
        # btnSave = QtGui.QPushButton('save csv')

        # w0.addWidget(btnHalf, row=0, col=0)
        # w0.addWidget(btnSave, row=0, col=1)

        # btnHalf.clicked.connect(self.halfDF)
        # btnSave.clicked.connect(self.saveDF)
        self.d0.addWidget(w0)
        # 
        self.scat_w0 = pg.PlotWidget()
        self.scat_plot0 = pg.ScatterPlotItem(size=10, pen=pg.mkPen(None), brush=pg.mkBrush(255,255,255,120))
        self.scat_w0.addItem(self.scat_plot0)
        self.scat_w0.showGrid(x=True,y=True)
        self.d1.addWidget(self.scat_w0)
        #        # self.imv2 = pg.ImageView()  # moving
        # self.d2.addWidget(self.imv2)
        # #
        self.scat_w1 = pg.PlotWidget()
        self.scat_plot1 = pg.ScatterPlotItem(size=10, pen=pg.mkPen(None), brush=pg.mkBrush(255,255,255,120))
        self.scat_w1.addItem(self.scat_plot1)
        self.scat_w1.showGrid(x=True,y=True)
        self.d11.addWidget(self.scat_w1)
        #
        # self.scat_w2 = pg.PlotWidget()
        # self.scat_plot2 = pg.ScatterPlotItem(size=10, pen=pg.mkPen(None), brush=pg.mkBrush(255,255,255,120))
        # self.scat_w2.addItem(self.scat_plot2)
        # self.scat_w2.showGrid(x=True,y=True)
        # self.d22.addWidget(self.scat_w2)
        # self.show()
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
            delay1 = None
            cam1 = None
            # delay2 = None
            # cam2 = None



            if (deviceId_name['zyla4'] in dataHPVX.keys() ): # plot SPB Shimadzu
                # plot mean of shimadzu
                # print(dataHPVX.shape)
                imageData1 = dataHPVX[deviceId_name['zyla4']][deviceId_property['camera']]*1.     # Zyla now
                cam1 = np.mean(imageData1)
                self.cam_que.append(cam1) # ... 



            if dataSlave is not None:   # plot XTD Shimadzu
                #print(dataSlave.shape)
                delay1 = dataSlave[deviceId_name['dg2']][deviceId_property['delayG']]*1.
                self.del1_que.append(delay1) # ... 

                # delay2 = dataSlave[deviceSlave[2]][deviceSlave[3]]*1.
                # cam2 = np.array( dataSlave[deviceHPVX[2]][deviceHPVX[1]]*1. )
                # print(imageData1.shape)
                # cam2Buff = np.mean(cam2,axis=(1,2))
                #print('XTD shape: ',imageData2.shape)
            
            if delay1 is not None and cam1 is not None:
                self.updateDF1(delay1,cam1)
                # calculate
                df1_sorted = self.df1.sort_values(by='delay1', ascending=False)
                # calculate delay
                cm = list(df1_sorted['cam1_mean'].to_numpy())
                mp = list(df1_sorted['delay1'].to_numpy())
                idx = np.arange(len(mp))
                # plot delays
                self.scat_plot0.clear()
                self.scat_plot0.addPoints(mp,cm, size=8, pen=pg.mkPen(None), brush=pg.mkBrush('w'))
                self.scat_plot1.clear()
                self.scat_plot1.addPoints(idx,mp, size=8, pen=pg.mkPen(None), brush=pg.mkBrush('w'))
                # self.scat_plot1.addPoints([{'pos': (idx[-1], mp[-1]), 'brush':pg.intColor(0, 100)}])

            # if delay2 != None and len(cam2Buff) > 1:
            #     self.updateDF2(delay2,cam2Buff)
            #     # calculate
            #     df2_sorted = self.df2.sort_values(by='delay2', ascending=False)
            #     d = df2_sorted['cam2_meanBuff'].to_numpy()
            #     l, w = (d.shape[0],d[0].shape[0])
            #     data_arr2 = np.concatenate(d).reshape(l,w)
            #     # plot
            #     self.imv2.setImage(data_arr2)
            #     # calculate delays
            #     mp = list(df2_sorted['delay2'].to_numpy())
            #     idx = np.arange(len(mp))
            #     # plot delays
            #     self.scat_plot2.clear()
            #     self.scat_plot2.addPoints(idx,mp, size=8, pen=pg.mkPen(None), brush=pg.mkBrush('w'))
            #     # self.scat_plot2.addPoints([{'pos': (idx[-1], mp[-1]), 'brush':pg.intColor(0, 100)}])
            
            
               
            QtGui.QApplication.processEvents()
            # print(self.imv.roi.pos(), self.imv.roi.size()) # access to ROI coords

    def update(self, d):
        """ called by bridgeClientShimadzu when bridge receives Shimadzu data
        d = data, meta
        """
        data, meta = d
        # trainId = meta[deviceHPVX[0]]['timestamp.tid']*1. 
        slaveData = None
        # slaveTrainId = None
        if len(self.queue)>0:
            slaveData, slaveMeta = self.getDataFromQueue()
            # slaveTrainId = slaveMeta[deviceSlave[0]]['timestamp.tid']*1.

            #     print("trainId difference = 0, for SPB trainId: ", trainId)
        self.updatePlot(data, slaveData)   # with data - SPB, and slaveData - XTD or None, if XTD is late or missing?

    
    def getDataFromQueue(self):

        data, meta = self.queue.pop()
        return data, meta
    
    def updateDF1(self, delay1, cam1_mean):
        newData = {"delay1": delay1, 'cam1_mean':cam1_mean}
        newIdx = self.df1.shape[0]
        self.df1.loc[newIdx] = newData
        # if self.df.shape[0] > self.max_len:   # if too long DataFrame
        #     self.df = self.df[int(self.max_len/2):].copy()
    # def updateDF2(self, delay2, cam2_meanBuff):
    #     newData = {"delay2": delay2, 'cam2_meanBuff':cam2_meanBuff}
    #     newIdx = self.df2.shape[0]
    #     self.df2.loc[newIdx] = newData


    def threadFin(self):
        pass
    
    def closeEvent(self,event):
        # stop thread here... weird on macos
        event.accept()

    # # button fun
    # def halfDF(self):
    #     half_size = int(self.df1.shape[0]//2)
    #     if self.df1.shape[0] > self.max_len:   # if too long DataFrame
    #         self.df1 = self.df1[half_size:].copy()
    #     # if self.df2.shape[0] > self.max_len:   # if too long DataFrame
    #     #     self.df2 = self.df2[half_size:].copy()
    #     print('DataFrame halved in size.')

    # def saveDF(self):
    #     outName = 'DF_delayBuffer1_{}.csv'.format(datetime.datetime.now().strftime('%Y%m%dT%H%M'))
    #     print('Saving as {}'.format(outName))
    #     self.df1.to_csv(outName)
    #     # outName = 'DF_delayBuffer2_{}.csv'.format(datetime.datetime.now().strftime('%Y%m%dT%H%M'))
    #     # print('Saving as {}'.format(outName))
    #     # self.df2.to_csv(outName)

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
