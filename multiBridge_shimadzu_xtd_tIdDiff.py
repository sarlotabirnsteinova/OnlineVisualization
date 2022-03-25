#!/gpfs/exfel/sw/software/xfel_anaconda3/1.1/bin/python
"""
online device
    
    combines data from 2 bridges
    shimadzu on one bridge and everything else on another

    - bridge client listens to XTD shimadzu bridge then emits data signal
    - data from second bridge is stored in queue buffer
    - data is corelated by trainIds (in self.update) then sent to self.updatePlot 
    
Plot trainId difference with respect to incomming Shimadzu XTD data
Plot both cameras, and mean across the buffer, and mean image intensity
"""
from PyQt5 import QtGui
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot, QTimer
from PyQt5.QtWidgets import QCheckBox
import pyqtgraph as pg
from pyqtgraph.dockarea import *
import sys
import time
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from karabo_bridge import Client
from collections import deque
from threading import Thread


# interfaceHPVX = 'tcp://10.253.0.52:54446'  # for Shimadzu SPB  and this is main.... 
# interfaceSlave = 'tcp://10.253.0.52:54445' # for Shimadzu XTD ....

interfaceSlave = 'tcp://10.253.0.52:54446'  # for Shimadzu SPB  .... 
interfaceHPVX = 'tcp://10.253.0.52:54445' # for Shimadzu XTD and this is main ....

## device, key pairs
deviceHPVX = ['SA1_XTD9_IMGPII45/CAM/BEAMVIEW_SHIMADZU:output','data.image.data']  #XTD Shimadzu
# deviceHPVX = ['SPB_EHD_HPVX2_1/CAM/CAMERA:output','data.image.data']  # SPB Shimadzu

# list of deviceIds keys on otherBridge
# deviceSlave = ['SA1_XTD9_IMGPII45/CAM/BEAMVIEW_SHIMADZU:output', 'data.image.data'] #XTD Shimadzu
deviceSlave = ['SPB_EHD_HPVX2_1/CAM/CAMERA:output','data.image.data'] # SPB Shimadzu


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
            time.sleep(0.01)


class imageView(QtGui.QMainWindow):
    def __init__(self, parent=None, title=None):
        super().__init__(parent)
        
        ## gui init
        self.widget = QtGui.QWidget()
        self.area = DockArea()
        self.setCentralWidget(self.area)
        self.resize(1600,1800)
        self.setWindowTitle('Shimadzu Camera View')
        #
        #
        self.ds = Dock("Image Shimadzu SPB", size=(800,800), closable=True)
        self.dx = Dock("Image Shimadzu XTD", size=(800,800), closable=True)
        self.d11 = Dock("TrainId difference" , size=(1600,300), closable=True)
        self.ds_mean = Dock("SPB mean of memory cells", size=(800,300), closable=True)
        self.dx_mean = Dock("XTD mean of memory cells", size=(800,300), closable=True)
        self.ds_mean_all = Dock("SPB mean of images", size=(800,300), closable=True)
        self.dx_mean_all = Dock("XTD mean of images", size=(800,300), closable=True)

        self.area.addDock(self.d11,'left')
        self.area.addDock(self.ds,'bottom',self.d11)
        self.area.addDock(self.dx,'right',self.ds)
        self.area.addDock(self.ds_mean,'bottom',self.ds)
        self.area.addDock(self.dx_mean,'bottom',self.dx)
        self.area.addDock(self.ds_mean_all,'bottom',self.ds_mean)
        self.area.addDock(self.dx_mean_all,'bottom',self.dx_mean)
        # self.area.addDock(self.d22,'bottom',self.d2)

        
        self.imv_spb = pg.ImageView()
        self.ds.addWidget(self.imv_spb)
        self.imv_xtd = pg.ImageView()
        self.dx.addWidget(self.imv_xtd)

        self.scat_w11 = pg.PlotWidget()
        self.scat_plot11 = pg.ScatterPlotItem(size=10, pen=pg.mkPen(None), brush=pg.mkBrush(255,255,255,120))
        self.scat_w11.addItem(self.scat_plot11)
        self.scat_w11.showGrid(x=True,y=True)
        self.d11.addWidget(self.scat_w11)

        self.mean_s = pg.PlotWidget()
        self.mean_spb = pg.ScatterPlotItem(size=8, pen=pg.mkPen(None), brush=pg.mkBrush(255,255,255,120))
        self.mean_s.addItem(self.mean_spb)
        self.mean_s.showGrid(x=True,y=True)
        self.ds_mean.addWidget(self.mean_s)

        self.mean_x = pg.PlotWidget()
        self.mean_xtd = pg.ScatterPlotItem(size=8, pen=pg.mkPen(None), brush=pg.mkBrush(255,255,255,120))
        self.mean_x.addItem(self.mean_xtd)
        self.mean_x.showGrid(x=True,y=True)
        self.dx_mean.addWidget(self.mean_x)

        self.mean_ss = pg.PlotWidget()
        self.mean_spb_all = pg.ScatterPlotItem(size=8, pen=pg.mkPen(None), brush=pg.mkBrush(255,255,255,120))
        self.mean_ss.addItem(self.mean_spb_all)
        self.mean_ss.showGrid(x=True,y=True)
        self.ds_mean_all.addWidget(self.mean_ss)

        self.mean_xx = pg.PlotWidget()
        self.mean_xtd_all = pg.ScatterPlotItem(size=8, pen=pg.mkPen(None), brush=pg.mkBrush(255,255,255,120))
        self.mean_xx.addItem(self.mean_xtd_all)
        self.mean_xx.showGrid(x=True,y=True)
        self.dx_mean_all.addWidget(self.mean_xx)

        max_len_avg = 800
        self.count = deque(maxlen=max_len_avg)
        self.tid_diff = deque(maxlen=max_len_avg)
        self.i = 0
        self.i_spb = 0
        self.mean_im_s = deque(maxlen=max_len_avg) 
        self.mean_im_x = deque(maxlen=max_len_avg)
        self.mean_all_s = deque(maxlen=max_len_avg)
        self.mean_all_x = deque(maxlen=max_len_avg) 
        self.prev_mean_s = deque(maxlen=1)
        self.prev_mean_x = deque(maxlen=1)  
 
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
        self.queue = deque(maxlen=10)    # ulozit poslednych 10 hodnot z hruheho interfacu: XTD kamera?
        self.acquire = Thread(target=request, args=(self.queue, interfaceSlave))
        self.acquire.daemon = True
        self.acquire.start()



    def updatePlot(self, dataHPVX, dataSlave):
            
            # if deviceId_name['motor1_4'] in dataSlave.keys():
            #     valueData_m14 = dataSlave[deviceId_name['motor1_4']][deviceId_property['motor']] 
            #     self.m4_que.append(valueData_m14)

            if (deviceHPVX[0] in dataHPVX.keys()):# and (self.devProperty in data[self.deviceId].keys()):
                # plot mean of shimadzu
                imageData1 = dataHPVX[deviceHPVX[0]][deviceHPVX[1]]*1. # mean image of 128 buffers
                if (len(imageData1.shape)==3):
                    self.imv_xtd.setImage(imageData1,xvals=np.linspace(1.,float(imageData1.shape[0]),imageData1.shape[0]))
                    # self.imv_xtd.play(rate=20) # rate in FPS

                else:
                    pass
                    #self.imv_xtd.setImage(imageData1)
                #print('xtd', imageData2.shape)
                if len(imageData1.shape)==3:
                    imageData1_mean = imageData1.mean(axis=(1,2))
                    self.mean_xtd.clear()
                    if len(self.prev_mean_x)>0:
                        #plot aj stare 
                        x,y = self.prev_mean_x.pop()
                        #for prev in l:
                        self.mean_xtd.addPoints( x,y, size=8, pen=pg.mkPen(None), brush=pg.mkBrush(255,255,255,180))
                    self.mean_xtd.addPoints(range(0,imageData1.shape[0]),imageData1_mean,size=8, pen=pg.mkPen(None), brush=pg.mkBrush(126,227,6,250))
                    self.prev_mean_x.append((range(0,imageData1.shape[0]),imageData1_mean)) 
                    
                    
                self.mean_all_x.append(imageData1.mean())   # plot mean intensity of all images for one trainId
                self.mean_xtd_all.clear()
                self.mean_xtd_all.addPoints(range(0,len(self.mean_all_x)),self.mean_all_x)
                self.i += 1

            if dataSlave is not None:
                imageData2 = dataSlave[deviceSlave[0]][deviceSlave[1]]*1. # mean image of 128 buffers
                if (len(imageData2.shape)==3):
                    self.imv_spb.setImage(imageData2,xvals=np.linspace(1.,float(imageData2.shape[0]),imageData2.shape[0]))
                    # self.imv_spb.play(rate=20) # rate in FPS

                else: 
                    pass
                    #self.imv_spb.setImage(imageData2)
                if len(imageData2.shape)==3:
                    imageData2_mean = imageData2.mean(axis=(1,2))
                    self.mean_spb.clear()
                    if len(self.prev_mean_s)>0:
                        #plot aj stare 
                        x,y = self.prev_mean_s.pop()
                        #for prev in l:
                        self.mean_spb.addPoints( x,y, size=8, pen=pg.mkPen(None), brush=pg.mkBrush(255,255,255,180))
                    self.mean_spb.addPoints(range(0,imageData2.shape[0]),imageData2_mean,size=8, pen=pg.mkPen(None), brush=pg.mkBrush(126,227,6,250))
                    self.prev_mean_s.append((range(0,imageData2.shape[0]),imageData2_mean)) 
                #print('spb', imageData1.shape,imageData1_mean.shape)
                # self.shimadzu_que.append(imageData1_mean)
                self.mean_all_s.append(imageData2.mean())   # plot mean intensity of all images for one trainId
                self.mean_spb_all.clear()
                self.mean_spb_all.addPoints(range(0,len(self.mean_all_s)),self.mean_all_s)
                self.i_spb += 1  

            
            if len(self.tid_diff)>0:
                self.scat_plot11.clear()
                self.scat_plot11.addPoints(self.tid_diff)
            #self.scat_plot11.addPoints([{'pos':(self.m4_que[-1],self.shimadzu_que[-1]), 'brush':pg.intColor(0,100)}])

            QtGui.QApplication.processEvents()
            # print(self.imv.roi.pos(), self.imv.roi.size()) # access to ROI coords

    def update(self, d):
        """ called by bridgeClientShimadzu when bridge receives Shimadzu data
        d = data, meta
        """
        data, meta = d
        trainId = meta[deviceHPVX[0]]['timestamp.tid']*1.   # now XTD Shimadzu
        slaveData = None   
        slaveTrainId = None
        if len(self.queue)>0:  # new SPB Shimadzu data?
            slaveData, slaveMeta = self.getDataFromQueue(trainId)
            slaveTrainId = slaveMeta[deviceSlave[0]]['timestamp.tid']*1.
            # add tid difference
            #self.count.append(self.i+1)
            difference  = trainId - slaveTrainId  # XTD - SPB
            #self.tid_diff.append(difference)
            if difference==0: 
                self.tid_diff.append({'pos':(self.i_spb,difference), 'brush':pg.mkBrush(87, 236, 6,240)})
            else:
                self.tid_diff.append({'pos':(self.i_spb,difference), 'brush':pg.mkBrush(233, 38, 4,240)})
        self.updatePlot(data, slaveData)

    
    def getDataFromQueue(self, trainId):

        data, meta = self.queue.pop()
        # slaveTrainId = meta[deviceSlave[0][0]]['timestamp.tid'] # might stay as it is...does not matter probably
        # if  slaveTrainId > trainId:   # SPB Shimadzu produkuje viac dat ako XTD?
        #     print('Missing XTD Shimadzu data')
        #     print('Shimadzu XTD trainId :{}  (oldest)'.format(meta[deviceSlave[0][0]]['timestamp.tid']),'Shimadzu SPB trainId :{}'.format(trainId))
        #     # print('Shimadzu XTD trainId :{}  (oldest)'.format(meta[deviceSlave[0][0]]['timestamp.tid']))
        #     return None, None
        # else:
        #     while slaveTrainId < trainId:
        #         print('Shimadzu XTD data: ', slaveTrainId, 'dropped, SPB trainId: ',trainId,' trainId difference: ',trainId - slaveTrainId)
        #         if len(self.queue)<=0:
        #             print('....')
        #             return None, None
        #         data, meta = self.queue.popleft() # bude vyhadzovat z queue zaradom 'starsie' data
        #         slaveTrainId = meta[deviceSlave[0][0]]['timestamp.tid']
        #     # tu moze najst aj najblizsie vacsie trainId ...ak toto   
        #     if slaveTrainId != trainId:
        #         print('There is no data with the same trainId for Shimadzu in SPB and XTD.')
        #         print('The next closest Shimadzu XTD trainId: ',slaveTrainId,', Shimadzu SPB trainId: ',trainId) 
        #         return None, None
        #     # 
        #     # print('Shimadzu XTD trainId: ',slaveTrainId,', Shimadzu SPB trainId: ',trainId) # tie by sa mali rovnat
        # print('___Yeey!!!___')
        return data, meta



    def threadFin(self):
        pass
    
    def closeEvent(self,event):
        # stop thread here... weird on macos
        event.accept()


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
