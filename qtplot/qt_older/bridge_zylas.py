#!/gpfs/exfel/sw/software/xfel_anaconda3/1.1/bin/python
"""
online device
    uses Qthread to fill a queue of karabo BRIDGE data

    plots all 3 Zyla cameras, from one bridge which contains 3 Zyla cameras (not neccessary all 3)
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

#interface = 'tcp://10.253.0.52:22500'
interface = 'tcp://10.253.0.52:22345'  # huge bridge with everything in it .... all Zyla camers

# for actual karabo_bridge
## device, key pairs ... dopisat motory, trigger a standby veci, xgms... \
deviceId_name = {'zyla2':'SPB_EXP_ZYLA/CAM/2:output','motor1_8':'SPB_EXP_KOHZU-1/MOTOR/ARIES_8','motor1_7':'SPB_EXP_KOHZU-1/MOTOR/ARIES_7','motor1_6':'SPB_EXP_KOHZU-1/MOTOR/ARIES_6','motor1_5':'SPB_EXP_KOHZU-1/MOTOR/ARIES_5','motor1_4':'SPB_EXP_KOHZU-1/MOTOR/ARIES_4','motor1_3':'SPB_EXP_KOHZU-1/MOTOR/ARIES_3','motor1_2':'SPB_EXP_KOHZU-1/MOTOR/ARIES_2','zyla1':'SPB_EXP_ZYLA/CAM/1:output','zyla3':'SPB_EXP_ZYLA/CAM/3:output','shimadzu2':'SPB_EHD_HPVX2_2/CAM/CAMERA:output','xgm9':'SPB_XTD9_XGM/XGM/DOOCS:output','xgm2':'SA1_XTD2_XGM/XGM/DOOCS:output','standby1':'SPB_EHD_HPVX2_1/TSYS/STBY','standby2':'SPB_EHD_HPVX2_2/TSYS/STBY','triger2':'SPB_EHD_HPVX2_2/TSYS/TRIG', 'triger1':'SPB_EHD_HPVX2_1/TSYS/TRIG','motor1_1':'SPB_EXP_KOHZU-1/MOTOR/ARIES_1'}
#{'SPB_EXP_ZYLA/CAM/1:output':'data.image.data','SPB_EXP_ZYLA/CAM/1:output':'data.image.data', 'SPB_EHD_HPVX2_2/CAM/CAMERA:output':'data.image.data'}
#'SA1_XTD2_XGM/XGM/DOOCS:output':'data.intensityTD', 'SPB_XTD9_XGM/XGM/DOOCS:output':'data.intensityTD'
#'SPB_EHD_HPVX2_1/TSYS/STBY':'actualDelay.value','SPB_EHD_HPVX2_1/TSYS/TRIG':'actualDelay.value','SPB_EHD_HPVX2_2/TSYS/TRIG':'actualDelay.value','SPB_EHD_HPVX2_2/TSYS/TRIG':'actualDelay.value'
#'SPB_EXP_KOHZU-1/MOTOR/ARIES_1':'actualPosition.value','SPB_EXP_KOHZU-1/MOTOR/ARIES_2':'actualPosition.value','SPB_EXP_KOHZU-1/MOTOR/ARIES_3':'actualPosition.value','SPB_EXP_KOHZU-1/MOTOR/ARIES_4':'actualPosition.value'
#'SPB_EXP_KOHZU-1/MOTOR/ARIES_5':'actualPosition.value','SPB_EXP_KOHZU-1/MOTOR/ARIES_6':'actualPosition.value','SPB_EXP_KOHZU-1/MOTOR/ARIES_7':'actualPosition.value','SPB_EXP_KOHZU-1/MOTOR/ARIES_8':'actualPosition.value'
deviceId_property = {'camera': 'data.image.data', 'xgm':'data.intensityTD', 'motor': 'actualPosition.value','delay':'actualDelay.value'}
#povodne# 
#deviceId = ['SPB_EHD_HPVX2_2/CAM/CAMERA:output','data.image.data']


class bridgeClient(QThread):
    newDataSignal =  pyqtSignal(object)
    def __init__(self, interface):
        super().__init__()
        self.client = Client(interface, timeout = 5)
        try: 
            data, meta = self.client.next()
            print( data.keys() )
        except TimeoutError as e:
            pass
            # print('bridge timeout...')
        #self.newDataSignal =  pyqtSignal(object)
    def __del__(self):
        self.wait()
    def run(self):
        print('starting queue thread')
        while True:
            try: 
                data, meta = self.client.next()
                self.newDataSignal.emit(data)
            except TimeoutError as e:
                #print('bridge timeout...')
                data = None
            time.sleep(0.5)


class imageView(QtGui.QMainWindow):
    def __init__(self, interface,  parent=None, title=None):
        super().__init__(parent)
        
        ## gui init
        self.widget = QtGui.QWidget()
        self.area = DockArea()
        self.setCentralWidget(self.area)
        self.resize(1600,1200)
        self.setWindowTitle('Zyla Cameras View')

        self.d1 = Dock("Image Zyla 1", size=(800,800), closable=True)
        self.d2 = Dock("Image Zyla 2", size=(800,800), closable=True)
        self.d3 = Dock("Image Zyla 3", size=(800,800), closable=True)

        # self.d5 = Dock("Total Measured Mean (Zyla 1) vs. Motor 4 pos.", size=(800,400))
        # self.d6 = Dock("Total Measured Mean (Zyla 3) vs. Motor 4 pos.", size=(800,400))
        # self.d1 = Dock("Mean of Zyla 1 vs. Motor 4 pos.", size=(800,400))
        # self.d2 = Dock("Mean of Zyla 3 vs. Motor 4 pos. ", size=(800,400))

        self.area.addDock(self.d1,'left')
        self.area.addDock(self.d2,'right')        
        self.area.addDock(self.d3,'right')
        # self.area.addDock(self.d2,'bottom',self.d3)
        # self.area.addDock(self.d6,'bottom',self.d3)

        # self.scat_w1 = pg.PlotWidget()
        # self.scat_plot1 = pg.ScatterPlotItem(size=10, pen=pg.mkPen(None), brush=pg.mkBrush(255,255,255,120))
        # self.scat_w1.addItem(self.scat_plot1)
        # self.d1.addWidget(self.scat_w1)
        # #
        # self.scat_w2 = pg.PlotWidget()
        # self.scat_plot2 = pg.ScatterPlotItem(size=10, pen=pg.mkPen(None), brush=pg.mkBrush(255,255,255,120))
        # self.scat_w2.addItem(self.scat_plot2)
        # self.d2.addWidget(self.scat_w2)
        # #
        # self.scat_w3 = pg.PlotWidget()
        # self.scat_plot3 = pg.ScatterPlotItem(size=10, pen=pg.mkPen(None), brush=pg.mkBrush(255,255,255,120))
        # self.scat_w3.addItem(self.scat_plot3)
        # self.d5.addWidget(self.scat_w3)
        # #
        # self.scat_w4 = pg.PlotWidget()
        # self.scat_plot4 = pg.ScatterPlotItem(size=10, pen=pg.mkPen(None), brush=pg.mkBrush(255,255,255,120))
        # self.scat_w4.addItem(self.scat_plot4)
        # self.d6.addWidget(self.scat_w4)
        #
        self.imv1 = pg.ImageView()  # moving
        self.d1.addWidget(self.imv1)
        #
        self.imv2 = pg.ImageView()  # moving
        self.d2.addWidget(self.imv2)
        #
        self.imv3 = pg.ImageView()  # moving
        self.d3.addWidget(self.imv3)
        #
        max_len_avg = 300	
        # self.m4_que = deque(maxlen=max_len_avg)
        # self.m2_que = deque(maxlen=max_len_avg)
        self.zyla1_que = deque(maxlen=max_len_avg)
        self.zyla2_que = deque(maxlen=max_len_avg)
        self.zyla3_que = deque(maxlen=max_len_avg)
        # self.plot_config = {'autoRange': False, 
        #                         'autoLevels': False, 
        #                         'autoHistogramRange': False,
        #                          'axes' : {'x':0, 'y':1},
        #                          'clear' : True
        #                          }
        self.timeVals = np.arange(128) # buffer index for train plot 
        # self.imv_config = {'autoRange': False, 
        #                     'autoLevels': False, 
        #                     'autoHistogramRange': False,
        #                     'axes' : {'x':0, 'y':1}
        #                     }
        # self.imv1_config = {'autoRange': False,
        #                     'autoLevels': False,
        #                     'autoHistogramRange': False,
        #                     'axes' : {'t':0, 'x':1, 'y':2},
        #                     'xvals' : self.timeVals
        #                     }
        self.show()
        # 
        self.clientThread = bridgeClient(interface)
        self.clientThread.start()
        self.clientThread.finished.connect(self.threadFin)
        self.clientThread.newDataSignal.connect( self.update)



    def update(self, data):
        #print(data.keys())

        if deviceId_name['zyla1'] in data.keys():
            imageData1 = np.array( data[deviceId_name['zyla1']][deviceId_property['camera']]*1. )
            sum_imageData1 = np.mean(imageData1)
            self.zyla1_que.append(sum_imageData1)
            self.imv1.setImage(imageData1)


        if deviceId_name['zyla2'] in data.keys():
            imageData2 = np.array( data[deviceId_name['zyla2']][deviceId_property['camera']]*1. )
            sum_imageData2 = np.mean(imageData2)
            self.zyla2_que.append(sum_imageData2)
            self.imv2.setImage(imageData2)


        if deviceId_name['zyla3'] in data.keys():
            imageData3 = np.array( data[deviceId_name['zyla3']][deviceId_property['camera']]*1. ) 
            # lineData_cam = np.mean( data[deviceId_i][devProperty_i], axis=(1,2))
            sum_imageData3 = np.mean(imageData3)
            self.zyla3_que.append(sum_imageData3)
            self.imv3.setImage(imageData3)

        # self.imv.roi.pos(), self.imv.roi.size() # access to ROI coords
        # #self.plt.plot(lineData, **self.plot_config)
        # self.scat_plot1.clear()
        # self.scat_plot1.addPoints(self.m4_que,self.zyla1_que)
        # self.scat_plot1.addPoints([{'pos':(self.m4_que[-1],self.zyla1_que[-1]), 'brush':pg.intColor(0,100)}])
        # #
        # self.scat_plot2.clear()
        # self.scat_plot2.addPoints(self.m4_que,self.zyla3_que)
        # self.scat_plot2.addPoints([{'pos':(self.m4_que[-1],self.zyla3_que[-1]), 'brush':pg.intColor(0,100)}])
        # #

        #self.imv1.play(rate=20) # rate in FPS
        QtGui.QApplication.processEvents()


    def threadFin(self):
        pass
    
    def closeEvent(self,event):
        # stop thread here... weird on macos?
        event.accept()

    # def setTranspose(self,state):
    #     self.transpose = state


def main():
    app = QtGui.QApplication(sys.argv)
    form = imageView(interface)
    form.show()
    # add timer to allow for catching ctrl+c :
    timer = QTimer()
    timer.timeout.connect(lambda: None)
    timer.start(100)
    # app.exec_()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()


