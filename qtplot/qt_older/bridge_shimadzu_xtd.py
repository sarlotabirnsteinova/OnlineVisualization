#!/gpfs/exfel/sw/software/xfel_anaconda3/1.1/bin/python
"""

online device
    uses Qthread to fill a queue of karabo bridge data

    plot the mean of each shimadzu image and the mean of all images in a plot and image widget

    plots XTD Shimadzu camera images   

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
# interface = 'tcp://10.253.0.52:22345'
interface = 'tcp://10.253.0.52:54445'   # XTD Shimadzu
# interface = 'tcp://10.253.0.52:54446' 


# for actual karabo_bridge
## device, key pairs ... dopisat motory, trigger a standby veci, xgms... \
deviceId_name = {'shimadzu_xtd':'SA1_XTD9_IMGPII45/CAM/BEAMVIEW_SHIMADZU:output','shimadzu_spb':'SPB_EHD_HPVX2_1/CAM/CAMERA:output'}
deviceId_property = {'camera': 'data.image.data', 'xgm':'data.intensityTD', 'motor': 'actualPosition.value','delay':'actualDelay.value'}



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
        self.resize(800,1200)
        self.setWindowTitle('Shimadzu XTD Camera View')

        self.d1 = Dock("Shimadzu XTD9", size=(800,800))
        self.d11 = Dock("...", size=(800,400), closable=True)



        self.area.addDock(self.d1,'left')
        self.area.addDock(self.d11,'bottom',self.d1)

        self.imv1 = pg.ImageView()  # moving
        self.d1.addWidget(self.imv1)
        #
        self.show()
        # 
        self.clientThread = bridgeClient(interface)
        self.clientThread.start()
        self.clientThread.finished.connect(self.threadFin)
        self.clientThread.newDataSignal.connect(self.update)



    def update(self, data):
        # if deviceId_name['motor1_4'] in data.keys():
        #     valueData_m14 = data[deviceId_name['motor1_4']][deviceId_property['motor']] 
        #     self.m4_que.append(valueData_m14)
        if deviceId_name['shimadzu_xtd'] in data.keys():
            imageData1 = np.array( data[deviceId_name['shimadzu_xtd']][deviceId_property['camera']]*1. )
            # sum_imageData1 = np.mean(imageData1)
            self.imv1.setImage(imageData1, xvals=np.linspace(1.,128.,imageData1.shape[0]))
            # self.zyla2_que.append(sum_imageData1)
        # QtGui.QApplication.processEvents()
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


