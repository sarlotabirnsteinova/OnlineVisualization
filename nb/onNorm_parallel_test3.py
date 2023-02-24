#!/gpfs/exfel/sw/software/xfel_anaconda3/1.1/bin/python
"""
online device
    - visualise raw and normalised image from Shimadzu X
    - one bridge
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
import sys, argparse
import time
import datetime
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from karabo_bridge import Client
from collections import deque
from threading import Thread

sys.path.append('../src')

from dffc.correction import DynamicFlatFieldCorrectionCython as DynamicFlatFieldCorrection
from dffc.offline import FlatFieldCorrectionFileProcessor
from dffc.parallel import ReaderBase



## device, key pairs
device_name = {'zyla5':'SPB_EXP_ZYLA/CAM/5:output','zyla3':'SPB_EXP_ZYLA/CAM/3:output','zyla4':'SPB_EXP_ZYLA/CAM/4:output','shimadzu3':'SPB_EHD_HPVX2_3/CAM/CAMERA:output','shimadzu2':'SPB_EHD_HPVX2_2/CAM/CAMERA:output','shimadzu1':'SPB_EHD_HPVX2_1/CAM/CAMERA:output','shimadzu1del':'SPB_EHD_HPVX2_1/TSYS/STBY','shimadzu2del':'SPB_EHD_HPVX2_2/TSYS/STBY','dg1':'SPB_EXP_SYS/DG/DG1','dg2':'SPB_EXP_SYS/DG/DG2'}
device_prop = {'camera': 'data.image.data', 'motor': 'actualPosition.value','delayG':'G.delay.value','delayA':'A.delay.value', 'delayCam': 'actualDelay.value'}

parser = argparse.ArgumentParser(description="""Pass Shimadzu  number for dynamic online flat-field correction.""")
parser.add_argument('-cno',type=int, dest='shimadzu_no',help="Shimadzu id number for data processing.")

args = parser.parse_args()
camno = args.shimadzu_no
interface = None
if camno == 1:
    interface = 'tcp://10.253.1.63:54456' # bridge Shimadzu 1 
elif camno == 3:
    interface = 'tcp://10.253.1.63:54458' # bridge Shimadzu 3
else:
    print('Unknown camera id!')
n_components = 20
downsample_factors = (2, 4)
#camno = 2
runno_dark = 37
runno_flat = 36
fn = f"pca_cam{camno}_d{runno_dark}_f{runno_flat}_r{n_components}.h5"
cam_source = f"SPB_EHD_HPVX2_{camno}/CAM/CAMERA:output"
dffc = DynamicFlatFieldCorrection.from_file(fn, cam_source, downsample_factors)

from threading import Thread
 
# custom thread class
class CustomThread(Thread):
    def __init__(self, reader, cam_source):
        super().__init__()
        self.reader = reader
        self.cam_source = cam_source

    def stop(self):
        self._running = False

    # override the run function
    def run(self):
        client = Client(interface, timeout = 5)
        self._running = True
        while self._running:
            try:
                data, meta = client.next()
                #print(data)
                #print(meta)
                if self.reader.data_chunk is None:
                    self.reader.data_tid = meta[self.cam_source]['timestamp.tid']
                    self.reader.data_chunk = data[self.cam_source][device_prop["camera"]]
                    print("get data", self.reader.data_tid)
                else:
                    print("drop data", self.reader.data_tid)

            except TimeoutError as e:
                data = None
            time.sleep(0)


class FlatFieldCorrectionBridgeReader(ReaderBase):

    def stop(self):
        self._running = False

    def read(self):
        count = 0 if self.data_chunk is None else len(self.data_chunk)
        return count

    def is_eof(self, nimg):
        return not self._running


    def __init__(self, shm, buf_size, alg_cls, args):
        super().__init__(shm, buf_size, alg_cls, args)                                                                                                     

        self._shm.map_arrays(self, ['images', 'images_corr'])

        self.source = args['source']
        _, self.ny, self.nx = self.images.shape


    def push(self, first, count):
        self.tid_queue.append(self.data_tid)

        i0 = first % self.buf_size
        iN = (i0 + count) % self.buf_size
        if iN < i0:
            k = self.buf_size - i0
            self.images[i0:] = self.data_chunk[:k]
            self.images[:iN] = self.data_chunk[k:]
        else:
            self.images[i0:iN] = self.data_chunk

        self.data_chunk = None
        print("push", self.data_tid)

    def pop(self, first, count):
        tid = self.tid_queue.pop(0)
        images_corr = np.zeros([count, self.ny, self.nx], float)

        i0 = first % self.buf_size
        iN = (i0 + count) % self.buf_size

        if iN < i0:
            k = self.buf_size - i0
            images_corr[:k] = self.images_corr[i0:]
            images_corr[k:] = self.images_corr[:iN]
        else:
            images_corr[:] = self.images_corr[i0:iN]

        print("pop", tid)
        # plot here
        self._gui.update_view(tid, images_corr)

    def run(self, gui):
        self._gui = gui
        self.bridge = CustomThread(self, self.source)

        self.data_chunk = None
        self.data_tid = 0
        
        self.tid_queue = []
        self.results = []
        self.trains = []

        self.bridge.start()
        self._running = True
        super().run()
        self.bridge.stop()
        self.bridge.join()


class FlatFieldCorrectionBridgeProcessor(FlatFieldCorrectionFileProcessor):

    Reader = FlatFieldCorrectionBridgeReader

    def run(self, gui):
        self.rdr.run(gui)


class ProcessingThread(QThread):
    def __init__(self, gui, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._gui = gui

    def stop(self):
        self.proc.rdr.stop()

    def run(self):
        self.proc = FlatFieldCorrectionBridgeProcessor(dffc, 32)
        self.proc.start_workers()
        self.proc.run(self._gui)
        self.proc.join_workers()
        

"""
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
"""

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
        self.d1 = Dock(f"Shimadzu {camno}: raw", size=(250,400))
        self.d11 = Dock(f"Shimadzu {camno}: normalised", size=(250,400), closable=True)
#         self.d2 = Dock(f"Shimadzu {shimadzu_no}: raw", size=(250,400))
#         self.d22 = Dock("Shimadzu 2: normalised", size=(250,400), closable=True)

    

        self.area.addDock(self.d0, 'left')
        self.area.addDock(self.d1,'bottom',self.d0)
        self.area.addDock(self.d11,'right',self.d1)
#         self.area.addDock(self.d2,'bottom',self.d1)
#         self.area.addDock(self.d22,'bottom',self.d11)



        # Shimadzu 1 
        self.imv1 = pg.ImageView()  
        self.d1.addWidget(self.imv1)
        self.imv11 = pg.ImageView()  
        self.d11.addWidget(self.imv11)
#         # Shimadzu 2
#         self.imv2 = pg.ImageView()  
#         self.d2.addWidget(self.imv2)
#         self.imv22 = pg.ImageView()  
#         self.d22.addWidget(self.imv22)


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
        #self.clientThread = bridgeClientShimadzu(interface)
        #self.clientThread.start()
        #self.clientThread.finished.connect(self.threadFin)
        #self.clientThread.newDataSignal.connect(self.update)
  

    def update_view(self, tid, image):
        self.imv11.setImage(image)
        QtGui.QApplication.processEvents()

    def update(self, dataBridge):
        """ called by bridgeClientShimadzu when bridge receives Shimadzu data
        d = data, meta
        """
        dataHPVX, meta = dataBridge
        imgs_Shim1 = None
#         imgs_Shim2 = None

        # raw data
        if device_name[f"shimadzu{camno}"] in dataHPVX.keys(): # 
            imgs_Shim1 = np.array(dataHPVX[device_name[f'shimadzu{camno}']][device_prop['camera']]*1.)   # SHimadzu 1   # *1. makes a copy                     
            self.imv1.setImage(imgs_Shim1)    
#         if device_name['shimadzu2'] in dataHPVX.keys():   # 
#             imgs_Shim2 = np.array( dataHPVX[device_name['shimadzu2']][device_prop['camera']]*1. )   # SHimadzu 2
#             self.imv2.setImage(imgs_Shim2)    


        # normalised
        if imgs_Shim1 is not None:
            tm0 = time.monotonic()
            proc = FlatFieldCorrectionFileProcessor(dffc,32)
            proc.start_workers()
            proc.run((42,imgs_Shim1))
            proc.join_workers()
            tm_f = time.monotonic() - tm0
            print(f"correction takes{tm_f}s")
            self.imv11.setImage(np.array(proc.rdr.results[0]))           
            
        QtGui.QApplication.processEvents()


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
    pt = ProcessingThread(form)
    pt.start()
    exitcode = app.exec_()
    pt.stop()
    #pt.join()
    sys.exit(exitcode)
    #sys.exit(app.exec_())


if __name__ == '__main__':
    main()
