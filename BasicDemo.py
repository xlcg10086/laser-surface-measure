# -*- coding: utf-8 -*-
import sys
import os
from PyQt5.QtWidgets import *
from PyQt5.QtCore import QTimer
from CamOperation_class import CameraOperation
from Mvlib import *
from Ui.PyUICBasicDemo import Ui_MainWindow
import ctypes
import numpy as np
import pandas as pd
# import math
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from Lib.Operateparadata import ReadCamcalibrationparameters,ReadLasercalibrationparameters
from Lib.Calculation import FitCircle

# 获取选取设备信息的索引，通过[]之间的字符去解析
def TxtWrapBy(start_str, end, all):
    start = all.find(start_str)
    if start >= 0:
        start += len(start_str)
        end = all.find(end, start)
        if end >= 0:
            return all[start:end].strip()


# 将返回的错误码转换为十六进制显示
def ToHexStr(num):
    chaDic = {10: 'a', 11: 'b', 12: 'c', 13: 'd', 14: 'e', 15: 'f'}
    hexStr = ""
    if num < 0:
        num = num + 2 ** 32
    while num >= 16:
        digit = num % 16
        hexStr = chaDic.get(digit, str(digit)) + hexStr
        num //= 16
    hexStr = chaDic.get(num, str(num)) + hexStr
    return hexStr


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=8, height=8, dpi=100):
        # print((width,height))
        self.fig = plt.figure(figsize=(width, height), dpi=dpi)
        super(MplCanvas, self).__init__(self.fig)
        self.parent =parent
        self.mpl_connect('button_press_event', self.on_press)
        self.mpl_connect('motion_notify_event', self.on_motion)
        self.mpl_connect('button_release_event', self.on_release)

        # 创建框选相关的变量
        self.press = None

    def on_press(self, event):
        if event.button == 1:
            self.press = event.xdata, event.ydata

            print(self.press)

    def on_motion(self, event):
        if self.press is not None:
            width = event.xdata - self.press[0]
            height = event.ydata - self.press[1]

            print((width,height))

    def on_release(self, event):
        if event.button == 1 and self.press is not None:
            self.press = event.xdata, event.ydata

            self.rect = None


if __name__ == "__main__":

    # ch:初始化SDK | en: initialize SDK
    MvCamera.MV_CC_Initialize()

    global deviceList
    deviceList = MV_CC_DEVICE_INFO_LIST()
    global cam
    cam = MvCamera()
    global nSelCamIndex
    nSelCamIndex = 0
    global obj_cam_operation
    obj_cam_operation = 0
    global isOpen
    isOpen = False
    global isGrabbing
    isGrabbing = False
    global isCalibMode  # 是否是标定模式（获取原始图像）
    isCalibMode = True
    global isstarttest
    isstarttest = False
    global iscutmode
    iscutmode = False
    global lineindexlist
    global newlinescloud
    global newlinescloud2d
    global transLM

    # 绑定下拉列表至设备信息索引
    def xFunc(event):
        global nSelCamIndex
        nSelCamIndex = TxtWrapBy("[", "]", ui.ComboDevices.get())

    # Decoding Characters
    def decoding_char(c_ubyte_value):
        c_char_p_value = ctypes.cast(c_ubyte_value, ctypes.c_char_p)
        try:
            decode_str = c_char_p_value.value.decode('gbk')  # Chinese characters
        except UnicodeDecodeError:
            decode_str = str(c_char_p_value.value)
        return decode_str

    # ch:枚举相机 | en:enum devices
    def enum_devices():
        global deviceList
        global obj_cam_operation

        deviceList = MV_CC_DEVICE_INFO_LIST()
        ret = MvCamera.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, deviceList)
        if ret != 0:
            strError = "Enum devices fail! ret = :" + ToHexStr(ret)
            QMessageBox.warning(mainWindow, "Error", strError, QMessageBox.Ok)
            return ret

        if deviceList.nDeviceNum == 0:
            QMessageBox.warning(mainWindow, "Info", "Find no device", QMessageBox.Ok)
            return ret
        print("Find %d devices!" % deviceList.nDeviceNum)

        devList = []
        for i in range(0, deviceList.nDeviceNum):
            mvcc_dev_info = cast(deviceList.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
            if mvcc_dev_info.nTLayerType == MV_GIGE_DEVICE:
                print("\ngige device: [%d]" % i)
                user_defined_name = decoding_char(mvcc_dev_info.SpecialInfo.stGigEInfo.chUserDefinedName)
                model_name = decoding_char(mvcc_dev_info.SpecialInfo.stGigEInfo.chModelName)
                print("device user define name: " + user_defined_name)
                print("device model name: " + model_name)

                nip1 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0xff000000) >> 24)
                nip2 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x00ff0000) >> 16)
                nip3 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x0000ff00) >> 8)
                nip4 = (mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x000000ff)
                print("current ip: %d.%d.%d.%d " % (nip1, nip2, nip3, nip4))
                devList.append(
                    "[" + str(i) + "]GigE: " + user_defined_name + " " + model_name + "(" + str(nip1) + "." + str(
                        nip2) + "." + str(nip3) + "." + str(nip4) + ")")
            elif mvcc_dev_info.nTLayerType == MV_USB_DEVICE:
                print("\nu3v device: [%d]" % i)
                user_defined_name = decoding_char(mvcc_dev_info.SpecialInfo.stUsb3VInfo.chUserDefinedName)
                model_name = decoding_char(mvcc_dev_info.SpecialInfo.stUsb3VInfo.chModelName)
                print("device user define name: " + user_defined_name)
                print("device model name: " + model_name)

                strSerialNumber = ""
                for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chSerialNumber:
                    if per == 0:
                        break
                    strSerialNumber = strSerialNumber + chr(per)
                print("user serial number: " + strSerialNumber)
                devList.append("[" + str(i) + "]USB: " + user_defined_name + " " + model_name
                               + "(" + str(strSerialNumber) + ")")

        ui.ComboDevices.clear()
        ui.ComboDevices.addItems(devList)
        ui.ComboDevices.setCurrentIndex(0)

    # ch:打开相机 | en:open device
    def open_device():
        global deviceList
        global nSelCamIndex
        global obj_cam_operation
        global isOpen
        if isOpen:
            QMessageBox.warning(mainWindow, "Error", 'Camera is Running!', QMessageBox.Ok)
            return MV_E_CALLORDER

        nSelCamIndex = ui.ComboDevices.currentIndex()
        if nSelCamIndex < 0:
            QMessageBox.warning(mainWindow, "Error", 'Please select a camera!', QMessageBox.Ok)
            return MV_E_CALLORDER

        obj_cam_operation = CameraOperation(cam, deviceList, nSelCamIndex)
        ret = obj_cam_operation.Open_device()
        if 0 != ret:
            strError = "Open device failed ret:" + ToHexStr(ret)
            QMessageBox.warning(mainWindow, "Error", strError, QMessageBox.Ok)
            isOpen = False
        else:
            set_continue_mode()

            get_param()

            isOpen = True
            enable_controls()
            ui.input_cam_cali.setEnabled(True)
            ui.input_laser_cali.setEnabled(True)

    # ch:开始取流 | en:Start grab image
    def start_grabbing():
        global obj_cam_operation
        global isGrabbing

        ret = obj_cam_operation.Start_grabbing(ui.widgetDisplay.winId())
        if ret != 0:
            strError = "Start grabbing failed ret:" + ToHexStr(ret)
            QMessageBox.warning(mainWindow, "Error", strError, QMessageBox.Ok)
        else:
            isGrabbing = True
            enable_controls()
            ui.btnstartcheck.setEnabled(True)

    # ch:停止取流 | en:Stop grab image
    def stop_grabbing():
        global obj_cam_operation
        global isGrabbing
        ret = obj_cam_operation.Stop_grabbing()
        if ret != 0:
            strError = "Stop grabbing failed ret:" + ToHexStr(ret)
            QMessageBox.warning(mainWindow, "Error", strError, QMessageBox.Ok)
        else:
            isGrabbing = False
            enable_controls()

    # ch:关闭设备 | Close device
    def close_device():
        global isOpen
        global isGrabbing
        global obj_cam_operation

        if isOpen:
            obj_cam_operation.Close_device()
            isOpen = False

        isGrabbing = False

        enable_controls()

    # ch:设置触发模式 | en:set trigger mode
    def set_continue_mode():
        strError = None

        ret = obj_cam_operation.Set_trigger_mode(False)
        if ret != 0:
            strError = "Set continue mode failed ret:" + ToHexStr(ret) + " mode is " + str(is_trigger_mode)
            QMessageBox.warning(mainWindow, "Error", strError, QMessageBox.Ok)
        else:
            ui.radioContinueMode.setChecked(True)
            ui.radioTriggerMode.setChecked(False)
            ui.bnSoftwareTrigger.setEnabled(False)

    # ch:设置软触发模式 | en:set software trigger mode
    def set_software_trigger_mode():

        ret = obj_cam_operation.Set_trigger_mode(True)
        if ret != 0:
            strError = "Set trigger mode failed ret:" + ToHexStr(ret)
            QMessageBox.warning(mainWindow, "Error", strError, QMessageBox.Ok)
        else:
            ui.radioContinueMode.setChecked(False)
            ui.radioTriggerMode.setChecked(True)
            ui.bnSoftwareTrigger.setEnabled(isGrabbing)

    # ch:设置触发命令 | en:set trigger software
    def trigger_once():
        ret = obj_cam_operation.Trigger_once()
        if ret != 0:
            strError = "TriggerSoftware failed ret:" + ToHexStr(ret)
            QMessageBox.warning(mainWindow, "Error", strError, QMessageBox.Ok)

    # ch:存图 | en:save image
    def save_bmp():
        ret = obj_cam_operation.Save_Bmp()
        if ret != MV_OK:
            strError = "Save BMP failed ret:" + ToHexStr(ret)
            QMessageBox.warning(mainWindow, "Error", strError, QMessageBox.Ok)
        else:
            print("Save image success")
            
    def is_float(str):
        try:
            float(str)
            return True
        except ValueError:
            return False
    

    # ch: 获取参数 | en:get param
    def get_param():
        ret = obj_cam_operation.Get_parameter()
        if ret != MV_OK:
            strError = "Get param failed ret:" + ToHexStr(ret)
            QMessageBox.warning(mainWindow, "Error", strError, QMessageBox.Ok)
        else:
            ui.edtExposureTime.setText("{0:.2f}".format(obj_cam_operation.exposure_time))
            ui.edtGain.setText("{0:.2f}".format(obj_cam_operation.gain))
            ui.edtFrameRate.setText("{0:.2f}".format(obj_cam_operation.frame_rate))

    # ch: 设置参数 | en:set param
    def set_param():
        frame_rate = ui.edtFrameRate.text()
        exposure = ui.edtExposureTime.text()
        gain = ui.edtGain.text()

        if is_float(frame_rate)!=True or is_float(exposure)!=True or is_float(gain)!=True:
            strError = "Set param failed ret:" + ToHexStr(MV_E_PARAMETER)
            QMessageBox.warning(mainWindow, "Error", strError, QMessageBox.Ok)
            return MV_E_PARAMETER
        
        ret = obj_cam_operation.Set_parameter(frame_rate, exposure, gain)
        if ret != MV_OK:
            strError = "Set param failed ret:" + ToHexStr(ret)
            QMessageBox.warning(mainWindow, "Error", strError, QMessageBox.Ok)

        return MV_OK
    
    # 打开文件路径
    def openFileDialogA():
        global filepath_cam
        global filepath_laser
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_path, _ = QFileDialog.getOpenFileName(None, "选择文件", "", "All Files (*);;Text Files (*.json)", options=options)
        if file_path:
            print(file_path)
            ui.filepath_camimgs.setText(file_path)
            filepath_cam = file_path

    # 打开文件路径
    def openFileDialogB():
        global filepath_cam
        global filepath_laser
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_path, _ = QFileDialog.getOpenFileName(None, "选择文件", "", "All Files (*);;Text Files (*.json)", options=options)
        if file_path:
            print(file_path)
            ui.filepath_camimgs.setText(file_path)
            filepath_laser = file_path
            
    
    def Inputcamcailpara():
        """
        从UI输入中获取相机标定参数文件路径和相机名称，读取相机标定参数列表，根据相机名称查找并打印对应的相机标定参数。

        Parameters:
            None

        Returns:
            None
        """
        # 从UI输入中获取相机标定参数文件路径和相机名称
        pathway_cam = ui.inpath_camcail.text()
        fileway_cam, camname = pathway_cam.split(";")

        # 读取相机标定参数列表
        camdata_list = ReadCamcalibrationparameters(fileway_cam)

        # 获取已存在的相机名称列表
        existing_camera_names = [params['camera_name'] for params in camdata_list]

        if camname in existing_camera_names:
            # 找到与相机名称匹配的相机标定参数
            index = existing_camera_names.index(camname)
            camdata = camdata_list[index]
            obj_cam_operation.Set_mtx(camdata['mat'])
            obj_cam_operation.Set_dist(camdata['dist']) 
            print(camdata)


    def Inputlasercailpara():
        global transLM
        # 从UI输入中获取相机标定参数文件路径和相机名称
        pathway_laser = ui.inpath_lasercail.text()
        fileway_laser, camname = pathway_laser.split(";")

        # 读取相机标定参数列表
        laserdata_list = ReadLasercalibrationparameters(fileway_laser)

        # 获取已存在的相机名称列表
        existing_camera_names = [params['camera_name'] for params in laserdata_list]

        if camname in existing_camera_names:
            # 找到与相机名称匹配的相机标定参数
            index = existing_camera_names.index(camname)
            laserdata = laserdata_list[index]
            obj_cam_operation.Set_abc(laserdata['abc']) 
            # 计算旋转矩阵获取激光面投影
            [a,b,c]=laserdata['abc']
            i_prime = np.array([1, 0, a])
            j_prime = np.array([a*b, -(a**2+1), -b])
            k_prime = np.array([a, b, -1])

            # 规范化基向量
            i_prime = i_prime / np.linalg.norm(i_prime)
            j_prime = j_prime / np.linalg.norm(j_prime)
            k_prime = k_prime / np.linalg.norm(k_prime)

            # 构建旋转矩阵
            transLM = np.column_stack((i_prime, j_prime, k_prime))
            print(f'平面参数为：{laserdata}，旋转矩阵为{transLM}')
            

    def StartTest():
        global isstarttest
        if isstarttest == False:
            obj_cam_operation.is_graph_process = True
            mainWindow.timertest3d.start(1500)
            mainWindow.timertest2d.start(1000)
            isstarttest = True
            ui.btnstartcheck.setText('停止测试')
        else:
            obj_cam_operation.is_graph_process = False
            mainWindow.timertest3d.stop()
            mainWindow.timertest2d.stop()
            isstarttest = False
            ui.btnstartcheck.setText('开始测试')

    # 点云3d重建画面
    def Drawpointcloud3d():
        # print(obj_cam_operation.lines_cloud)
        global iscutmode
        global lineindexlist
        global newlinescloud
        ui.graph3d.cla()
        if iscutmode:
            # print(len(lineindexlist))
            if len(lineindexlist) % 2 == 0:
                newlinescloud = np.empty((0, 3))  # 初始化一个空的二维数组，用于存储提取的子序列

                # 遍历lineindexlist中的成对整数
                for index in range(int(len(lineindexlist) / 2)):
                    startindex = int(lineindexlist[index * 2])  # 获取起始索引
                    endindex = int(lineindexlist[index * 2 + 1])  # 获取结束索引

                    subset = obj_cam_operation.lines_cloud[startindex:endindex]  # 提取指定索引范围的子序列
                    newlinescloud = np.concatenate([newlinescloud, subset])  # 将子序列与newlinescloud进行组合，并更新newlinescloud

                # print(newlinescloud)

                x = newlinescloud.T[0]
                y = newlinescloud.T[1]
                z = newlinescloud.T[2]
                ui.graph3d.scatter(x, y, z, c='r', marker='o')

                 # 设置坐标轴标签
                ui.graph3d.set_xlabel('X')
                ui.graph3d.set_ylabel('Y')
                ui.graph3d.set_zlabel('Z')

                # ui.graph.text(5, 5,5, 'Your Text Here', fontsize=12, color='red')
                ui.labelradius.setText(f'点云数量为{obj_cam_operation.lines_cloud.shape}')
            else:
                print('error,index setting error')
        else:

            newlinescloud = obj_cam_operation.lines_cloud
            
            x = newlinescloud.T[0]
            y = newlinescloud.T[1]
            z = newlinescloud.T[2]
            ui.graph3d.scatter(x, y, z, c='r', marker='o')

            # 设置坐标轴标签
            ui.graph3d.set_xlabel('X')
            ui.graph3d.set_ylabel('Y')
            ui.graph3d.set_zlabel('Z')
            ui.labelradius.setText(f'点云数量为{obj_cam_operation.lines_cloud.shape}')

        # 刷新画布
        ui.canvas3d.draw()

    # 进行坐标的旋转变换
    def Trans3dto2d(line3d):
        global transLM

        lines2d = np.dot(line3d,transLM)
        # print(lines2d)
        return lines2d[:,0:2]


    # 点云2d重建画面
    def Drawpointcloud2d():
        # print(obj_cam_operation.lines_cloud)
        global iscutmode
        global lineindexlist
        global newlinescloud
        global newlinescloud2d
        ui.graph2d.cla()
        if iscutmode:
            # print(len(lineindexlist))
            if len(lineindexlist) % 2 == 0:
                newlinescloud = np.empty((0, 3))  # 初始化一个空的二维数组，用于存储提取的子序列

                # 遍历lineindexlist中的成对整数
                for index in range(int(len(lineindexlist) / 2)):
                    startindex = int(lineindexlist[index * 2])  # 获取起始索引
                    endindex = int(lineindexlist[index * 2 + 1])  # 获取结束索引

                    subset = obj_cam_operation.lines_cloud[startindex:endindex]  # 提取指定索引范围的子序列
                    newlinescloud = np.concatenate([newlinescloud, subset])  # 将子序列与newlinescloud进行组合，并更新newlinescloud

                newlinescloud2d = Trans3dto2d(newlinescloud)
                x = newlinescloud2d.T[0]
                y = newlinescloud2d.T[1]

                ui.graph2d.plot(x, y, c='r', marker='o')

                ui.graph2d.set_xlabel('X')  # 添加x轴标签
                ui.graph2d.set_ylabel('Y')  # 添加y轴标签
                ui.graph2d.set_title(f'pointclouds:{obj_cam_operation.lines_cloud.shape}')  # 添加标题
                
            else:
                print('error,index setting error')
        else:

            newlinescloud = obj_cam_operation.lines_cloud

            newlinescloud2d = Trans3dto2d(newlinescloud)
            x = newlinescloud2d.T[0]
            y = newlinescloud2d.T[1]

            ui.graph2d.plot(x, y, c='r', marker='o')

            ui.graph2d.set_xlabel('X')  # 添加x轴标签
            ui.graph2d.set_ylabel('Y')  # 添加y轴标签
            ui.graph2d.set_title(f'pointclouds:{obj_cam_operation.lines_cloud.shape}')  # 添加标题

        # 刷新画布
        ui.canvas2d.draw()

    def Savelinecloud(demension,file_path='output.xlsx'):
        global newlinescloud

        # 检查文件是否存在
        if os.path.exists(file_path):
            # 打开现有文件
            writer = pd.ExcelWriter(file_path, mode='a', engine='openpyxl')
            existing_data = pd.read_excel(file_path, sheet_name=None)
            sheets = existing_data.keys()
            startrow = max([existing_data[sheet].shape[0] for sheet in sheets])
        else:
            # 创建新文件
            writer = pd.ExcelWriter(file_path, engine='openpyxl')
            startrow = 0

        # 创建 DataFrame 对象
        if demension == '3d':
            df = pd.DataFrame(newlinescloud)
        elif demension == '2d':
            df = pd.DataFrame(newlinescloud2d)

        # 添加表头
        # header_df = pd.DataFrame(['linecloud'], columns=['linecloud'])

        # # 合并 DataFrame 对象
        # df = pd.concat([header_df, df])

        # 获取现有的sheet名称
        if startrow !=0:
            existing_sheets = set(sheets) if sheets else set()

            # 生成新的sheet名称
            sheet_num = 1
            new_sheet_name = f'Sheet{sheet_num}'
            while new_sheet_name in existing_sheets:
                sheet_num += 1
                new_sheet_name = f'Sheet{sheet_num}'
            startrow = 0
        else:
            new_sheet_name = 'Sheet1'

        # 保存为 Excel 文件
        df.to_excel(writer, index=False, header=False, startrow=startrow, sheet_name=new_sheet_name)

        # 关闭并保存文件
        writer.save()
        writer.close()



    def Oncutmode():
        global iscutmode
        global lineindexlist
        iscutmode = True
        # ui.btncutcclouds.setText('关闭剪裁')
        string = ui.linerange.text()
        lineindexlist = string.split(',')
        
    def Offcutmode():
        global iscutmode
        global lineindexlist
        iscutmode = False
        # ui.btncutcclouds.setText('关闭剪裁')
        string = ui.linerange.text()
        lineindexlist = string.split(',')

    def Testedgelength():
        thlength = float(ui.editedgeheight.text())
        ydiff = abs(np.diff(newlinescloud2d[:,1]))
        # 找到满足条件的索引（差值大于thlength）
        indices = np.where(ydiff >= thlength)[0]
        print(f'断点为{indices}，高度为{np.max(ydiff)}')
        if(len(indices) == 2):
            leftx = newlinescloud2d[:,0][indices[0]+2]
            rightx = newlinescloud2d[:,0][indices[1]-2]
            xlength = rightx-leftx
            if 10 < xlength <20:
                leftx = newlinescloud2d[:,0][indices[0]+1]
                rightx = newlinescloud2d[:,0][indices[1]-1]
                xlength = rightx-leftx
            elif xlength <10:
                leftx = (newlinescloud2d[:,0][indices[0]+1]*2+newlinescloud2d[:,0][indices[0]])/3
                rightx = (newlinescloud2d[:,0][indices[1]-1]*2+newlinescloud2d[:,0][indices[1]])/3
                xlength = rightx-leftx
            # print(xlength)
            ui.text_length.setText(f'长度为{round(xlength,4)}mm')
        # 打开文件（如果文件不存在，则会创建一个新文件）
            file = open("result.txt", "a")
            # 添加内容到文件末尾
            content = f'长度为{round(xlength,4)}mm,高度为{np.max(ydiff)}mm\n'
            file.write(content)
            # 关闭文件
            file.close()

    def Testedgecircle():
        centerpoint,radius = FitCircle(newlinescloud2d,0,0,70,1)
        
        file = open("result.txt", "a")
        # 添加内容到文件末尾
        content = f'圆心为{centerpoint},半径为{radius}mm\n'
        print(content)
        file.write(content)
        # 关闭文件
        file.close()

    # ch: 设置控件状态 | en:set enable status
    def enable_controls():
        global isGrabbing
        global isOpen

        # 先设置group的状态，再单独设置各控件状态
        ui.groupGrab.setEnabled(isOpen)
        ui.groupParam.setEnabled(isOpen)

        ui.bnOpen.setEnabled(not isOpen)
        ui.bnClose.setEnabled(isOpen)

        ui.bnStart.setEnabled(isOpen and (not isGrabbing))
        ui.bnStop.setEnabled(isOpen and isGrabbing)
        ui.bnSoftwareTrigger.setEnabled(isGrabbing and ui.radioTriggerMode.isChecked())

        ui.bnSaveImage.setEnabled(isOpen and isGrabbing)

    # ch: 初始化app, 绑定控件与函数 | en: Init app, bind ui and api
    app = QApplication(sys.argv)
    mainWindow = QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(mainWindow)
    # 创建3d画布
    ui.canvas3d = MplCanvas(ui,8,6,100)
    ui.graph3d = ui.canvas3d.fig.add_subplot(111,projection='3d')
    # 创建一个layout
    ui.vbox3d = QHBoxLayout(ui.widgetradius3d)
    ui.vbox3d.addWidget(ui.canvas3d)
    # 创建2d画布
    ui.canvas2d = MplCanvas(ui,8,6,100)
    ui.graph2d = ui.canvas2d.fig.add_subplot(111)
    # 创建一个layout
    ui.vbox2d = QHBoxLayout(ui.widgetradius)
    ui.vbox2d.addWidget(ui.canvas2d)

    ui.bnEnum.clicked.connect(enum_devices)
    ui.bnOpen.clicked.connect(open_device)
    ui.bnClose.clicked.connect(close_device)
    ui.bnStart.clicked.connect(start_grabbing)
    ui.bnStop.clicked.connect(stop_grabbing)

    ui.bnSoftwareTrigger.clicked.connect(trigger_once)
    ui.radioTriggerMode.clicked.connect(set_software_trigger_mode)
    ui.radioContinueMode.clicked.connect(set_continue_mode)

    ui.bnGetParam.clicked.connect(get_param)
    ui.bnSetParam.clicked.connect(set_param)

    ui.bnSaveImage.clicked.connect(save_bmp)

    ui.openFileA.clicked.connect(openFileDialogA)
    ui.openFileB.clicked.connect(openFileDialogB)
    ui.input_cam_cali.clicked.connect(Inputcamcailpara)
    ui.input_laser_cali.clicked.connect(Inputlasercailpara)
    ui.btnstartcheck.clicked.connect(StartTest)
    ui.btncutclouds.clicked.connect(Oncutmode)
    ui.btnrdtclouds.clicked.connect(Offcutmode)
    ui.savelinescloud3d.clicked.connect(lambda: Savelinecloud('3d'))
    ui.savelinescloud2d.clicked.connect(lambda: Savelinecloud('2d'))
    ui.btntestedge.clicked.connect(Testedgelength)
    ui.btntestcircle.clicked.connect(Testedgecircle)


    mainWindow.timertest3d = QTimer(mainWindow)
    mainWindow.timertest3d.timeout.connect(Drawpointcloud3d)
    mainWindow.timertest2d = QTimer(mainWindow)
    mainWindow.timertest2d.timeout.connect(Drawpointcloud2d)


    # 状态设置
    ui.Widgetcircle.setCurrentIndex(0)
    ui.input_cam_cali.setEnabled(False)
    ui.input_laser_cali.setEnabled(False)
    ui.btnstartcheck.setEnabled(False)
    ui.tabWidget_2.setCurrentIndex(0)

    
    mainWindow.show()
    app.exec_()

    close_device()

    # ch:反初始化SDK | en: finalize SDK
    MvCamera.MV_CC_Finalize()

    sys.exit()
