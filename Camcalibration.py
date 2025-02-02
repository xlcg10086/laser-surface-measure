# -*- coding: utf-8 -*-
import sys
from PyQt5.QtWidgets import *
from CamOperation_class_cail import CameraOperation
from Mvlib import *
from Ui.PyUICamcalibration import Ui_MainWindow
import ctypes
import cv2
import numpy as np
import glob
from Lib.Operateparadata import SaveCamcalibrationparameters,SaveLasercalibrationparameters
from numpy.linalg import lstsq

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

# 路径迭代器
def imgs_save_path_generator():
    # 定义文件路径后缀的列表
    paths = ['highexp', 'lowexp', 'ori']
    while True:  # 创建一个无限循环
        for path in paths:
            print(f'{filepath_laser}/{path}')
            yield f'{filepath_laser}/{path}'

# 名称迭代器
def imgs_name_generator():
    # 定义文件路径后缀的列表
    count = 0
    while True:  # 创建一个无限循环
        count +=1
        print(f'img{count}')
        yield f'img{count}'

def laser_trans(p, inter, Rc_1, Tc_1):
    """
    输入像素坐标系下的激光点 (U,V,1)，得到相机坐标系下的 (x,y,z)
    参数:
    p - 像素坐标点 (U,V,1)
    inter - 内参矩阵
    Rc_1 - 旋转矩阵
    Tc_1 - 平移矩阵
    """
    # 像素坐标系 -> 图像坐标系
    # 归一化 x_c/z_c = (u - u_0) / fx
    normlization = np.zeros(2)
    normlization[0] = (p[0] - inter[0, 2]) / inter[0, 0]
    # 归一化 y_c/z_c = (v - v_0) / fy
    normlization[1] = (p[1] - inter[1, 2]) / inter[1, 1]

    # 求伪逆
    t = np.linalg.pinv(Rc_1)
    # print(t)
    a = t[2, 0]
    b = t[2, 1]
    c = t[2, 2]
    d = Tc_1[0]*a + Tc_1[1]*b + Tc_1[2]*c

    # 根据归一化的 x/z 和 y/z 求得 (x,y,z)
    z = d / (a * normlization[0] + b * normlization[1] + c)
    x = z * normlization[0]
    y = z * normlization[1]

    return np.array([x, y, z])


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

    global filepath_cam
    global filepath_laser
    filepath_cam = 'Calibrate/cam_imgs'
    filepath_laser = 'Calibrate/laser_imgs'
    global ret, mtx, dist,abc
    laserimgs_path_generator = imgs_save_path_generator()
    imgname_generator = imgs_name_generator()
    global imgname
    imgname = None
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
        global imgname
        if ui.tabWidget.currentIndex() == 0:
            # filepath_cam = ui.filepath_camimgs.text()
            imgname = next(imgname_generator)
            ret = obj_cam_operation.Save_Bmp(filepath_cam,imgname)
            ui.path_tips.setText(f'保存至{filepath_cam}/{imgname}')
        else:
            # filepath_laser = ui.filepath_laserimgs.text()
            next_path = next(laserimgs_path_generator)      
            if next_path == f'{filepath_laser}/highexp':
                imgname = str(next(imgname_generator))
            ui.path_tips.setText(f'保存至{next_path}/{imgname}')
            ret = obj_cam_operation.Save_Bmp(next_path,imgname)
            
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
    
    # 设置文件浏览弹窗
    def openDirectoryDialog():
        global filepath_cam
        global filepath_laser
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        directory_path = QFileDialog.getExistingDirectory(None, "选择文件夹", options=options)    
        if directory_path:
            if ui.tabWidget.currentIndex() == 0:
                print(directory_path)
                ui.filepath_camimgs.setText(directory_path)
                filepath_cam = directory_path
            else:
                print(directory_path)
                ui.filepath_laserimgs.setText(directory_path)
                filepath_laser = directory_path

    # 校准函数
    def Calibration_cam():
        global ret, mtx, dist
        #设定参数
        strborad = ui.checkborad.text()
        CHECKBORAD = [int(num) for num in strborad.split(",")]
        squareSize = float(ui.squaresize.text())
        # 设置寻找亚像素角点的参数，采用的停止准则是最大循环次数30和最大误差容限0.001
        criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)
        # 获取标定板角点的位置
        objp = np.zeros((CHECKBORAD[0] * CHECKBORAD[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:CHECKBORAD[0], 0:CHECKBORAD[1]].T.reshape(-1, 2)*squareSize # 将世界坐标系建在标定板上，所有点的Z坐标全部为0，所以只需要赋值x和y
        obj_points = [] # 存储3D点
        img_points = [] # 存储2D点
        images = glob.glob(f"{filepath_cam}/*.bmp")
        numofimages = len(images)
        for index,fname in enumerate(images):
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            size = gray.shape[::-1]
            ret, corners = cv2.findChessboardCorners(gray, (CHECKBORAD[0],CHECKBORAD[1]), None)
            print(ret)
            if ret:
                obj_points.append(objp)
                corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria) # 在原角点的基础上寻找亚像素角点
                #print(corners2)
                if [corners2]:
                    img_points.append(corners2)
                else:
                    img_points.append(corners)
            cv2.drawChessboardCorners(img, (CHECKBORAD[0],CHECKBORAD[1]), corners, ret) # 记住，OpenCV的绘制函数一般无返回值
            img = cv2.resize(img, None, fx=0.25, fy=0.25)
            cv2.imshow('img', img)
            cv2.waitKey(20)
            ui.probar_camcali.setValue(int(index*100/numofimages))
            print(len(img_points))
        cv2.destroyAllWindows()
        ui.probar_camcali.setValue(100)
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, size, None, None)
        print("ret:", ret)
        print("mtx:\n", mtx) # 内参数矩阵
        print("dist:\n", dist) # 畸变系数  distortion cofficients = (k_1,k_2,p_1,p_2,k_3)
        print("rvecs:\n", rvecs) # 旋转向量 # 外参数
        print("tvecs:\n", tvecs[0] ) # 平移向量 # 外参数
        print("-----------------------------------------------------")
        ui.output_cam_cali.setEnabled(True)

    def Calibration_laser():
        global ret, mtx, dist,abc
        # 添加一个列表记录一下成功标定的图片标号
        index_list = []
        #设定参数
        strborad = ui.checkborad.text()
        CHECKBORAD = [int(num) for num in strborad.split(",")]
        print(CHECKBORAD)
        squareSize = float(ui.squaresize.text())
        # 设置寻找亚像素角点的参数，采用的停止准则是最大循环次数30和最大误差容限0.001
        criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)
        # 获取标定板角点的位置
        objp = np.zeros((CHECKBORAD[0] * CHECKBORAD[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:CHECKBORAD[0], 0:CHECKBORAD[1]].T.reshape(-1, 2)*squareSize # 将世界坐标系建在标定板上，所有点的Z坐标全部为0，所以只需要赋值x和y
        obj_points = [] # 存储3D点
        img_points = [] # 存储2D点
        images = glob.glob(f"{filepath_laser}/ori/*.bmp")
        numofimages = len(images)
        for index,fname in enumerate(images):
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            size = gray.shape[::-1]
            ret, corners = cv2.findChessboardCorners(gray, (CHECKBORAD[0],CHECKBORAD[1]), None)
            print(ret)
            if ret:
                obj_points.append(objp)
                corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria) # 在原角点的基础上寻找亚像素角点
                #print(corners2)
                if [corners2]:
                    img_points.append(corners2)
                else:
                    img_points.append(corners)
                index_list.append(index)

            cv2.drawChessboardCorners(img, (CHECKBORAD[0],CHECKBORAD[1]), corners, ret) # 记住，OpenCV的绘制函数一般无返回值
            img = cv2.resize(img, None, fx=0.25, fy=0.25)
            cv2.imshow('img', img)
            cv2.waitKey(20)
            ui.probar_lasercali.setValue(int(index*50/numofimages))
            print(index)
        cv2.destroyAllWindows()
        ui.probar_lasercali.setValue(50)
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, size, None, None)
        print("ret:", ret)
        print("mtx:\n", mtx) # 内参数矩阵
        print("dist:\n", dist) # 畸变系数  distortion cofficients = (k_1,k_2,p_1,p_2,k_3)
        print("-----------------------------------------------------")
        # 第二步：获取激光线的图像位置

        images_lowexp = glob.glob(f"{filepath_laser}/lowexp/*.bmp")  # 使用glob模块搜索并获取所有低曝光激光图像的文件路径
        laser_lines = []  # 初始化一个列表来存储激光线的坐标

        # 遍历索引列表，每个索引对应一个图像
        for index in index_list:
            filelow = images_lowexp[index]  # 获取当前索引对应的图像文件路径
            corner_x_list = np.array(img_points[index][:,:,0])  # 提取图像中角点的x坐标
            corner_range = [np.min(corner_x_list), np.max(corner_x_list)]  # 计算x坐标的最小和最大值，确定角点的范围

            img = cv2.imread(filelow)  # 使用OpenCV读取图像文件
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 将图像转换为灰度图

            # 计算用于分析激光线的列的范围
            col_0 = int((corner_range[1] - corner_range[0]) * 1/10 + corner_range[0])
            col_1 = int((corner_range[1] - corner_range[0]) * 9/10 + corner_range[0])
            print((col_0, col_1))  # 打印列的范围

            # 对灰度图像应用阈值处理，将灰度值低于阈值的像素设置为0，高于阈值的保持不变
            _, binary_image = cv2.threshold(gray, 50, 255, cv2.THRESH_TOZERO)

            img = cv2.resize(binary_image, None, fx=0.25, fy=0.25)
            cv2.imshow('img', img)
            cv2.waitKey(20)
            ui.probar_lasercali.setValue(50+int(index*30/numofimages))

            laser_line = []  # 初始化一个列表来存储当前图像的激光线坐标
            # 遍历列的范围，寻找激光线的y坐标
            for x in range(col_0, col_1):
                sum_col = np.sum(binary_image[:, x])  # 计算当前列的像素值总和
                sum_col_weight = np.dot(np.arange(binary_image[:, x].shape[0]), binary_image[:, x])  # 计算加权和

                # 如果列的总和不为0，则计算y坐标
                if sum_col != 0:
                    y = int(sum_col_weight / sum_col)  # 计算y坐标
                    laser_line.append([x, y])  # 将坐标添加到激光线列表中

            laser_lines.append(laser_line)  # 将当前图像的激光线坐标添加到总列表中
        cv2.destroyAllWindows()
        ui.probar_lasercali.setValue(80)

        #第三步坐标转化 像素坐标系->相机坐标系
        transed_locs = []
        # 遍历提供的索引列表中的每个索引
        for index in index_list:
            linel = laser_lines[index]  # 获取当前索引处的激光线
            line = np.array(linel)  # 将激光线转换为NumPy数组
            ones_column = np.ones((line.shape[0], 1))  # 创建一个全为1的列
            line = np.hstack((line, ones_column))  # 将全1列添加到线数组中

            Rc = rvecs[index]  # 当前索引的旋转向量
            Rm, _ = cv2.Rodrigues(Rc)  # 将旋转向量转换为旋转矩阵
            Tc = tvecs[index]  # 当前索引的平移向量

            # 使用laser_trans函数转换线中的每个点
            for point in line:
                loc = laser_trans(point, mtx, Rm, Tc)  # 应用转换
                transed_locs.append(loc)  # 将转换后的位置添加到列表中

            file = open("lines.txt", "a")
            # 添加内容到文件末尾
            content = f'{laser_trans(line[0], mtx, Rm, Tc)}\n {laser_trans(line[-1], mtx, Rm, Tc)}\n'
            file.write(content)
            # 关闭文件
            file.close()

            # 在UI中更新进度条
            ui.probar_lasercali.setValue(80 + int(index * 20 / numofimages))

        # 将转换后的位置列表转换为NumPy数组，并选择第一个切片
        transed_locs = np.array(transed_locs)[:, :, 0]
        # 从转换后的位置中提取X，Y，Z坐标
        loc_X = transed_locs[:, 0].T
        loc_Y = transed_locs[:, 1].T
        loc_Z = transed_locs[:, 2].T
        # print(loc_X.shape)
        # file = open("lines.txt", "a")
        # # 添加内容到文件末尾
        # for index in range(loc_X.shape[0]):
        #     content = f'{loc_X[index]} {loc_Y[index]} {loc_Z[index]}\n'
        #     file.write(content)
        # # 关闭文件
        # file.close()

        # 将坐标与一列1堆叠起来
        xyz = np.column_stack((np.ones(len(loc_X)), loc_X, loc_Y))
        # 打印loc_X和xyz的形状以进行调试
        print(f'locx{loc_X.shape}xyz{xyz.shape}')
        # 执行最小二乘拟合以找到平面方程参数
        para, _, _, _ = lstsq(xyz, loc_Z, rcond=None)
        # 打印激光平面的方程
        print(f'激光刀面的方程为：z={para[1]}x+{para[2]}y+{para[0]}')
        abc = [para[1],para[2],para[0]]
        # 将UI中的进度条设置为100（完成）
        ui.probar_lasercali.setValue(100)
        ui.output_laser_cali.setEnabled(True)


    # 保存数据
    def Savecamcailparameter():
        str = ui.outpath_camcail.text()
        filepath_output,camname = str.split(";")
        SaveCamcalibrationparameters(filepath_output,camname,mtx,dist,ret)

    # 保存数据
    def Savelasercailparameter():
        str = ui.outpath_lasercail.text()
        filepath_output,camname = str.split(";")
        SaveLasercalibrationparameters(filepath_output,camname,mtx,dist,abc,ret)
        
    

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

    ui.filepath_edit_cam.clicked.connect(openDirectoryDialog)
    ui.filepath_edit_laser.clicked.connect(openDirectoryDialog)
    ui.start_cam_cali.clicked.connect(Calibration_cam)
    ui.output_cam_cali.clicked.connect(Savecamcailparameter)
    ui.output_laser_cali.clicked.connect(Savelasercailparameter)
    ui.start_laser_cali.clicked.connect(Calibration_laser)


    ui.tabWidget.setCurrentIndex(0)
    ui.output_cam_cali.setEnabled(False)
    ui.output_laser_cali.setEnabled(False)

    mainWindow.show()

    app.exec_()

    close_device()

    # ch:反初始化SDK | en: finalize SDK
    MvCamera.MV_CC_Finalize()

    sys.exit()
