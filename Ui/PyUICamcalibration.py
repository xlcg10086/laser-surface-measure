# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Ui/Camcalibration.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(850, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout_7.setContentsMargins(5, 5, 5, 5)
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.diswidget = QtWidgets.QWidget(self.centralwidget)
        self.diswidget.setObjectName("diswidget")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.diswidget)
        self.verticalLayout_3.setContentsMargins(5, 5, 5, 5)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.ComboDevices = QtWidgets.QComboBox(self.diswidget)
        self.ComboDevices.setObjectName("ComboDevices")
        self.verticalLayout_3.addWidget(self.ComboDevices)
        self.widgetDisplay = QtWidgets.QWidget(self.diswidget)
        self.widgetDisplay.setObjectName("widgetDisplay")
        self.verticalLayout_3.addWidget(self.widgetDisplay)
        self.tabWidget = QtWidgets.QTabWidget(self.diswidget)
        self.tabWidget.setObjectName("tabWidget")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.tab)
        self.verticalLayout.setObjectName("verticalLayout")
        self.widget_5 = QtWidgets.QWidget(self.tab)
        self.widget_5.setObjectName("widget_5")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.widget_5)
        self.horizontalLayout.setContentsMargins(3, 3, 3, 3)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label_8 = QtWidgets.QLabel(self.widget_5)
        self.label_8.setObjectName("label_8")
        self.horizontalLayout.addWidget(self.label_8)
        self.filepath_camimgs = QtWidgets.QLineEdit(self.widget_5)
        self.filepath_camimgs.setObjectName("filepath_camimgs")
        self.horizontalLayout.addWidget(self.filepath_camimgs)
        self.filepath_edit_cam = QtWidgets.QPushButton(self.widget_5)
        self.filepath_edit_cam.setObjectName("filepath_edit_cam")
        self.horizontalLayout.addWidget(self.filepath_edit_cam)
        self.verticalLayout.addWidget(self.widget_5)
        self.widget_6 = QtWidgets.QWidget(self.tab)
        self.widget_6.setObjectName("widget_6")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.widget_6)
        self.horizontalLayout_2.setContentsMargins(3, 3, 3, 3)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label_9 = QtWidgets.QLabel(self.widget_6)
        self.label_9.setObjectName("label_9")
        self.horizontalLayout_2.addWidget(self.label_9)
        self.probar_camcali = QtWidgets.QProgressBar(self.widget_6)
        self.probar_camcali.setProperty("value", 0)
        self.probar_camcali.setObjectName("probar_camcali")
        self.horizontalLayout_2.addWidget(self.probar_camcali)
        self.start_cam_cali = QtWidgets.QPushButton(self.widget_6)
        self.start_cam_cali.setObjectName("start_cam_cali")
        self.horizontalLayout_2.addWidget(self.start_cam_cali)
        self.verticalLayout.addWidget(self.widget_6)
        self.widget_4 = QtWidgets.QWidget(self.tab)
        self.widget_4.setObjectName("widget_4")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.widget_4)
        self.horizontalLayout_3.setContentsMargins(3, 3, 3, 3)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.label_7 = QtWidgets.QLabel(self.widget_4)
        self.label_7.setObjectName("label_7")
        self.horizontalLayout_3.addWidget(self.label_7)
        self.outpath_camcail = QtWidgets.QLineEdit(self.widget_4)
        self.outpath_camcail.setObjectName("outpath_camcail")
        self.horizontalLayout_3.addWidget(self.outpath_camcail)
        self.output_cam_cali = QtWidgets.QPushButton(self.widget_4)
        self.output_cam_cali.setObjectName("output_cam_cali")
        self.horizontalLayout_3.addWidget(self.output_cam_cali)
        self.verticalLayout.addWidget(self.widget_4)
        self.tabWidget.addTab(self.tab, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.tab_2)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.widget = QtWidgets.QWidget(self.tab_2)
        self.widget.setObjectName("widget")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.widget)
        self.horizontalLayout_4.setContentsMargins(3, 3, 3, 3)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.label = QtWidgets.QLabel(self.widget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)
        self.label.setObjectName("label")
        self.horizontalLayout_4.addWidget(self.label)
        self.filepath_laserimgs = QtWidgets.QLineEdit(self.widget)
        self.filepath_laserimgs.setObjectName("filepath_laserimgs")
        self.horizontalLayout_4.addWidget(self.filepath_laserimgs)
        self.filepath_edit_laser = QtWidgets.QPushButton(self.widget)
        self.filepath_edit_laser.setObjectName("filepath_edit_laser")
        self.horizontalLayout_4.addWidget(self.filepath_edit_laser)
        self.verticalLayout_2.addWidget(self.widget)
        self.widget_2 = QtWidgets.QWidget(self.tab_2)
        self.widget_2.setObjectName("widget_2")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout(self.widget_2)
        self.horizontalLayout_5.setContentsMargins(3, 3, 3, 3)
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.label_2 = QtWidgets.QLabel(self.widget_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_2.sizePolicy().hasHeightForWidth())
        self.label_2.setSizePolicy(sizePolicy)
        self.label_2.setSizeIncrement(QtCore.QSize(0, 0))
        self.label_2.setBaseSize(QtCore.QSize(0, 0))
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_5.addWidget(self.label_2)
        self.probar_lasercali = QtWidgets.QProgressBar(self.widget_2)
        self.probar_lasercali.setProperty("value", 0)
        self.probar_lasercali.setObjectName("probar_lasercali")
        self.horizontalLayout_5.addWidget(self.probar_lasercali)
        self.start_laser_cali = QtWidgets.QPushButton(self.widget_2)
        self.start_laser_cali.setObjectName("start_laser_cali")
        self.horizontalLayout_5.addWidget(self.start_laser_cali)
        self.verticalLayout_2.addWidget(self.widget_2)
        self.widget_3 = QtWidgets.QWidget(self.tab_2)
        self.widget_3.setObjectName("widget_3")
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout(self.widget_3)
        self.horizontalLayout_6.setContentsMargins(3, 3, 3, 3)
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.label_3 = QtWidgets.QLabel(self.widget_3)
        self.label_3.setObjectName("label_3")
        self.horizontalLayout_6.addWidget(self.label_3)
        self.outpath_lasercail = QtWidgets.QLineEdit(self.widget_3)
        self.outpath_lasercail.setObjectName("outpath_lasercail")
        self.horizontalLayout_6.addWidget(self.outpath_lasercail)
        self.output_laser_cali = QtWidgets.QPushButton(self.widget_3)
        self.output_laser_cali.setObjectName("output_laser_cali")
        self.horizontalLayout_6.addWidget(self.output_laser_cali)
        self.verticalLayout_2.addWidget(self.widget_3)
        self.tabWidget.addTab(self.tab_2, "")
        self.verticalLayout_3.addWidget(self.tabWidget)
        self.verticalLayout_3.setStretch(1, 3)
        self.verticalLayout_3.setStretch(2, 1)
        self.horizontalLayout_7.addWidget(self.diswidget)
        self.sideWidget = QtWidgets.QWidget(self.centralwidget)
        self.sideWidget.setObjectName("sideWidget")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.sideWidget)
        self.verticalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_4.setSpacing(5)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.groupInit = QtWidgets.QGroupBox(self.sideWidget)
        self.groupInit.setObjectName("groupInit")
        self.gridLayoutWidget = QtWidgets.QWidget(self.groupInit)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(0, 20, 201, 81))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.bnClose = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.bnClose.setEnabled(False)
        self.bnClose.setObjectName("bnClose")
        self.gridLayout.addWidget(self.bnClose, 2, 2, 1, 1)
        self.bnOpen = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.bnOpen.setObjectName("bnOpen")
        self.gridLayout.addWidget(self.bnOpen, 2, 1, 1, 1)
        self.bnEnum = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.bnEnum.setObjectName("bnEnum")
        self.gridLayout.addWidget(self.bnEnum, 1, 1, 1, 2)
        self.verticalLayout_4.addWidget(self.groupInit)
        self.groupGrab = QtWidgets.QGroupBox(self.sideWidget)
        self.groupGrab.setEnabled(False)
        self.groupGrab.setObjectName("groupGrab")
        self.gridLayoutWidget_2 = QtWidgets.QWidget(self.groupGrab)
        self.gridLayoutWidget_2.setGeometry(QtCore.QRect(0, 19, 202, 141))
        self.gridLayoutWidget_2.setObjectName("gridLayoutWidget_2")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.gridLayoutWidget_2)
        self.gridLayout_2.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.radioContinueMode = QtWidgets.QRadioButton(self.gridLayoutWidget_2)
        self.radioContinueMode.setObjectName("radioContinueMode")
        self.gridLayout_2.addWidget(self.radioContinueMode, 0, 0, 1, 1)
        self.radioTriggerMode = QtWidgets.QRadioButton(self.gridLayoutWidget_2)
        self.radioTriggerMode.setObjectName("radioTriggerMode")
        self.gridLayout_2.addWidget(self.radioTriggerMode, 0, 1, 1, 1)
        self.bnSoftwareTrigger = QtWidgets.QPushButton(self.gridLayoutWidget_2)
        self.bnSoftwareTrigger.setEnabled(False)
        self.bnSoftwareTrigger.setObjectName("bnSoftwareTrigger")
        self.gridLayout_2.addWidget(self.bnSoftwareTrigger, 3, 0, 1, 2)
        self.bnSaveImage = QtWidgets.QPushButton(self.gridLayoutWidget_2)
        self.bnSaveImage.setEnabled(False)
        self.bnSaveImage.setObjectName("bnSaveImage")
        self.gridLayout_2.addWidget(self.bnSaveImage, 4, 0, 1, 2)
        self.bnStart = QtWidgets.QPushButton(self.gridLayoutWidget_2)
        self.bnStart.setEnabled(False)
        self.bnStart.setObjectName("bnStart")
        self.gridLayout_2.addWidget(self.bnStart, 2, 0, 1, 1)
        self.bnStop = QtWidgets.QPushButton(self.gridLayoutWidget_2)
        self.bnStop.setEnabled(False)
        self.bnStop.setObjectName("bnStop")
        self.gridLayout_2.addWidget(self.bnStop, 2, 1, 1, 1)
        self.path_tips = QtWidgets.QLabel(self.groupGrab)
        self.path_tips.setGeometry(QtCore.QRect(0, 170, 261, 16))
        self.path_tips.setObjectName("path_tips")
        self.verticalLayout_4.addWidget(self.groupGrab)
        self.groupParam = QtWidgets.QGroupBox(self.sideWidget)
        self.groupParam.setEnabled(False)
        self.groupParam.setObjectName("groupParam")
        self.gridLayoutWidget_3 = QtWidgets.QWidget(self.groupParam)
        self.gridLayoutWidget_3.setGeometry(QtCore.QRect(0, 20, 201, 131))
        self.gridLayoutWidget_3.setObjectName("gridLayoutWidget_3")
        self.gridLayoutParam = QtWidgets.QGridLayout(self.gridLayoutWidget_3)
        self.gridLayoutParam.setContentsMargins(0, 0, 0, 0)
        self.gridLayoutParam.setObjectName("gridLayoutParam")
        self.label_6 = QtWidgets.QLabel(self.gridLayoutWidget_3)
        self.label_6.setObjectName("label_6")
        self.gridLayoutParam.addWidget(self.label_6, 3, 0, 1, 1)
        self.edtGain = QtWidgets.QLineEdit(self.gridLayoutWidget_3)
        self.edtGain.setObjectName("edtGain")
        self.gridLayoutParam.addWidget(self.edtGain, 1, 1, 1, 1)
        self.label_5 = QtWidgets.QLabel(self.gridLayoutWidget_3)
        self.label_5.setObjectName("label_5")
        self.gridLayoutParam.addWidget(self.label_5, 1, 0, 1, 1)
        self.label_4 = QtWidgets.QLabel(self.gridLayoutWidget_3)
        self.label_4.setObjectName("label_4")
        self.gridLayoutParam.addWidget(self.label_4, 0, 0, 1, 1)
        self.edtExposureTime = QtWidgets.QLineEdit(self.gridLayoutWidget_3)
        self.edtExposureTime.setObjectName("edtExposureTime")
        self.gridLayoutParam.addWidget(self.edtExposureTime, 0, 1, 1, 1)
        self.bnGetParam = QtWidgets.QPushButton(self.gridLayoutWidget_3)
        self.bnGetParam.setObjectName("bnGetParam")
        self.gridLayoutParam.addWidget(self.bnGetParam, 4, 0, 1, 1)
        self.bnSetParam = QtWidgets.QPushButton(self.gridLayoutWidget_3)
        self.bnSetParam.setObjectName("bnSetParam")
        self.gridLayoutParam.addWidget(self.bnSetParam, 4, 1, 1, 1)
        self.edtFrameRate = QtWidgets.QLineEdit(self.gridLayoutWidget_3)
        self.edtFrameRate.setObjectName("edtFrameRate")
        self.gridLayoutParam.addWidget(self.edtFrameRate, 3, 1, 1, 1)
        self.gridLayoutParam.setColumnStretch(0, 2)
        self.gridLayoutParam.setColumnStretch(1, 3)
        self.verticalLayout_4.addWidget(self.groupParam)
        self.groupBorad = QtWidgets.QGroupBox(self.sideWidget)
        self.groupBorad.setObjectName("groupBorad")
        self.gridLayoutWidget_4 = QtWidgets.QWidget(self.groupBorad)
        self.gridLayoutWidget_4.setGeometry(QtCore.QRect(0, 20, 201, 61))
        self.gridLayoutWidget_4.setObjectName("gridLayoutWidget_4")
        self.gridLayoutborad = QtWidgets.QGridLayout(self.gridLayoutWidget_4)
        self.gridLayoutborad.setContentsMargins(0, 0, 0, 0)
        self.gridLayoutborad.setObjectName("gridLayoutborad")
        self.checkborad = QtWidgets.QLineEdit(self.gridLayoutWidget_4)
        self.checkborad.setObjectName("checkborad")
        self.gridLayoutborad.addWidget(self.checkborad, 0, 1, 1, 1)
        self.label_10 = QtWidgets.QLabel(self.gridLayoutWidget_4)
        self.label_10.setObjectName("label_10")
        self.gridLayoutborad.addWidget(self.label_10, 1, 0, 1, 1)
        self.squaresize = QtWidgets.QLineEdit(self.gridLayoutWidget_4)
        self.squaresize.setObjectName("squaresize")
        self.gridLayoutborad.addWidget(self.squaresize, 1, 1, 1, 1)
        self.label_11 = QtWidgets.QLabel(self.gridLayoutWidget_4)
        self.label_11.setObjectName("label_11")
        self.gridLayoutborad.addWidget(self.label_11, 0, 0, 1, 1)
        self.verticalLayout_4.addWidget(self.groupBorad)
        self.verticalLayout_4.setStretch(0, 3)
        self.verticalLayout_4.setStretch(1, 5)
        self.verticalLayout_4.setStretch(2, 4)
        self.verticalLayout_4.setStretch(3, 2)
        self.horizontalLayout_7.addWidget(self.sideWidget)
        self.horizontalLayout_7.setStretch(0, 3)
        self.horizontalLayout_7.setStretch(1, 1)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(1)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "相机校准插件"))
        self.label_8.setText(_translate("MainWindow", "图片文件夹地址："))
        self.filepath_camimgs.setText(_translate("MainWindow", "Calibrate/cam_imgs"))
        self.filepath_edit_cam.setText(_translate("MainWindow", "浏览"))
        self.label_9.setText(_translate("MainWindow", "镜头畸变校准 ："))
        self.start_cam_cali.setText(_translate("MainWindow", "开始校准"))
        self.label_7.setText(_translate("MainWindow", "校准参数输出地址："))
        self.outpath_camcail.setText(_translate("MainWindow", "Static/cam_calibration.json"))
        self.output_cam_cali.setText(_translate("MainWindow", "输出"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("MainWindow", "镜头校准"))
        self.label.setText(_translate("MainWindow", "图片文件夹地址："))
        self.filepath_laserimgs.setText(_translate("MainWindow", "Calibrate/laser_imgs"))
        self.filepath_edit_laser.setText(_translate("MainWindow", "浏览"))
        self.label_2.setText(_translate("MainWindow", "激光刀面校准 ："))
        self.start_laser_cali.setText(_translate("MainWindow", "开始校准"))
        self.label_3.setText(_translate("MainWindow", "校准参数输出地址："))
        self.outpath_lasercail.setText(_translate("MainWindow", "Static/laser_calibration.json"))
        self.output_laser_cali.setText(_translate("MainWindow", "输出"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("MainWindow", "刀面校准"))
        self.groupInit.setTitle(_translate("MainWindow", "初始化"))
        self.bnClose.setText(_translate("MainWindow", "关闭设备"))
        self.bnOpen.setText(_translate("MainWindow", "打开设备"))
        self.bnEnum.setText(_translate("MainWindow", "查找设备"))
        self.groupGrab.setTitle(_translate("MainWindow", "采集"))
        self.radioContinueMode.setText(_translate("MainWindow", "连续模式"))
        self.radioTriggerMode.setText(_translate("MainWindow", "触发模式"))
        self.bnSoftwareTrigger.setText(_translate("MainWindow", "软触发一次"))
        self.bnSaveImage.setText(_translate("MainWindow", "保存图像"))
        self.bnStart.setText(_translate("MainWindow", "开始采集"))
        self.bnStop.setText(_translate("MainWindow", "停止采集"))
        self.path_tips.setText(_translate("MainWindow", "保存至"))
        self.groupParam.setTitle(_translate("MainWindow", "参数"))
        self.label_6.setText(_translate("MainWindow", "帧率"))
        self.edtGain.setText(_translate("MainWindow", "0"))
        self.label_5.setText(_translate("MainWindow", "增益"))
        self.label_4.setText(_translate("MainWindow", "曝光"))
        self.edtExposureTime.setText(_translate("MainWindow", "0"))
        self.bnGetParam.setText(_translate("MainWindow", "获取参数"))
        self.bnSetParam.setText(_translate("MainWindow", "设置参数"))
        self.edtFrameRate.setText(_translate("MainWindow", "0"))
        self.groupBorad.setTitle(_translate("MainWindow", "标定板"))
        self.checkborad.setText(_translate("MainWindow", "6,9"))
        self.label_10.setText(_translate("MainWindow", "标定板尺度："))
        self.squaresize.setText(_translate("MainWindow", "14.5"))
        self.label_11.setText(_translate("MainWindow", "标定板规格："))
