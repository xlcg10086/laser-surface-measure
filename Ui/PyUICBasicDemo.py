# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Ui/PyUICBasicDemo.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(766, 592)
        self.centralWidget = QtWidgets.QWidget(MainWindow)
        self.centralWidget.setObjectName("centralWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.centralWidget)
        self.horizontalLayout.setContentsMargins(11, 11, 11, 11)
        self.horizontalLayout.setSpacing(6)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.displaywidget = QtWidgets.QWidget(self.centralWidget)
        self.displaywidget.setObjectName("displaywidget")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.displaywidget)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setSpacing(6)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.Widgetcircle = QtWidgets.QTabWidget(self.displaywidget)
        self.Widgetcircle.setObjectName("Widgetcircle")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.tab)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setSpacing(6)
        self.verticalLayout.setObjectName("verticalLayout")
        self.ComboDevices = QtWidgets.QComboBox(self.tab)
        self.ComboDevices.setObjectName("ComboDevices")
        self.verticalLayout.addWidget(self.ComboDevices)
        self.widgetDisplay = QtWidgets.QWidget(self.tab)
        self.widgetDisplay.setObjectName("widgetDisplay")
        self.verticalLayout.addWidget(self.widgetDisplay)
        self.Widgetcircle.addTab(self.tab, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.tab_2)
        self.verticalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_4.setSpacing(6)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.labelradius = QtWidgets.QLabel(self.tab_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.labelradius.sizePolicy().hasHeightForWidth())
        self.labelradius.setSizePolicy(sizePolicy)
        self.labelradius.setObjectName("labelradius")
        self.verticalLayout_4.addWidget(self.labelradius)
        self.widgetradius3d = QtWidgets.QWidget(self.tab_2)
        self.widgetradius3d.setObjectName("widgetradius3d")
        self.verticalLayout_4.addWidget(self.widgetradius3d)
        self.Widgetcircle.addTab(self.tab_2, "")
        self.tab_5 = QtWidgets.QWidget()
        self.tab_5.setObjectName("tab_5")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout(self.tab_5)
        self.verticalLayout_6.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_6.setSpacing(0)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.widgetradius = QtWidgets.QWidget(self.tab_5)
        self.widgetradius.setObjectName("widgetradius")
        self.verticalLayout_6.addWidget(self.widgetradius)
        self.Widgetcircle.addTab(self.tab_5, "")
        self.verticalLayout_2.addWidget(self.Widgetcircle)
        self.tabWidget_2 = QtWidgets.QTabWidget(self.displaywidget)
        self.tabWidget_2.setObjectName("tabWidget_2")
        self.tab_3 = QtWidgets.QWidget()
        self.tab_3.setObjectName("tab_3")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.tab_3)
        self.verticalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_3.setSpacing(6)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.widget_4 = QtWidgets.QWidget(self.tab_3)
        self.widget_4.setObjectName("widget_4")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.widget_4)
        self.horizontalLayout_3.setContentsMargins(3, 3, 3, 3)
        self.horizontalLayout_3.setSpacing(6)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.label_7 = QtWidgets.QLabel(self.widget_4)
        self.label_7.setObjectName("label_7")
        self.horizontalLayout_3.addWidget(self.label_7)
        self.inpath_camcail = QtWidgets.QLineEdit(self.widget_4)
        self.inpath_camcail.setObjectName("inpath_camcail")
        self.horizontalLayout_3.addWidget(self.inpath_camcail)
        self.openFileA = QtWidgets.QPushButton(self.widget_4)
        self.openFileA.setObjectName("openFileA")
        self.horizontalLayout_3.addWidget(self.openFileA)
        self.input_cam_cali = QtWidgets.QPushButton(self.widget_4)
        self.input_cam_cali.setObjectName("input_cam_cali")
        self.horizontalLayout_3.addWidget(self.input_cam_cali)
        self.verticalLayout_3.addWidget(self.widget_4)
        self.widget_5 = QtWidgets.QWidget(self.tab_3)
        self.widget_5.setObjectName("widget_5")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.widget_5)
        self.horizontalLayout_4.setContentsMargins(3, 3, 3, 3)
        self.horizontalLayout_4.setSpacing(6)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.label_8 = QtWidgets.QLabel(self.widget_5)
        self.label_8.setObjectName("label_8")
        self.horizontalLayout_4.addWidget(self.label_8)
        self.inpath_lasercail = QtWidgets.QLineEdit(self.widget_5)
        self.inpath_lasercail.setObjectName("inpath_lasercail")
        self.horizontalLayout_4.addWidget(self.inpath_lasercail)
        self.openFileB = QtWidgets.QPushButton(self.widget_5)
        self.openFileB.setObjectName("openFileB")
        self.horizontalLayout_4.addWidget(self.openFileB)
        self.input_laser_cali = QtWidgets.QPushButton(self.widget_5)
        self.input_laser_cali.setObjectName("input_laser_cali")
        self.horizontalLayout_4.addWidget(self.input_laser_cali)
        self.verticalLayout_3.addWidget(self.widget_5)
        self.widget = QtWidgets.QWidget(self.tab_3)
        self.widget.setObjectName("widget")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.widget)
        self.horizontalLayout_2.setContentsMargins(5, 5, 5, 5)
        self.horizontalLayout_2.setSpacing(6)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.btnstartcheck = QtWidgets.QPushButton(self.widget)
        self.btnstartcheck.setObjectName("btnstartcheck")
        self.horizontalLayout_2.addWidget(self.btnstartcheck)
        self.savelinescloud3d = QtWidgets.QPushButton(self.widget)
        self.savelinescloud3d.setObjectName("savelinescloud3d")
        self.horizontalLayout_2.addWidget(self.savelinescloud3d)
        self.savelinescloud2d = QtWidgets.QPushButton(self.widget)
        self.savelinescloud2d.setObjectName("savelinescloud2d")
        self.horizontalLayout_2.addWidget(self.savelinescloud2d)
        self.verticalLayout_3.addWidget(self.widget)
        self.tabWidget_2.addTab(self.tab_3, "")
        self.tab_4 = QtWidgets.QWidget()
        self.tab_4.setObjectName("tab_4")
        self.text_length = QtWidgets.QLabel(self.tab_4)
        self.text_length.setGeometry(QtCore.QRect(10, 40, 491, 16))
        self.text_length.setObjectName("text_length")
        self.btntestedge = QtWidgets.QPushButton(self.tab_4)
        self.btntestedge.setGeometry(QtCore.QRect(10, 10, 75, 23))
        self.btntestedge.setObjectName("btntestedge")
        self.editedgeheight = QtWidgets.QLineEdit(self.tab_4)
        self.editedgeheight.setGeometry(QtCore.QRect(100, 10, 113, 21))
        self.editedgeheight.setInputMask("")
        self.editedgeheight.setObjectName("editedgeheight")
        self.btntestcircle = QtWidgets.QPushButton(self.tab_4)
        self.btntestcircle.setGeometry(QtCore.QRect(10, 60, 75, 23))
        self.btntestcircle.setObjectName("btntestcircle")
        self.text_circle = QtWidgets.QLabel(self.tab_4)
        self.text_circle.setGeometry(QtCore.QRect(10, 90, 491, 16))
        self.text_circle.setObjectName("text_circle")
        self.tabWidget_2.addTab(self.tab_4, "")
        self.verticalLayout_2.addWidget(self.tabWidget_2)
        self.verticalLayout_2.setStretch(0, 3)
        self.verticalLayout_2.setStretch(1, 1)
        self.horizontalLayout.addWidget(self.displaywidget)
        self.sideWidget = QtWidgets.QWidget(self.centralWidget)
        self.sideWidget.setObjectName("sideWidget")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.sideWidget)
        self.verticalLayout_5.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_5.setSpacing(6)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.groupInit = QtWidgets.QGroupBox(self.sideWidget)
        self.groupInit.setObjectName("groupInit")
        self.verticalLayout_7 = QtWidgets.QVBoxLayout(self.groupInit)
        self.verticalLayout_7.setContentsMargins(2, 2, 2, 2)
        self.verticalLayout_7.setSpacing(2)
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setSpacing(6)
        self.gridLayout.setObjectName("gridLayout")
        self.bnOpen = QtWidgets.QPushButton(self.groupInit)
        self.bnOpen.setObjectName("bnOpen")
        self.gridLayout.addWidget(self.bnOpen, 2, 1, 1, 1)
        self.bnEnum = QtWidgets.QPushButton(self.groupInit)
        self.bnEnum.setObjectName("bnEnum")
        self.gridLayout.addWidget(self.bnEnum, 1, 1, 1, 2)
        self.bnClose = QtWidgets.QPushButton(self.groupInit)
        self.bnClose.setEnabled(False)
        self.bnClose.setObjectName("bnClose")
        self.gridLayout.addWidget(self.bnClose, 2, 2, 1, 1)
        self.verticalLayout_7.addLayout(self.gridLayout)
        self.verticalLayout_5.addWidget(self.groupInit)
        self.groupGrab = QtWidgets.QGroupBox(self.sideWidget)
        self.groupGrab.setEnabled(False)
        self.groupGrab.setObjectName("groupGrab")
        self.verticalLayout_8 = QtWidgets.QVBoxLayout(self.groupGrab)
        self.verticalLayout_8.setContentsMargins(2, 2, 2, 2)
        self.verticalLayout_8.setSpacing(2)
        self.verticalLayout_8.setObjectName("verticalLayout_8")
        self.gridLayout_2 = QtWidgets.QGridLayout()
        self.gridLayout_2.setSpacing(6)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.radioContinueMode = QtWidgets.QRadioButton(self.groupGrab)
        self.radioContinueMode.setObjectName("radioContinueMode")
        self.gridLayout_2.addWidget(self.radioContinueMode, 0, 0, 1, 1)
        self.radioTriggerMode = QtWidgets.QRadioButton(self.groupGrab)
        self.radioTriggerMode.setObjectName("radioTriggerMode")
        self.gridLayout_2.addWidget(self.radioTriggerMode, 0, 1, 1, 1)
        self.bnSoftwareTrigger = QtWidgets.QPushButton(self.groupGrab)
        self.bnSoftwareTrigger.setEnabled(False)
        self.bnSoftwareTrigger.setObjectName("bnSoftwareTrigger")
        self.gridLayout_2.addWidget(self.bnSoftwareTrigger, 3, 0, 1, 2)
        self.bnSaveImage = QtWidgets.QPushButton(self.groupGrab)
        self.bnSaveImage.setEnabled(False)
        self.bnSaveImage.setObjectName("bnSaveImage")
        self.gridLayout_2.addWidget(self.bnSaveImage, 4, 0, 1, 2)
        self.bnStart = QtWidgets.QPushButton(self.groupGrab)
        self.bnStart.setEnabled(False)
        self.bnStart.setObjectName("bnStart")
        self.gridLayout_2.addWidget(self.bnStart, 2, 0, 1, 1)
        self.bnStop = QtWidgets.QPushButton(self.groupGrab)
        self.bnStop.setEnabled(False)
        self.bnStop.setObjectName("bnStop")
        self.gridLayout_2.addWidget(self.bnStop, 2, 1, 1, 1)
        self.verticalLayout_8.addLayout(self.gridLayout_2)
        self.savePath = QtWidgets.QLabel(self.groupGrab)
        self.savePath.setObjectName("savePath")
        self.verticalLayout_8.addWidget(self.savePath)
        self.verticalLayout_5.addWidget(self.groupGrab)
        self.groupParam = QtWidgets.QGroupBox(self.sideWidget)
        self.groupParam.setEnabled(False)
        self.groupParam.setObjectName("groupParam")
        self.verticalLayout_9 = QtWidgets.QVBoxLayout(self.groupParam)
        self.verticalLayout_9.setContentsMargins(2, 2, 2, 2)
        self.verticalLayout_9.setSpacing(2)
        self.verticalLayout_9.setObjectName("verticalLayout_9")
        self.gridLayoutParam = QtWidgets.QGridLayout()
        self.gridLayoutParam.setSpacing(6)
        self.gridLayoutParam.setObjectName("gridLayoutParam")
        self.label_6 = QtWidgets.QLabel(self.groupParam)
        self.label_6.setObjectName("label_6")
        self.gridLayoutParam.addWidget(self.label_6, 3, 0, 1, 1)
        self.edtGain = QtWidgets.QLineEdit(self.groupParam)
        self.edtGain.setObjectName("edtGain")
        self.gridLayoutParam.addWidget(self.edtGain, 1, 1, 1, 1)
        self.label_5 = QtWidgets.QLabel(self.groupParam)
        self.label_5.setObjectName("label_5")
        self.gridLayoutParam.addWidget(self.label_5, 1, 0, 1, 1)
        self.label_4 = QtWidgets.QLabel(self.groupParam)
        self.label_4.setObjectName("label_4")
        self.gridLayoutParam.addWidget(self.label_4, 0, 0, 1, 1)
        self.edtExposureTime = QtWidgets.QLineEdit(self.groupParam)
        self.edtExposureTime.setObjectName("edtExposureTime")
        self.gridLayoutParam.addWidget(self.edtExposureTime, 0, 1, 1, 1)
        self.bnGetParam = QtWidgets.QPushButton(self.groupParam)
        self.bnGetParam.setObjectName("bnGetParam")
        self.gridLayoutParam.addWidget(self.bnGetParam, 4, 0, 1, 1)
        self.bnSetParam = QtWidgets.QPushButton(self.groupParam)
        self.bnSetParam.setObjectName("bnSetParam")
        self.gridLayoutParam.addWidget(self.bnSetParam, 4, 1, 1, 1)
        self.edtFrameRate = QtWidgets.QLineEdit(self.groupParam)
        self.edtFrameRate.setObjectName("edtFrameRate")
        self.gridLayoutParam.addWidget(self.edtFrameRate, 3, 1, 1, 1)
        self.gridLayoutParam.setColumnStretch(0, 2)
        self.gridLayoutParam.setColumnStretch(1, 3)
        self.verticalLayout_9.addLayout(self.gridLayoutParam)
        self.verticalLayout_5.addWidget(self.groupParam)
        self.groupBox = QtWidgets.QGroupBox(self.sideWidget)
        self.groupBox.setObjectName("groupBox")
        self.verticalLayout_10 = QtWidgets.QVBoxLayout(self.groupBox)
        self.verticalLayout_10.setContentsMargins(2, 2, 2, 2)
        self.verticalLayout_10.setSpacing(2)
        self.verticalLayout_10.setObjectName("verticalLayout_10")
        self.gridLayout_3 = QtWidgets.QGridLayout()
        self.gridLayout_3.setSpacing(6)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.linerange = QtWidgets.QLineEdit(self.groupBox)
        self.linerange.setObjectName("linerange")
        self.gridLayout_3.addWidget(self.linerange, 0, 1, 1, 1)
        self.label = QtWidgets.QLabel(self.groupBox)
        self.label.setObjectName("label")
        self.gridLayout_3.addWidget(self.label, 0, 0, 1, 1)
        self.btncutclouds = QtWidgets.QPushButton(self.groupBox)
        self.btncutclouds.setObjectName("btncutclouds")
        self.gridLayout_3.addWidget(self.btncutclouds, 1, 1, 1, 1)
        self.btnrdtclouds = QtWidgets.QPushButton(self.groupBox)
        self.btnrdtclouds.setObjectName("btnrdtclouds")
        self.gridLayout_3.addWidget(self.btnrdtclouds, 1, 0, 1, 1)
        self.gridLayout_3.setColumnStretch(0, 1)
        self.gridLayout_3.setColumnStretch(1, 1)
        self.gridLayout_3.setRowStretch(0, 1)
        self.gridLayout_3.setRowStretch(1, 1)
        self.verticalLayout_10.addLayout(self.gridLayout_3)
        self.verticalLayout_5.addWidget(self.groupBox)
        self.verticalLayout_5.setStretch(0, 3)
        self.verticalLayout_5.setStretch(1, 5)
        self.verticalLayout_5.setStretch(2, 4)
        self.verticalLayout_5.setStretch(3, 3)
        self.horizontalLayout.addWidget(self.sideWidget)
        self.horizontalLayout.setStretch(0, 10)
        self.horizontalLayout.setStretch(1, 4)
        MainWindow.setCentralWidget(self.centralWidget)

        self.retranslateUi(MainWindow)
        self.Widgetcircle.setCurrentIndex(1)
        self.tabWidget_2.setCurrentIndex(1)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.Widgetcircle.setTabText(self.Widgetcircle.indexOf(self.tab), _translate("MainWindow", "原始图像"))
        self.labelradius.setText(_translate("MainWindow", "圆半径为"))
        self.Widgetcircle.setTabText(self.Widgetcircle.indexOf(self.tab_2), _translate("MainWindow", "空间拟合"))
        self.Widgetcircle.setTabText(self.Widgetcircle.indexOf(self.tab_5), _translate("MainWindow", "平面拟合"))
        self.label_7.setText(_translate("MainWindow", "镜头校准参数："))
        self.inpath_camcail.setText(_translate("MainWindow", "Static/cam_calibration.json;camerahkvs"))
        self.openFileA.setText(_translate("MainWindow", "浏览"))
        self.input_cam_cali.setText(_translate("MainWindow", "注入"))
        self.label_8.setText(_translate("MainWindow", "激光刀面参数："))
        self.inpath_lasercail.setText(_translate("MainWindow", "Static/laser_calibration.json;camerahkvs"))
        self.openFileB.setText(_translate("MainWindow", "浏览"))
        self.input_laser_cali.setText(_translate("MainWindow", "注入"))
        self.btnstartcheck.setText(_translate("MainWindow", "开始测试"))
        self.savelinescloud3d.setText(_translate("MainWindow", "3d点云保存"))
        self.savelinescloud2d.setText(_translate("MainWindow", "2d点云保存"))
        self.tabWidget_2.setTabText(self.tabWidget_2.indexOf(self.tab_3), _translate("MainWindow", "参数导入"))
        self.text_length.setText(_translate("MainWindow", "长度为"))
        self.btntestedge.setText(_translate("MainWindow", "测边缘"))
        self.editedgeheight.setPlaceholderText(_translate("MainWindow", "输入高度差"))
        self.btntestcircle.setText(_translate("MainWindow", "测直径"))
        self.text_circle.setText(_translate("MainWindow", "长度为"))
        self.tabWidget_2.setTabText(self.tabWidget_2.indexOf(self.tab_4), _translate("MainWindow", "数值拟合"))
        self.groupInit.setTitle(_translate("MainWindow", "初始化"))
        self.bnOpen.setText(_translate("MainWindow", "打开设备"))
        self.bnEnum.setText(_translate("MainWindow", "查找设备"))
        self.bnClose.setText(_translate("MainWindow", "关闭设备"))
        self.groupGrab.setTitle(_translate("MainWindow", "采集"))
        self.radioContinueMode.setText(_translate("MainWindow", "连续模式"))
        self.radioTriggerMode.setText(_translate("MainWindow", "触发模式"))
        self.bnSoftwareTrigger.setText(_translate("MainWindow", "软触发一次"))
        self.bnSaveImage.setText(_translate("MainWindow", "保存图像"))
        self.bnStart.setText(_translate("MainWindow", "开始采集"))
        self.bnStop.setText(_translate("MainWindow", "停止采集"))
        self.savePath.setText(_translate("MainWindow", "保存至"))
        self.groupParam.setTitle(_translate("MainWindow", "参数"))
        self.label_6.setText(_translate("MainWindow", "帧率"))
        self.edtGain.setText(_translate("MainWindow", "0"))
        self.label_5.setText(_translate("MainWindow", "增益"))
        self.label_4.setText(_translate("MainWindow", "曝光"))
        self.edtExposureTime.setText(_translate("MainWindow", "0"))
        self.bnGetParam.setText(_translate("MainWindow", "获取参数"))
        self.bnSetParam.setText(_translate("MainWindow", "设置参数"))
        self.edtFrameRate.setText(_translate("MainWindow", "0"))
        self.groupBox.setTitle(_translate("MainWindow", "GroupBox"))
        self.linerange.setText(_translate("MainWindow", "50,200"))
        self.label.setText(_translate("MainWindow", "范围设定:"))
        self.btncutclouds.setText(_translate("MainWindow", "裁剪点云"))
        self.btnrdtclouds.setText(_translate("MainWindow", "还原点云"))