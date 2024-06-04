# __init__.py

# 导入模块
from .CameraParams_const import *
from .CameraParams_header import *
from .MvCameraControl_class import *
from .MvErrorDefine_const import *
from .PixelType_header import *
# from .CameraParams_header import *

# 定义包级别的变量
# package_variable = 42

# 定义包级别的函数
def package_introduce():
    print("相机的底层参数信息")

# # 定义外部可访问的接口
# __all__ = ['module1', 'module2', 'package_variable', 'package_function']