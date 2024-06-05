import json
import os
import numpy as np

class Camerapara():
    def __init__(self, name, matrix, distortion):
        self.name = name  # 相机名称
        self.matrix = matrix  # 相机内参矩阵
        self.distortion = distortion  # 畸变系数

def readCamsetfile(filepath ='data/camera_calibpara.json'):
    """
    读取相机设置文件，解析其中的相机名称、内参矩阵和畸变系数，返回相机对象列表。

    参数:
    filepath (str): 相机设置文件的文件名，不包含后缀名。

    返回:
    list: 包含相机对象的列表，每个相机对象有三个属性: 相机名称、内参矩阵和畸变系数。
    """
    with open(f'./{filepath}', 'r') as file:
        data_list = json.load(file)

    if len(data_list) == 0:
        return 'error: no data'

    campara_list = []
    for data in data_list:
        # 获取变量
        camera_name = data["camera_name"]
        camera_matrix = np.array(data["camera_matrix"])
        distortion_coefficients = np.array(data["distortion_coefficients"])
        
        # 创建相机对象并加入列表
        camera = Camerapara(camera_name, camera_matrix, distortion_coefficients)
        campara_list.append(camera)
    
    return campara_list

def SaveCamsetfile(campara_list: list, filepath='data/camera_calibpara.json'):
    """
    将相机参数列表保存到 JSON 文件中。

    参数:
    campara_list (list): 包含相机参数的列表。
    filename (str): 要保存的文件名，默认为 'data/camera_calibpara.json'。
    """
    # 将相机参数列表转换为字典列表
    data_list = []
    for camera in campara_list:
        data = {
            "camera_name": camera.camera_name,
            "camera_matrix": camera.camera_matrix.tolist(),  # 转换为列表以便 JSON 序列化
            "distortion_coefficients": camera.distortion_coefficients.tolist()  # 转换为列表
        }
        data_list.append(data)

    if os.path.exists(filepath):
        # 如果文件存在，加载现有数据
        with open(filepath, 'r') as file:
            existing_data = json.load(file)

        # 检查是否已存在相同名称的相机
        for cam_data in data_list:
            cam_data_name = cam_data['camera_name']
            existing_camera_names = [params['camera_name'] for params in existing_data]
            if cam_data_name in existing_camera_names:
                # 如果存在相同名称的相机，更新该相机的参数
                index = existing_camera_names.index(cam_data_name)
                existing_data[index] = cam_data
                print(f'Updated calibration parameters for {cam_data_name} in {filepath}')
            else:
                # 如果不存在相同名称的相机，添加新的参数
                existing_data.append(cam_data)
                print(f'Added calibration parameters for {cam_data_name} to {filepath}')

            # 保存更新后的数据
            with open(filepath, 'w') as file:
                json.dump(existing_data, file, indent=4)
    else:
        # 如果文件不存在，创建新文件并保存参数
        with open(filepath, 'w') as file:
            json.dump(data_list, file, indent=4)
        print(f'Calibration parameters saved to {filepath}')


def SaveCamsettingparafile(camsettingpara_list: list, filepath='Static/camera_setting_para.json'):
    """
    将相机参数列表保存到 JSON 文件中。

    参数:
    camsettingpara_list (list): 包含相机参数的列表。
    filename (str): 要保存的文件名，默认为 'data/camera_setting_para.json'。
    """
    # 将相机参数列表转换为字典列表

    if os.path.exists(filepath):
        # 如果文件存在，加载现有数据
        with open(filepath, 'r') as file:
            existing_data = json.load(file)

        # 检查是否已存在相同名称的相机
        for cam_data in camsettingpara_list:
            cam_data_name = cam_data['camera_name']
            existing_camera_names = [params['camera_name'] for params in existing_data]
            if cam_data_name in existing_camera_names:
                # 如果存在相同名称的相机，更新该相机的参数
                index = existing_camera_names.index(cam_data_name)
                existing_data[index] = cam_data
                print(f'Updated calibration parameters for {cam_data_name} in {filepath}')
            else:
                # 如果不存在相同名称的相机，添加新的参数
                existing_data.append(cam_data)
                print(f'Added calibration parameters for {cam_data_name} to {filepath}')

            # 保存更新后的数据
            with open(filepath, 'w') as file:
                json.dump(existing_data, file, indent=4)
    else:
        # 如果文件不存在，创建新文件并保存参数
        with open(filepath, 'w') as file:
            json.dump(camsettingpara_list, file, indent=4)
        print(f'Calibration parameters saved to {filepath}')

def readCamsettingparafile(filepath ='data/camera_setting_para.json'):
    """
    读取相机设置文件，解析其中的相机名称、内参矩阵和畸变系数，返回相机对象列表。

    参数:
    filepath (str): 相机设置文件的文件名，不包含后缀名。

    返回:
    list: 包含相机对象的列表，每个相机对象有三个属性: 相机名称、内参矩阵和畸变系数。
    """
    if os.path.exists(filepath):
        with open(f'./{filepath}', 'r') as file:
            data_list = json.load(file)

        if len(data_list) == 0:
            print('error: no data') 
            return None

        cam_setting_para_list = data_list

        return cam_setting_para_list
    else:
        print('error: no such file') 
        return None


def Savelasersurfaceparafile(lasersurface_list: list, filepath='data/laser_surface_para.json'):
    """
    将激光面列表保存到 JSON 文件中。

    参数:
    lasersurface_list (list): 包含激光面参数的列表,列表组成为字典,包含的key包括laser_name,A,B,C。
    filepath (str): 要保存的文件名，默认为 'data/laser_surface_para.json'。
    """
    # 将相机参数列表转换为字典列表

    if os.path.exists(filepath):
        # 如果文件存在，加载现有数据
        with open(filepath, 'r') as file:
            existing_data = json.load(file)

        # 检查是否已存在相同名称的相机
        for laser_data in lasersurface_list:
            laser_data_name = laser_data['laser_name']
            existing_laser_names = [params['laser_name'] for params in existing_data]
            if laser_data_name in existing_laser_names:
                # 如果存在相同名称的相机，更新该相机的参数
                index = existing_laser_names.index(laser_data_name)
                existing_data[index] = laser_data
                print(f'Updated calibration parameters for {laser_data_name} in {filepath}')
            else:
                # 如果不存在相同名称的相机，添加新的参数
                existing_data.append(laser_data)
                print(f'Added calibration parameters for {laser_data_name} to {filepath}')

            # 保存更新后的数据
            with open(filepath, 'w') as file:
                json.dump(existing_data, file, indent=4)
    else:
        # 如果文件不存在，创建新文件并保存参数
        with open(filepath, 'w') as file:
            json.dump(lasersurface_list, file, indent=4)
        print(f'Calibration parameters saved to {filepath}')


def Readlasersurfaceparafile(filepath ='data/laser_surface_para.json'):
    """
    读取激光面设置文件，获取激光面文件列表。

    参数:
    filepath (str): 相机设置文件的文件名，不包含后缀名。

    返回:
    list: 包含激光面参数的列表，每个激光面有4个key laser_name,A,B,C。
    """
    if os.path.exists(filepath):
        with open(f'./{filepath}', 'r') as file:
            data_list = json.load(file)

        if len(data_list) == 0:
            print('error: no data') 
            return None

        laser_surface_para_list = data_list

        return laser_surface_para_list
    else:
        print('error: no such file') 
        return None
    

def SaveCamcalibrationparameters(file_path,camera_name, mtx, dist, ret):
    # 构建字典保存参数
    calibration_params = {
    'camera_name': camera_name,
    'mat': mtx.tolist(),
    'dist': dist.tolist(),
    'Conf': ret
    }

    # 检查文件是否存在
    if os.path.exists(file_path):
    # 如果文件存在，加载现有数据
        with open(file_path, 'r') as file:
            try:
                existing_data = json.load(file)
                # 确保现有数据是列表格式
                if not isinstance(existing_data, list):
                    existing_data = []
            except json.JSONDecodeError:
                # 如果JSON文件损坏或为空，则创建一个空列表
                existing_data = []

        # 检查是否已存在相同名称的相机
        existing_camera_names = [params['camera_name'] for params in existing_data]
        if camera_name in existing_camera_names:
            # 如果存在相同名称的相机，更新该相机的参数
            index = existing_camera_names.index(camera_name)
            # 可以在这里添加一个条件来检查ret值是否表示成功的校准
            if ret < 3:  # 假设some_threshold是成功校准的最小值
                existing_data[index] = calibration_params
                print(f'Updated calibration parameters for {camera_name} in {file_path}')
            else:
                print(f'Calibration for {camera_name} was not successful enough to update.')
        else:
            # 如果不存在相同名称的相机，添加新的参数
            existing_data.append(calibration_params)
            print(f'Added calibration parameters for {camera_name} to {file_path}')

        # 保存更新后的数据
        with open(file_path, 'w') as file:
            json.dump(existing_data, file, indent=4)
    else:
        # 如果文件不存在，创建新文件并保存参数
        with open(file_path, 'w') as file:
            json.dump([calibration_params], file, indent=4)
        print(f'Calibration parameters saved to {file_path}')