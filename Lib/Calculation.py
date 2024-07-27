import numpy as np
from scipy.optimize import least_squares

def FitCircle(point_data,center_x,center_y,R,ratio:int = 4):
    '''
    获取圆猜测其半径圆心
    args：
    point_data:点云数据
    center_x:猜测的圆心x位置，单位为mm
    center_y:猜测的圆心y位置，单位为mm
    R:猜测的半径，单位为mm
    ratio:比率，随机选择1/ratio之一的点参与计算

    return：
    center：圆心
    radius：半径
    '''
    # 获取矩阵的行数
    num_rows = point_data.shape[0]
    # 随机选择三分之一的行索引
    random_indices = np.random.choice(num_rows, int(num_rows / ratio), replace=False)
    # 使用索引获取相应的坐标
    selected_points = point_data[random_indices]

    # 将 (x, y) 的坐标分离
    x_data, y_data = zip(*selected_points)

    # 定义圆的方程
    def circle_residuals(params, x, y):
    # 定义拟合函数的残差
        centerX, centerY, radius = params
        return (x - centerX)**2 + (y - centerY)**2 - radius**2

    # 初始猜测值，可以根据实际情况调整
    initial_guess = (center_x, center_y, R)

    # 使用 curve_fit 进行拟合
    result = least_squares(circle_residuals, initial_guess, args=(x_data, y_data))
    # print(result)
    # 提取拟合结果
    center = np.around([result.x[0], result.x[1]],3)
    radius = np.around(result.x[2],3)

    return center, radius

def Findintersection(A, B, C, H, V, F):
    '''
    线面相交获取交点
    args：
    面方程为：z=Ax+By+C
    直线方程为：x/H = y/V = z/F
    return：
    基于当前相机光心为原点的三轴坐标[x,y,z]
    '''
    A = np.array([[A, B, -1],
    [V, -H, 0],
    [0, F, -V]])

    # 结果向量
    B = np.array([-C, 0, 0])
    # 使用 linalg.solve 方法求解
    solution = np.linalg.solve(A, B)
    return solution

def Transcordsystem(point,R,T):
    '''
    转化空间点的坐标系
    args：
    point：空间中的坐标点 
    R：从原坐标系变换到新坐标系的旋转矩阵
    T：从原坐标系变换到新坐标系的平移矩阵
    基于当前相机光心为原点的三轴坐标[x,y,z]
    return：
    Tpoint：point在新坐标系中的空间坐标
    例子：
    Tpoint = [x,y,z] = point[a,b,c] *  R[[11,12,13] + T[t1,t2,t3]
                                        [21,22,23]
                                        [31,32,33]]
    '''
    point = np.array(point)
    R = np.array(R)
    T = np.array(T)
    return np.dot(point,R)+T



