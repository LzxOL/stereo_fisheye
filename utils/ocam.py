import torch
import numpy as np
from scipy.io import loadmat
from easydict import EasyDict as Edict
from utils.log import *
from utils.common import *
from utils.geometry import *

# 标定驱动的多项式鱼眼模型；不是针孔模型的 fx*X/Z 投影。
class OcamModel:
    def __init__(self): pass
    def setConfig(self, cfg):
        self.id = cfg['cam_id']
        num_pol = cfg['poly'][0]
        if len(cfg['poly']) - 1 != num_pol :
            LOG_WARNING('Number of coeffs does not match in ocam\'s poly')
        self.poly = cfg['poly'][-1:0:-1]
        num_invpol = cfg['inv_poly'][0]
        if len(cfg['inv_poly']) - 1 != num_invpol :
            LOG_WARNING('Number of coeffs does not match in ocam\'s inv_poly')
        self.inv_poly = cfg['inv_poly'][-1:0:-1]
        self.xc, self.yc = cfg['center']
        self.c, self.d, self.e = cfg['affine']
        self.height, self.width = cfg['image_size']
        self.max_theta = np.deg2rad(cfg['max_fov']) / 2.0
        self.invalid_mask_file = cfg['invalid_mask']
        # pose 前三项为 Rodrigues 旋转向量，后三项为平移；随后求逆用于 rig -> cam。
        self.cam2rig = np.array(cfg['pose']).reshape((6, 1))
        self.rig2cam = inverseTransform(self.cam2rig)
    def pixelToRay(self, p, out_theta=False, max_theta=None):
        # 鱼眼像素 [2,N] -> 相机射线 [3,N]；超出角视场的方向置 NaN，供生成输入掩码。
        if max_theta is None: max_theta = self.max_theta
        x = p[1,:].reshape((1, -1)) - self.xc
        y = p[0,:].reshape((1, -1)) - self.yc
        p = concat((x, y), axis=0)
        invdet = 1.0 / (self.c - self.d * self.e)
        A_inv = invdet * np.array([
            [      1, -self.d], 
            [-self.e,  self.c]]) 
        p = A_inv.dot(p)
        x = p[1,:].reshape((1, -1))
        y = p[0,:].reshape((1, -1))
        rho = sqrt(x * x + y * y)
        z = polyval(self.poly, rho).reshape((1, -1))
        theta = atan2(rho, -z) 
        out = concat((x, y, -z), axis=0)
        out[:,theta.squeeze() > max_theta] = np.nan
        if out_theta:
            return out, theta
        else:
            return out
    def rayToPixel(self, P, out_theta=False, max_theta=None):
        # Spherical Sweeping 的投影核心：相机坐标 [3,H*W] -> 像素 [2,H*W]。
        # 注意代码内部的 x2 是图像行，y2 是列，故输出顺序 (y2,x2) 才是 (u,v)。
        if max_theta is None: max_theta = self.max_theta
        norm = sqrt(P[0,:]**2 + P[1,:]**2) + EPS
        theta = atan2(-P[2,:], norm)
        # inv_poly 将射线角映射成鱼眼径向像素距离 rho；它不是逆深度 rho。
        rho = polyval(self.inv_poly, theta)
        theta = theta + np.pi / 2
        x = P[1,:] / norm * rho
        y = P[0,:] / norm * rho
        # 仿射参数 c/d/e 与中心 xc/yc 修正传感器的轴比例、倾斜和主点偏移。
        x2 = x * self.c + y * self.d + self.xc
        y2 = x * self.e + y          + self.yc
        x2 = x2.reshape((1, -1))
        y2 = y2.reshape((1, -1))
        out = concat((y2, x2), axis=0)
        # 视场外用很大的负像素坐标编码；LUT 中将其变为 [-2,-2] 以触发零采样。
        # 此判断只检查角视场，不检测场景遮挡或输入 invalid_mask 的任意形状缺口。
        out[:,theta.squeeze() > max_theta] = -1e5
        if out_theta:
            return out, theta
        else:
            return out
    def makeInvisibleMask(self) -> np.ndarray:
        xs, ys = np.meshgrid(range(self.width), range(self.height))
        p = np.concatenate((xs.reshape((1, -1)), ys.reshape((1, -1))), axis=0)
        ray = self.pixelToRay(p)
        invisible = np.isnan(ray[0, :])
        return invisible.reshape((self.height, self.width)).astype(bool)
