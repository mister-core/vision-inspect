"""图像源抽象层。

FrameSource 基类 + 三个实现:
  SimSource    模拟器:在暗色"工件"背景上放置亮色缺陷,无需硬件即可
               演示完整的 检测→报警 链路。
  ImageSource  读取一张真实产品照片(离线调参用)。
  HikSource    海康 MV 相机(MVS SDK,新版 MvCamera 封装)。
               本机未装 MVS / 未接相机时,open() 会抛出带指引的 RuntimeError。

约定:每帧返回 BGR ndarray(H,W,3,uint8)。停止时 read() 返回 None。
"""
from __future__ import annotations

import ctypes
import os
import time
from typing import Optional

import cv2
import numpy as np

SimColor = {"white": (255, 255, 255), "red": (0, 0, 255),
            "green": (0, 255, 0), "blue": (255, 0, 0)}  # BGR


class FrameSource:
    name = "base"
    is_camera = False

    def __init__(self) -> None:
        self._opened = False

    def open(self) -> None:
        self._opened = True

    def read(self) -> Optional[np.ndarray]:
        raise NotImplementedError

    def set_exposure(self, us: int) -> None:
        pass

    def set_gain(self, db: float) -> None:
        pass

    def set_fps(self, fps: int) -> None:
        pass

    def close(self) -> None:
        self._opened = False


class SimSource(FrameSource):
    """程序化合成 1280x1024 暗背景,带少量灰色走线反光;
    缺陷块(白色/红/绿/蓝)按参数出现,位置每次随机。"""

    name = "sim"
    W, H = 1280, 1024

    def __init__(self) -> None:
        super().__init__()
        self.defect = "none"   # none/white/red/green/blue
        self.defect_w, self.defect_h = 60, 40
        self.defect_xy: Optional[tuple[int, int]] = None
        self._frame_idx = 0
        self._base = self._build_base()

    def _build_base(self) -> np.ndarray:
        img = np.full((self.H, self.W, 3), (32, 34, 40), np.uint8)  # 深灰蓝背景
        rng = np.random.default_rng(7)
        # 几条"走线":浅灰(~150),低于白色窗口下限 180,不会误报
        for _ in range(14):
            y = int(rng.integers(50, self.H - 50))
            x0 = int(rng.integers(0, self.W - 300))
            cv2.line(img, (x0, y), (x0 + int(rng.integers(200, 600)), y),
                     (int(rng.integers(120, 165)),) * 3, int(rng.integers(1, 3)))
        # 一个中等亮度的"反光点",亮度刻意压在 178 ~ 180 边界,接近但不越限
        cx, cy = int(rng.integers(200, self.W - 200)), int(rng.integers(200, self.H - 200))
        cv2.circle(img, (cx, cy), 12, (178, 178, 178), -1)
        return img

    def open(self) -> None:
        super().open()
        self._frame_idx = 0

    def set_defect(self, kind: str) -> None:
        if kind not in SimColor:
            kind = "none"
        self.defect = kind
        if kind == "none":
            self.defect_xy = None
        else:
            rng = np.random.default_rng()
            self.defect_xy = (int(rng.integers(100, self.W - self.defect_w - 100)),
                              int(rng.integers(100, self.H - self.defect_h - 100)))

    def set_defect_size(self, w: int, h: int) -> None:
        self.defect_w, self.defect_h = max(8, w), max(8, h)
        if self.defect != "none":
            self.set_defect(self.defect)  # 重新随机位置

    def read(self) -> Optional[np.ndarray]:
        if not self._opened:
            return None
        self._frame_idx += 1
        img = self._base.copy()
        if self.defect != "none" and self.defect_xy is not None:
            x, y = self.defect_xy
            color = SimColor[self.defect]
            cv2.rectangle(img, (x, y), (x + self.defect_w, y + self.defect_h), color, -1)
            # 缺陷边缘轻微羽化,更像真实反光边缘
            cv2.rectangle(img, (x + 3, y + 3), (x + self.defect_w - 3, y + self.defect_h - 3),
                          tuple(int(c * 0.92) for c in color), -1)
        time.sleep(0.02)  # 模拟 ~30fps
        return img


class ImageSource(FrameSource):
    """单张图片源:不停返回同一帧(可以拖进来调参)。"""

    name = "image"

    def __init__(self, path: str) -> None:
        super().__init__()
        self.path = path

    def open(self) -> None:
        super().open()
        img = cv2.imread(self.path, cv2.IMREAD_COLOR)
        if img is None:
            self._opened = False
            raise RuntimeError(f"无法读取图片: {self.path}")
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        self._frame = img

    def read(self) -> Optional[np.ndarray]:
        return self._frame if self._opened else None


class HikSource(FrameSource):
    """海康 MV 相机取流 —— 适配【新版 MVS Python 封装】(MvCamera 类,实例句柄)。

    与旧版"MV_CC_HANDLE + 模块函数"的封装完全不同,API 差异:
      - 设备枚举: MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)     [类方法]
      - 连接:     cam = MvCamera(); cam.MV_CC_CreateHandle(设备信息);
                  cam.MV_CC_OpenDevice()  (句柄封装在实例内,不再手传)
      - 取流:     MV_CC_StartGrabbing() / MV_CC_GetOneFrameTimeout(缓存, 大小, MV_FRAME_OUT_INFO_EX, 超时)
      - 参数:     cam.MV_CC_SetFloatValue("ExposureTime"/"Gain"/"AcquisitionFrameRate", v)
                  cam.MV_CC_SetEnumValue("PixelFormat", PixelType_Gvsp_BayerGB8)
    彩色相机强制像素格式 BayerGB8,逐帧按 stFrameInfo.enPixelType 转 BGR;
    若现场 Viewer 里相机不支持 BayerGB8(极少见),在 _ensure_pixel_format 中改枚举值即可。
    """

    name = "hik"
    is_camera = True

    def __init__(self) -> None:
        super().__init__()
        self.mvs = None            # MvCameraControl_class 模块(懒加载)
        self.cam = None            # MvCamera 实例
        self._buf = None           # 取帧缓存(ctypes 数组)
        self._buf_size = 16 << 20  # 16MB,1280x1024 全格式都够
        self._finfo = None         # MV_FRAME_OUT_INFO_EX
        self._grabbing = False
        # 像素格式 → (每像素字节数, cv2 转换或 None=直接reshape)
        self._px_map: dict[int, tuple[int, int | None]] = {}
        self._warned = set()

    # ---------------- MVS 加载 ----------------
    def _load_mvs(self) -> None:
        if self.mvs is not None:
            return
        import sys

        mvimport = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "MvImport")
        if not os.path.isdir(mvimport):
            raise RuntimeError(
                "找不到项目根目录下的 MvImport 文件夹。请把 MVS 安装目录下 "
                "Development\\Samples\\Python\\MvImport 整个文件夹复制到本项目根目录后重试。")
        if mvimport not in sys.path:
            sys.path.insert(0, mvimport)
        try:
            import MvCameraControl_class  # noqa: PLC0415
            self.mvs = MvCameraControl_class
        except Exception as e:
            raise RuntimeError(
                f"MVS Python 封装加载失败({e})。请确认: 1) MVS 已安装; 2) MvCameraControl.dll "
                f"所在目录(通常是 C:\\Program Files (x86)\\Common Files\\MVS\\Runtime\\Win64_x64) "
                f"已加入系统 PATH(重装 MVS 或手动加后重启程序); 3) 上面提到的 MvImport 已复制。") from e

    def _px_code(self, en: int):
        """返回 (bytes_per_px, cv2_code|None)。None 表示直接 reshape 成 BGR。"""
        m = self.mvs
        table = [
            ("PixelType_Gvsp_BayerGB8", cv2.COLOR_BayerGB2BGR, 1),
            ("PixelType_Gvsp_BayerRG8", cv2.COLOR_BayerRG2BGR, 1),
            ("PixelType_Gvsp_BayerGR8", cv2.COLOR_BayerGR2BGR, 1),
            ("PixelType_Gvsp_BayerBG8", cv2.COLOR_BayerBG2BGR, 1),
            ("PixelType_Gvsp_Mono8", None, 1),
            ("PixelType_Gvsp_RGB8_Packed", cv2.COLOR_RGB2BGR, 3),
            ("PixelType_Gvsp_BGR8_Packed", None, 3),
        ]
        for name, code, bpp in table:
            v = getattr(m, name, None)
            if v is not None:
                self._px_map[v] = (bpp, code)
        return self._px_map.get(en)

    # ---------------- 设备枚举(信息用,连接默认取第 0 台) ----------------
    def list_devices(self) -> list[str]:
        self._load_mvs()
        mvs = self.mvs
        dev_list = mvs.MV_CC_DEVICE_INFO_LIST()
        ret = mvs.MvCamera.MV_CC_EnumDevices(mvs.MV_GIGE_DEVICE | mvs.MV_USB_DEVICE, dev_list)
        if ret != 0 or dev_list.nDeviceNum == 0:
            return []
        out = []
        from ctypes import POINTER, cast
        for i in range(dev_list.nDeviceNum):
            try:
                dev = cast(dev_list.pDeviceInfo[i], POINTER(mvs.MV_CC_DEVICE_INFO)).contents
                tlayer = int(dev.nTLayerType)
                if tlayer == mvs.MV_GIGE_DEVICE:
                    gige = dev.SpecialInfo.stGigEInfo
                    model = bytes(gige.chModelName).split(b"\0")[0].decode(errors="ignore")
                    ip = ".".join(str(gige.nCurrentIp >> (8 * k) & 0xFF) for k in range(4))
                    out.append(f"{model} @ {ip}")
                elif tlayer == mvs.MV_USB_DEVICE:
                    usb = dev.SpecialInfo.stUsb3VInfo
                    model = bytes(usb.chModelName).split(b"\0")[0].decode(errors="ignore")
                    out.append(f"{model} (USB3)")
                else:
                    out.append(f"设备 {i}")
            except Exception:
                out.append(f"设备 {i}")
        return out

    # ---------------- 连接 ----------------
    def open_device(self, index: int = 0) -> None:
        self._load_mvs()
        mvs = self.mvs
        from ctypes import POINTER, cast

        dev_list = mvs.MV_CC_DEVICE_INFO_LIST()
        ret = mvs.MvCamera.MV_CC_EnumDevices(mvs.MV_GIGE_DEVICE | mvs.MV_USB_DEVICE, dev_list)
        if ret != 0 or dev_list.nDeviceNum == 0:
            raise RuntimeError("未枚举到海康相机。请检查: 网线连接、相机供电、电脑网卡与相机 "
                               "IP 同网段、防火墙放行、MVS Viewer 能出图(出图后关掉 Viewer 释放独占)。")
        if index >= dev_list.nDeviceNum:
            raise RuntimeError(f"设备序号越界: {index} / {dev_list.nDeviceNum}")
        dev = cast(dev_list.pDeviceInfo[index], POINTER(mvs.MV_CC_DEVICE_INFO)).contents

        self.cam = mvs.MvCamera()
        if self.cam.MV_CC_CreateHandle(dev) != 0:
            raise RuntimeError("MV_CC_CreateHandle 失败")
        ret = self.cam.MV_CC_OpenDevice()  # 默认独占访问
        if ret != 0:
            raise RuntimeError(f"MV_CC_OpenDevice 失败(ret={ret})——相机可能被 MVS Viewer 或其他程序独占,请先关闭。")
        # 强制彩色相机输出 BayerGB8(单通道 8bit),长度 = 宽×高,转换最稳
        self._ensure_pixel_format()
        self._finfo = mvs.MV_FRAME_OUT_INFO_EX()
        self._buf = (ctypes.c_ubyte * self._buf_size)()
        print(f"[hik] 设备 {index} 已打开,取流缓存 {self._buf_size // (1 << 20)}MB")

    def _ensure_pixel_format(self) -> None:
        mvs = self.mvs
        target = getattr(mvs, "PixelType_Gvsp_BayerGB8", 17301514)
        ret = self.cam.MV_CC_SetEnumValue("PixelFormat", target)
        if ret != 0:
            # 退回按相机当前格式自适应(逐帧读 enPixelType)
            print(f"[hik] 设置 BayerGB8 失败(ret={ret}),将按相机当前像素格式自适应")

    def open(self) -> None:
        self.open_device(0)
        ret = self.cam.MV_CC_StartGrabbing()
        if ret != 0:
            raise RuntimeError(f"MV_CC_StartGrabbing 失败(ret={ret})")
        self._grabbing = True
        super().open()

    # ---------------- 参数控制 ----------------
    def _set(self, fn, *a, what: str):
        if self.cam is None:
            return
        try:
            ret = fn(*a)
            if ret != 0:
                print(f"[hik] {what} 设置失败(ret={ret})")
        except Exception as e:
            print(f"[hik] {what} 设置异常: {e}")

    def set_exposure(self, us: int) -> None:
        self._set(self.cam.MV_CC_SetFloatValue, "ExposureTime", float(us), what="曝光")

    def set_gain(self, db: float) -> None:
        self._set(self.cam.MV_CC_SetFloatValue, "Gain", float(db), what="增益")

    def set_fps(self, fps: int) -> None:
        if self.cam is None:
            return
        # 帧率控制开关(Enum: 1=开) → 目标帧率(Float)
        self._set(self.cam.MV_CC_SetEnumValue, "AcquisitionFrameRateEnable", 1, what="帧率开关")
        self._set(self.cam.MV_CC_SetFloatValue, "AcquisitionFrameRate", float(fps), what="帧率")

    # ---------------- 取流 ----------------
    def read(self) -> Optional[np.ndarray]:
        if not self._grabbing or self.cam is None:
            return None
        ret = self.cam.MV_CC_GetOneFrameTimeout(self._buf, self._buf_size, self._finfo, 500)
        if ret != 0:
            return None
        try:
            fi = self._finfo
            w, h, flen, en = int(fi.nWidth), int(fi.nHeight), int(fi.nFrameLen), int(fi.enPixelType)
            if w <= 0 or h <= 0 or flen <= 0:
                return None
            spec = self._px_code(en)
            if spec is None:
                if en not in self._warned:
                    self._warned.add(en)
                    print(f"[hik] 未知像素格式 en={en},按单通道 8bit 试转换(w={w},h={h},len={flen})")
                spec = (1, cv2.COLOR_BayerGB2BGR if flen == w * h else None)
            bpp, code = spec
            arr = np.frombuffer(self._buf, dtype=np.uint8, count=flen).copy()
            if code is None and bpp == 3:
                return arr.reshape(h, w, 3)          # 已是 BGR
            if code == cv2.COLOR_RGB2BGR:
                return cv2.cvtColor(arr.reshape(h, w, 3), code)
            if code is None and flen == w * h:
                return np.repeat(arr.reshape(h, w, 1), 3, axis=2)   # Mono8 → 伪彩
            if flen == w * h:
                return cv2.cvtColor(arr.reshape(h, w), code)        # Bayer8 → BGR
            raise RuntimeError(f"帧长异常: flen={flen} != {w * h * bpp}")
        except Exception as e:
            print(f"[hik] 帧转换失败: {e}")
            return None

    def close(self) -> None:
        if self._grabbing and self.cam is not None:
            try:
                self.cam.MV_CC_StopGrabbing()
            except Exception:
                pass
            try:
                self.cam.MV_CC_CloseDevice()
            except Exception:
                pass
            try:
                self.cam.MV_CC_DestroyHandle()
            except Exception:
                pass
        self._grabbing = False
        self.cam = None
        super().close()
