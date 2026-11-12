"""vision_inspect 配置模块:参数结构与 config.json 持久化。

所有可调参数集中在这里,UI/处理线程/报警线程共享同一份 AppConfig 实例。
线程约定:主线程(GUI)写参数,采集线程每帧读参数快照 —— 数值类型(int/float/bool/tuple)
在 CPython 下赋值是原子的,采集线程一次读取看到的是一致的值。
"""
from __future__ import annotations

import copy
import json
import os
from dataclasses import asdict, dataclass, field

# 报警触发帧(串口十六进制)
ALARM_FRAME = bytes([0x07, 0x00, 0x00, 0xA7])


@dataclass
class ChannelWindow:
    """单个颜色通道的阈值窗口 [lo, hi]"""

    lo: int = 180
    hi: int = 255

    def in_range(self, v: int) -> bool:
        return self.lo <= v <= self.hi

    def as_tuple(self) -> tuple[int, int]:
        return (self.lo, self.hi)


@dataclass
class AppConfig:
    # ---- 信号源 ----
    source: str = "sim"            # "sim" 模拟器 / "image" 图片 / "hik" 海康相机
    image_path: str = ""           # 图片源文件
    sim_defect: str = "white"      # 模拟缺陷类型: "none"/"white"/"red"/"green"/"blue"
    sim_defect_w: int = 60         # 模拟缺陷尺寸
    sim_defect_h: int = 40

    # ---- ROI(原始图像坐标)----
    roi_x: int = 0
    roi_y: int = 0
    roi_w: int = 1280
    roi_h: int = 1024

    # ---- 三通道阈值窗口(OpenCV 顺序 B,G,R;界面按红绿蓝显示)----
    window_b: ChannelWindow = field(default_factory=lambda: ChannelWindow(180, 255))
    window_g: ChannelWindow = field(default_factory=lambda: ChannelWindow(180, 255))
    window_r: ChannelWindow = field(default_factory=lambda: ChannelWindow(180, 255))

    # ---- 报警 ----
    alarm_count: int = 800         # 报警阈值:ROI 内三通道同时达标像素数 N >= 此值
    debounce_frames: int = 3       # 连续多少帧超阈值才真正报警
    show_mask: bool = True         # 画面叠加"符合像素"红色蒙层

    # ---- 相机参数(仅 hik 源生效;曝光 us / 增益 dB / 帧率 fps)----
    exposure_us: int = 2000
    gain_db: float = 0.0
    target_fps: int = 30

    # ---- 串口报警设备 ----
    use_serial: bool = True        # False = 仅写日志,不碰串口
    serial_port: str = "COM2"
    serial_baud: int = 9600

    @property
    def roi(self) -> tuple[int, int, int, int]:
        return (self.roi_x, self.roi_y, self.roi_w, self.roi_h)

    @property
    def windows(self) -> tuple[ChannelWindow, ChannelWindow, ChannelWindow]:
        return (self.window_b, self.window_g, self.window_r)


PRESETS: dict[str, tuple[tuple[int, int], tuple[int, int], tuple[int, int]]] = {
    # 名称: (B窗口, G窗口, R窗口)  -- 界面按红绿蓝编辑,内部 BGR
    "白色高亮": ((180, 255), (180, 255), (180, 255)),
    "红色亮斑": ((0, 120), (0, 120), (180, 255)),
    "绿色亮斑": ((0, 120), (180, 255), (0, 120)),
    "蓝色亮斑": ((180, 255), (0, 120), (0, 120)),
    "全白(纯白)": ((230, 255), (230, 255), (230, 255)),
    "最暗背景": ((0, 60), (0, 60), (0, 60)),
}


class ConfigStore:
    """读写 config.json。GUI 每次变更可调 save() 落盘。"""

    def __init__(self, path: str | os.PathLike):
        self.path = os.fspath(path)

    def load(self) -> AppConfig:
        cfg = AppConfig()
        if not os.path.exists(self.path):
            return cfg
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                d = json.load(f)
            # 逐字段合并,容忍缺字段/类型变化
            merged = asdict(cfg)
            for k, v in d.items():
                if k in merged and isinstance(v, type(merged[k])):
                    merged[k] = v
            for ch in ("window_b", "window_g", "window_r"):
                if ch in d and isinstance(d[ch], dict):
                    setattr(cfg, ch, ChannelWindow(int(d[ch].get("lo", 180)), int(d[ch].get("hi", 255))))
            for k in ("roi_x", "roi_y", "roi_w", "roi_h", "alarm_count", "debounce_frames",
                      "exposure_us", "target_fps", "serial_baud"):
                if k in d:
                    setattr(cfg, k, int(d[k]))
            for k in ("gain_db",):
                if k in d:
                    setattr(cfg, k, float(d[k]))
            for k in ("use_serial", "show_mask"):
                if k in d:
                    setattr(cfg, k, bool(d[k]))
            for k in ("source", "image_path", "sim_defect", "serial_port"):
                if k in d:
                    setattr(cfg, k, str(d[k]))
            if not (isinstance(cfg.sim_defect_w, int) and cfg.sim_defect_w > 0):
                cfg.sim_defect_w = 60
            if not (isinstance(cfg.sim_defect_h, int) and cfg.sim_defect_h > 0):
                cfg.sim_defect_h = 40
        except Exception as e:  # 配置损坏时用默认值,不让程序起不来
            print(f"[config] 配置文件解析失败,使用默认参数: {e}")
            cfg = AppConfig()
        return cfg

    def save(self, cfg: AppConfig) -> None:
        d = asdict(cfg)
        try:
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(d, f, ensure_ascii=False, indent=2)
            os.replace(tmp, self.path)
        except Exception as e:
            print(f"[config] 保存失败: {e}")


def clamp_roi_to_image(roi: tuple[int, int, int, int], w: int, h: int) -> tuple[int, int, int, int]:
    """把 ROI 夹到图像范围内,返回 (x,y,w,h);空 ROI 返回 (0,0,w,h)。"""
    x, y, rw, rh = roi
    x = max(0, min(int(x), w - 1))
    y = max(0, min(int(y), h - 1))
    rw = max(1, min(int(rw), w - x))
    rh = max(1, min(int(rh), h - y))
    return (x, y, rw, rh)


def snapshot_name(prefix: str = "alarm") -> str:
    import time

    return f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}.png"


def cfg_snapshot(cfg: AppConfig) -> AppConfig:
    """深拷贝一份快照,供采集线程读取,避免 UI 线程中途修改导致半新半旧。"""
    return copy.deepcopy(cfg)
