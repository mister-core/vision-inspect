"""像素处理核心:三通道窗口判定 + 高亮像素计数(纯 numpy/OpenCV,无 GUI 依赖)。

判定规则(与用户确认):
  像素 (B,G,R) 只有当  B∈[b_lo,b_hi] 且 G∈[g_lo,g_hi] 且 R∈[r_lo,r_hi]
  三个条件【同时】成立时,才计入"目标像素"(彩色高亮像素)。
ROI 内目标像素总数 N 与 报警阈值(alarm_count) 比较 → 是否超阈值。

另输出各通道【单独】达标计数(B_only/G_only/R_only)供界面诊断:
  例如白色异物三通道全达标 → 三路计数都很高;
  蓝色污渍只过 B 窗口 → 只有 B_only 高,联合计数把它排除 —— 正是三通道
  窗口相对"只看亮度"的价值所在。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .config import ChannelWindow, clamp_roi_to_image

OVERLAY_COLOR = (0, 0, 255)  # BGR 红:蒙层用


@dataclass
class PixelResult:
    n: int = 0                      # ROI 内三通道同时达标像素总数
    n_b: int = 0                    # 仅 B 通道窗口达标(诊断)
    n_g: int = 0
    n_r: int = 0
    roi: tuple[int, int, int, int] = (0, 0, 0, 0)
    roi_pixels: int = 0             # ROI 总像素数
    mask: np.ndarray = field(default_factory=lambda: np.zeros((1, 1), bool))

    @property
    def ratio(self) -> float:
        """目标像素占 ROI 比例,0~1,界面辅助显示。"""
        return self.n / self.roi_pixels if self.roi_pixels > 0 else 0.0


def count_highlight(frame: np.ndarray,
                    roi: tuple[int, int, int, int],
                    win_b: ChannelWindow, win_g: ChannelWindow, win_r: ChannelWindow,
                    blur_ksize: int = 3) -> PixelResult:
    """统计一帧 ROI 内的彩色高亮(目标色窗口)像素。

    blur_ksize>0 时先做轻高斯模糊抑制噪点(3 即 3x3,代价小、去假阳性明显)。
    """
    h, w = frame.shape[:2]
    x, y, rw, rh = clamp_roi_to_image(roi, w, h)

    roi_img = frame[y:y + rh, x:x + rw]
    if blur_ksize > 1:
        roi_img = cv2.GaussianBlur(roi_img, (blur_ksize, blur_ksize), 0)

    b, g, r = roi_img[..., 0], roi_img[..., 1], roi_img[..., 2]
    m_b = (b >= win_b.lo) & (b <= win_b.hi)
    m_g = (g >= win_g.lo) & (g <= win_g.hi)
    m_r = (r >= win_r.lo) & (r <= win_r.hi)
    mask = m_b & m_g & m_r

    res = PixelResult(
        n=int(mask.sum()),
        n_b=int(m_b.sum()),
        n_g=int(m_g.sum()),
        n_r=int(m_r.sum()),
        roi=(x, y, rw, rh),
        roi_pixels=rw * rh,
        mask=mask,
    )
    return res


def overlay_mask(frame: np.ndarray, res: PixelResult, alpha: float = 0.45) -> np.ndarray:
    """把"符合像素"叠成半透明红色蒙层(叠加到整帧原始坐标),便于调窗口时观察命中。"""
    if res.mask is None or res.mask.size <= 1:
        return frame.copy()
    x, y, rw, rh = res.roi
    out = frame.copy()
    overlay = np.zeros_like(out)
    overlay[y:y + rh, x:x + rw][res.mask] = OVERLAY_COLOR
    out = cv2.addWeighted(out, 1.0, overlay, alpha, 0)
    # 轮廓描边,看得更清楚
    m8 = np.zeros((rh, rw), np.uint8)
    m8[res.mask] = 255
    cnts, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, OVERLAY_COLOR, 1, offset=(x, y))
    return out


def draw_roi(frame: np.ndarray, roi: tuple[int, int, int, int],
             color=(0, 255, 0), label: str = "ROI") -> np.ndarray:
    out = frame.copy()
    x, y, rw, rh = roi
    cv2.rectangle(out, (x, y), (x + rw, y + rh), color, 2)
    if label:
        cv2.putText(out, label, (x, max(20, y - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, color, 2, cv2.LINE_AA)
    return out
