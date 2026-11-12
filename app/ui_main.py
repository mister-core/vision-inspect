"""视觉检查系统 —— PySide6 图形界面。

界面结构:
  左: 实时画面(VideoLabel,支持鼠标拖拽画 ROI;右键恢复全幅)
  右: 控制面板
    信号源(模拟器/图片/海康相机)、相机参数(曝光/增益/帧率)、
    三通道阈值窗口 R/G/B 各 lo~hi + 快捷预设、
    ROI 数值微调、报警阈值/去抖帧数、串口设置、报警状态灯与计数、日志。

线程模型:采集+统计在 CaptureWorker(QThread),通过 Signal 回传界面;
  所有参数放在共享 AppConfig(cfg)里,GUI 直接写字段,worker 每帧读取
  (数值字段在 CPython 下赋值原子,快照足够一致)。
"""
from __future__ import annotations

import os
import sys
import time
from datetime import datetime

import cv2
import numpy as np

from PySide6.QtCore import QRectF, Qt, QThread, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFrame,
    QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMainWindow,
    QMessageBox, QPlainTextEdit, QPushButton, QScrollArea, QSpinBox,
    QVBoxLayout, QWidget,
)

from .alarm import AlarmEngine, list_com_ports
from .config import AppConfig, ConfigStore, PRESETS, cfg_snapshot
from .processor import count_highlight, draw_roi, overlay_mask
from .sources import ImageSource, SimSource, HikSource

SNAPSHOT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "snapshots")


# --------------------------------------------------------------------------
# 视频显示控件:显示 + ROI 拖拽
# --------------------------------------------------------------------------
class VideoLabel(QWidget):
    roi_changed = Signal(int, int, int, int)  # x,y,w,h(图像原始坐标)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap: QPixmap | None = None
        self._fit = QRectF()          # 图像在控件内的实际绘制区域
        self._img_size = (1280, 1024)
        self.cfg = None
        self._drag_start: tuple[int, int] | None = None
        self._drag_cur: tuple[int, int] | None = None
        self.setMinimumSize(640, 480)
        self.setMouseTracking(False)

    def set_frame(self, bgr: np.ndarray):
        self._img_size = (bgr.shape[1], bgr.shape[0])
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0], 3 * rgb.shape[1],
                      QImage.Format.Format_RGB888).copy()
        self._pixmap = QPixmap.fromImage(qimg)
        self.update()

    def set_cfg(self, cfg: AppConfig):
        self.cfg = cfg

    def paintEvent(self, _ev):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(20, 20, 24))
        if self._pixmap is None:
            p.setPen(QColor(140, 140, 140))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "未启动采集")
            return
        # 等比缩放绘制
        fw = self.width() / self._pixmap.width()
        fh = self.height() / self._pixmap.height()
        f = min(fw, fh)
        w, h = self._pixmap.width() * f, self._pixmap.height() * f
        self._fit = QRectF((self.width() - w) / 2, (self.height() - h) / 2, w, h)
        p.drawPixmap(self._fit, self._pixmap, self._pixmap.rect())
        # ROI 框
        if self.cfg is not None:
            self._draw_rect(p, self.cfg.roi, QColor(0, 255, 0), "ROI")
        # 拖拽中的临时框
        if self._drag_start and self._drag_cur:
            x0, y0 = min(self._drag_start[0], self._drag_cur[0]), min(self._drag_start[1], self._drag_cur[1])
            x1, y1 = max(self._drag_start[0], self._drag_cur[0]), max(self._drag_start[1], self._drag_cur[1])
            self._draw_rect(p, (x0, y0, x1 - x0, y1 - y0), QColor(255, 220, 0), "新ROI")
        p.end()

    def _draw_rect(self, p: QPainter, roi, color: QColor, label: str):
        x, y, w, h = roi
        r = QRectF(self._img_x(x), self._img_y(y), self._img_x(x + w) - self._img_x(x),
                   self._img_y(y + h) - self._img_y(y))
        pen = QPen(color, 2)
        p.setPen(pen)
        p.drawRect(r)
        if self._fit.width() > 0:
            p.setPen(QPen(color, 1))
            p.setFont(QFont("Microsoft YaHei", 9))
            p.drawText(int(r.left()), max(14, int(r.top()) - 5), label + f" {w}x{h}")

    # 坐标换算
    def _img_x(self, ix: float) -> float:
        if self._fit.width() <= 0:
            return 0
        return self._fit.left() + ix / self._img_size[0] * self._fit.width()

    def _img_y(self, iy: float) -> float:
        if self._fit.height() <= 0:
            return 0
        return self._fit.top() + iy / self._img_size[1] * self._fit.height()

    def _to_img(self, px: float, py: float) -> tuple[int, int]:
        if self._fit.width() <= 0:
            return 0, 0
        ix = int((px - self._fit.left()) / self._fit.width() * self._img_size[0])
        iy = int((py - self._fit.top()) / self._fit.height() * self._img_size[1])
        return max(0, ix), max(0, iy)

    def mousePressEvent(self, ev):
        if ev.button() == Qt.MouseButton.RightButton:
            if self.cfg is not None:
                w, h = self._img_size
                self.cfg.roi_x, self.cfg.roi_y, self.cfg.roi_w, self.cfg.roi_h = 0, 0, w, h
                self.roi_changed.emit(0, 0, w, h)
                self.update()
            return
        if ev.button() == Qt.MouseButton.LeftButton and self._pixmap is not None:
            self._drag_start = self._to_img(ev.position().x(), ev.position().y())
            self._drag_cur = self._drag_start

    def mouseMoveEvent(self, ev):
        if self._drag_start and self._pixmap is not None:
            self._drag_cur = self._to_img(ev.position().x(), ev.position().y())
            self.update()

    def mouseReleaseEvent(self, ev):
        if self._drag_start and self._drag_cur and self.cfg is not None:
            x0, y0 = min(self._drag_start[0], self._drag_cur[0]), min(self._drag_start[1], self._drag_cur[1])
            x1, y1 = max(self._drag_start[0], self._drag_cur[0]), max(self._drag_start[1], self._drag_cur[1])
            w, h = x1 - x0, y1 - y0
            if w >= 8 and h >= 8:
                self.cfg.roi_x, self.cfg.roi_y, self.cfg.roi_w, self.cfg.roi_h = x0, y0, w, h
                self.roi_changed.emit(x0, y0, w, h)
        self._drag_start = self._drag_cur = None
        self.update()


# --------------------------------------------------------------------------
# 采集/统计/报警工作线程
# --------------------------------------------------------------------------
class CaptureWorker(QThread):
    frame_ready = Signal(object)    # ndarray(含蒙层,未画ROI)
    stats = Signal(dict)            # n/n_b/n_g/n_r/roi_pixels/over/fps
    state_changed = Signal(str)     # IDLE/PENDING/LATCHED
    alarm_fired = Signal(dict)      # n/frame/snapshot
    log = Signal(str)
    stopped = Signal()

    def __init__(self, cfg: AppConfig, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self._stop = False
        self._reset_req = False
        self._engine: AlarmEngine | None = None

    # ---- 界面线程调用 ----
    def stop(self):
        self._stop = True
        self.wait(3000)

    def request_reset(self):
        self._reset_req = True

    # ---- 主体 ----
    def run(self):
        cfg = self.cfg
        source = self._build_source(cfg)
        if source is None:
            self.stopped.emit()
            return
        self.log.emit(f"采集源已启动: {source.name}")
        try:
            source.open()
        except Exception as e:
            self.log.emit(f"错误: {e}")
            self.stopped.emit()
            return
        try:
            self._run_loop(source)
        finally:
            try:
                source.close()
            except Exception:
                pass
            if self._engine is not None:
                self._engine.close()
            self.stopped.emit()

    def _build_source(self, cfg: AppConfig):
        if cfg.source == "sim":
            s = SimSource()
            s.set_defect(cfg.sim_defect)
            s.set_defect_size(cfg.sim_defect_w, cfg.sim_defect_h)
            return s
        if cfg.source == "image":
            if not cfg.image_path or not os.path.exists(cfg.image_path):
                self.log.emit("错误: 图片源需要先选择一张图片")
                return None
            return ImageSource(cfg.image_path)
        if cfg.source == "hik":
            return HikSource()
        self.log.emit(f"错误: 未知信号源 {cfg.source}")
        return None

    def _run_loop(self, source):
        cfg = self.cfg
        self._engine = AlarmEngine(
            alarm_count=cfg.alarm_count, debounce_frames=cfg.debounce_frames,
            use_serial=cfg.use_serial, port=cfg.serial_port, baud=cfg.serial_baud,
            on_triggered=lambda n, fi: None,  # 事件改由线程内直接处理,避免回调线程
        )
        self.log.emit(f"报警输出: {self._engine.sink_note} (阈值 N≥{self._engine.alarm_count}, 连续 {self._engine.debounce} 帧)")
        # 相机参数:初始下发一次
        if source.is_camera:
            self._apply_camera_params(source, force=True)
        applied_exp = applied_gain = applied_fps = None

        os.makedirs(SNAPSHOT_DIR, exist_ok=True)
        last_disp = 0.0
        last_frame_time = time.time()
        fps_ema = 0.0

        while not self._stop:
            # 动态改相机参数
            if source.is_camera:
                self._apply_camera_params(source, force=False)
            # 模拟缺陷动态切换
            if isinstance(source, SimSource):
                if getattr(source, "defect", None) != cfg.sim_defect:
                    source.set_defect(cfg.sim_defect)
                if (source.defect_w, source.defect_h) != (cfg.sim_defect_w, cfg.sim_defect_h):
                    source.set_defect_size(cfg.sim_defect_w, cfg.sim_defect_h)
            # 复位请求
            if self._reset_req and self._engine is not None:
                self._reset_req = False
                self._engine.reset()
                self.state_changed.emit(self._engine.state)

            frame = source.read()
            if frame is None:
                if self._stop:
                    break
                time.sleep(0.01)
                continue
            snap = cfg_snapshot(cfg)   # 一致的参数快照
            res = count_highlight(
                frame, snap.roi,
                snap.window_b, snap.window_g, snap.window_r, blur_ksize=3)

            state = self._engine.feed(res.n)
            # 超阈值但未锁存 → PENDING 提示; feed 只在锁存后返回 LATCHED
            if state != AlarmEngine.LATCHED and self._engine.is_over(res.n):
                state = "PENDING"

            # 报警触发处理(锁存进入时执行一次:发状态、存截图)
            if state == AlarmEngine.LATCHED and self._engine.trigger_count > 0 and \
                    not getattr(self, "_snap_done", False):
                self._snap_done = True
                fname = f"alarm_{datetime.now():%Y%m%d_%H%M%S}_N{res.n}.png"
                fpath = os.path.join(SNAPSHOT_DIR, fname)
                try:
                    annotated = draw_roi(frame, res.roi)
                    cv2.imwrite(fpath, annotated)
                    self.log.emit(f"⚠ 报警触发 N={res.n} 已发送触发帧,截图: {fpath}")
                except Exception as e:
                    self.log.emit(f"截图保存失败: {e}")
                self.alarm_fired.emit({"n": res.n, "frame": self._engine.frame_idx, "snapshot": fpath})
            # 状态变化才发信号(去重,防每帧刷界面)
            if state != getattr(self, "_prev_state", None):
                self.state_changed.emit(state)
                self._prev_state = state
            if state != AlarmEngine.LATCHED:
                self._snap_done = False

            # 显示帧:原图或叠加蒙层,节流到 ~25fps 防界面卡顿
            now = time.time()
            if now - last_disp >= 0.04:
                last_disp = now
                disp = overlay_mask(frame, res) if snap.show_mask else frame.copy()
                self._last_frame = frame.copy()  # 供"保存当前帧"按钮使用
                self.frame_ready.emit(disp)
                now2 = time.time()
                inst = 1.0 / max(1e-6, now2 - last_frame_time)
                fps_ema = inst if fps_ema <= 0 else fps_ema * 0.9 + inst * 0.1
                last_frame_time = now2
                self.stats.emit({
                    "n": res.n, "n_b": res.n_b, "n_g": res.n_g, "n_r": res.n_r,
                    "roi_pixels": res.roi_pixels, "over": state != AlarmEngine.IDLE,
                    "state": state, "fps": round(fps_ema, 1),
                    "ratio": round(res.ratio * 100, 2),
                })
            # 目标帧率(模拟/图片源按需限速)
            target = max(1, snap.target_fps)
            if source.is_camera:
                target = 60  # 相机按自身帧率跑,不做额外限速
            time.sleep(max(0.0, 1.0 / target - 0.004))

    def _apply_camera_params(self, source, force: bool):
        cfg = self.cfg
        try:
            if force or getattr(self, "_a_exp", None) != cfg.exposure_us:
                source.set_exposure(cfg.exposure_us)
                self._a_exp = cfg.exposure_us
            if force or getattr(self, "_a_gain", None) != cfg.gain_db:
                source.set_gain(cfg.gain_db)
                self._a_gain = cfg.gain_db
            if force or getattr(self, "_a_fps", None) != cfg.target_fps:
                source.set_fps(cfg.target_fps)
                self._a_fps = cfg.target_fps
        except Exception as e:
            self.log.emit(f"相机参数设置失败: {e}")


# --------------------------------------------------------------------------
# 主窗口
# --------------------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self, cfg: AppConfig, store: ConfigStore):
        super().__init__()
        self.cfg = cfg
        self.store = store
        self.worker: CaptureWorker | None = None
        self.setWindowTitle("视觉检查系统 · 彩色高亮像素计数(三通道窗口)")
        self.resize(1560, 940)
        self._build_ui()
        self._sync_ui_from_cfg()
        self.log("系统就绪。左上角选择信号源后点『启动采集』;在画面上拖拽框选 ROI(右键=全幅)。")

    # ================= UI 构建 =================
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        # ---- 左:画面 ----
        self.video = VideoLabel()
        self.video.set_cfg(self.cfg)
        self.video.roi_changed.connect(self._on_roi_changed)
        root.addWidget(self.video, 1)

        # ---- 右:控制面板 ----
        panel = QWidget()
        panel.setFixedWidth(430)
        pv = QVBoxLayout(panel)
        pv.setContentsMargins(0, 0, 0, 0)

        # 报警状态区
        box = QGroupBox("报警状态")
        g = QGridLayout(box)
        self.lamp = QLabel("● 监视中")
        self.lamp.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lamp.setStyleSheet(self._lamp_style("#2e7d32"))
        self.lamp.setMinimumHeight(40)
        self.lbl_n = QLabel("N = 0")
        self.lbl_n.setStyleSheet("font-size:26px;font-weight:bold;")
        self.lbl_detail = QLabel("单通道达标  R:0  G:0  B:0   |   ROI占比 0.00%")
        g.addWidget(self.lamp, 0, 0, 1, 2)
        g.addWidget(self.lbl_n, 1, 0)
        g.addWidget(self.lbl_detail, 1, 1)
        self.btn_reset = QPushButton("复位报警")
        self.btn_reset.setEnabled(False)
        self.btn_reset.clicked.connect(self._on_reset)
        g.addWidget(self.btn_reset, 2, 0, 1, 2)
        pv.addWidget(box)

        # 信号源
        box = QGroupBox("信号源")
        g = QGridLayout(box)
        self.cmb_source = QComboBox()
        self.cmb_source.addItem("模拟器(演示)", "sim")
        self.cmb_source.addItem("图片(离线调参)", "image")
        self.cmb_source.addItem("海康相机 MVS", "hik")
        self.cmb_source.currentIndexChanged.connect(self._on_source_changed)
        self.btn_pick_image = QPushButton("选择图片…")
        self.btn_pick_image.clicked.connect(self._on_pick_image)
        self.ed_image = QLineEdit()
        self.ed_image.setReadOnly(True)
        self.ed_image.setPlaceholderText("(模拟器/相机时无需图片)")
        self.cmb_defect = QComboBox()
        for label, key in [("无缺陷", "none"), ("白色异物", "white"), ("红色异物", "red"),
                           ("绿色异物", "green"), ("蓝色异物", "blue")]:
            self.cmb_defect.addItem(label, key)
        self.cmb_defect.currentIndexChanged.connect(
            lambda _: setattr(self.cfg, "sim_defect", self.cmb_defect.currentData()))
        self.btn_start = QPushButton("▶ 启动采集")
        self.btn_start.setStyleSheet("background:#1565c0;color:white;font-size:15px;padding:6px;")
        self.btn_start.clicked.connect(self._on_start_stop)
        self.btn_saveframe = QPushButton("保存当前帧")
        self.btn_saveframe.clicked.connect(self._on_save_frame)
        g.addWidget(self.cmb_source, 0, 0, 1, 2)
        g.addWidget(self.ed_image, 1, 0)
        g.addWidget(self.btn_pick_image, 1, 1)
        g.addWidget(QLabel("模拟缺陷:"), 2, 0)
        g.addWidget(self.cmb_defect, 2, 1)
        g.addWidget(self.btn_start, 3, 0)
        g.addWidget(self.btn_saveframe, 3, 1)
        pv.addWidget(box)

        # ROI
        box = QGroupBox("ROI 区域(画面上拖拽框选;右键=全幅)")
        g = QGridLayout(box)
        self.sp_roi = {}
        for i, (k, label) in enumerate([("roi_x", "X"), ("roi_y", "Y"),
                                        ("roi_w", "宽"), ("roi_h", "高")]):
            sp = QSpinBox()
            sp.setRange(0, 100000)
            sp.valueChanged.connect(self._roi_spin_changed)
            g.addWidget(QLabel(label), 0, i * 2)
            g.addWidget(sp, 0, i * 2 + 1)
            self.sp_roi[k] = sp
        self.btn_roi_full = QPushButton("ROI=全幅")
        self.btn_roi_full.clicked.connect(lambda: self._set_roi(0, 0, 0, 0))
        g.addWidget(self.btn_roi_full, 1, 0, 1, 8)
        pv.addWidget(box)

        # 三通道窗口
        box = QGroupBox("彩色高亮判定:三通道窗口(像素须同时落入三个窗口)")
        g = QGridLayout(box)
        self.sp_win = {}
        ch_names = [("window_r", "R 红", 2), ("window_g", "G 绿", 1), ("window_b", "B 蓝", 0)]
        for row, (key, label, _idx) in enumerate(ch_names):
            g.addWidget(QLabel(label), row, 0)
            for col, part in enumerate(["lo", "hi"]):
                sp = QSpinBox()
                sp.setRange(0, 255)
                sp.valueChanged.connect(self._win_changed)
                g.addWidget(sp, row, 1 + col * 2)
                self.sp_win[f"{key}_{part}"] = sp
        g.addWidget(QLabel("快捷预设:"), 3, 0)
        self._preset_btns = {}
        for i, name in enumerate(PRESETS):
            b = QPushButton(name)
            b.setMaximumWidth(90)
            b.clicked.connect(lambda _=False, n=name: self._apply_preset(n))
            g.addWidget(b, 3, 1 + i)
            self._preset_btns[name] = b
        self.chk_mask = QCheckBox("红色蒙层显示命中像素")
        self.chk_mask.toggled.connect(lambda v: setattr(self.cfg, "show_mask", v))
        g.addWidget(self.chk_mask, 4, 0, 1, 8)
        pv.addWidget(box)

        # 报警阈值
        box = QGroupBox("报警判定")
        g = QGridLayout(box)
        self.sp_alarm = QSpinBox()
        self.sp_alarm.setRange(1, 100_000_000)
        self.sp_alarm.valueChanged.connect(lambda v: setattr(self.cfg, "alarm_count", v))
        self.sp_debounce = QSpinBox()
        self.sp_debounce.setRange(1, 100)
        self.sp_debounce.valueChanged.connect(lambda v: setattr(self.cfg, "debounce_frames", v))
        g.addWidget(QLabel("报警阈值 N≥:"), 0, 0)
        g.addWidget(self.sp_alarm, 0, 1)
        g.addWidget(QLabel("连续帧数:"), 0, 2)
        g.addWidget(self.sp_debounce, 0, 3)
        pv.addWidget(box)

        # 相机参数
        box = QGroupBox("相机参数(海康相机源生效)")
        g = QGridLayout(box)
        self.sp_exp = QSpinBox()
        self.sp_exp.setRange(5, 10_000_000)
        self.sp_exp.setSuffix(" µs")
        self.sp_exp.valueChanged.connect(lambda v: setattr(self.cfg, "exposure_us", v))
        self.sp_gain = QDoubleSpinBox()
        self.sp_gain.setRange(0, 24)
        self.sp_gain.setSingleStep(0.1)
        self.sp_gain.setSuffix(" dB")
        self.sp_gain.valueChanged.connect(lambda v: setattr(self.cfg, "gain_db", v))
        self.sp_fps = QSpinBox()
        self.sp_fps.setRange(1, 500)
        self.sp_fps.setSuffix(" fps")
        self.sp_fps.valueChanged.connect(lambda v: setattr(self.cfg, "target_fps", v))
        g.addWidget(QLabel("曝光:"), 0, 0)
        g.addWidget(self.sp_exp, 0, 1)
        g.addWidget(QLabel("增益:"), 1, 0)
        g.addWidget(self.sp_gain, 1, 1)
        g.addWidget(QLabel("目标帧率:"), 2, 0)
        g.addWidget(self.sp_fps, 2, 1)
        pv.addWidget(box)

        # 串口报警设备
        box = QGroupBox("外部报警设备(串口)")
        g = QGridLayout(box)
        self.cmb_port = QComboBox()
        self.cmb_port.setEditable(True)
        self.cmb_port.currentTextChanged.connect(lambda t: setattr(self.cfg, "serial_port", t.strip()))
        self.btn_refresh_port = QPushButton("刷新")
        self.btn_refresh_port.clicked.connect(self._refresh_ports)
        self.cmb_baud = QComboBox()
        for b in (9600, 19200, 38400, 115200):
            self.cmb_baud.addItem(str(b))
        self.cmb_baud.currentTextChanged.connect(lambda t: setattr(self.cfg, "serial_baud", int(t)))
        self.chk_serial = QCheckBox("启用串口报警(不勾=仅日志)")
        self.chk_serial.toggled.connect(lambda v: setattr(self.cfg, "use_serial", v))
        g.addWidget(QLabel("端口:"), 0, 0)
        g.addWidget(self.cmb_port, 0, 1)
        g.addWidget(self.btn_refresh_port, 0, 2)
        g.addWidget(QLabel("波特率:"), 1, 0)
        g.addWidget(self.cmb_baud, 1, 1)
        g.addWidget(self.chk_serial, 2, 0, 1, 3)
        pv.addWidget(box)

        # 日志
        box = QGroupBox("日志")
        vb = QVBoxLayout(box)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(500)
        vb.addWidget(self.log_view)
        pv.addWidget(box, 1)

        # 保存配置
        self.btn_save_cfg = QPushButton("保存配置到 config.json")
        self.btn_save_cfg.clicked.connect(lambda: (self.store.save(self.cfg), self.log("配置已保存")))
        pv.addWidget(self.btn_save_cfg)

        scroll = QScrollArea()
        scroll.setWidget(panel)
        scroll.setWidgetResizable(True)
        root.addWidget(scroll)

    # ================= 样式与同步 =================
    @staticmethod
    def _lamp_style(color: str) -> str:
        return (f"background:{color};color:white;font-size:16px;font-weight:bold;"
                "border-radius:6px;padding:4px;")

    def _sync_ui_from_cfg(self):
        c = self.cfg
        self.cmb_source.setCurrentIndex(self.cmb_source.findData(c.source))
        self.ed_image.setText(c.image_path)
        self.cmb_defect.setCurrentIndex(self.cmb_defect.findData(c.sim_defect))
        for k, sp in self.sp_roi.items():
            sp.blockSignals(True)
            sp.setValue(getattr(c, k))
            sp.blockSignals(False)
        for key, part in [("window_r", "lo"), ("window_r", "hi"), ("window_g", "lo"),
                          ("window_g", "hi"), ("window_b", "lo"), ("window_b", "hi")]:
            sp = self.sp_win[f"{key}_{part}"]
            sp.blockSignals(True)
            sp.setValue(getattr(getattr(c, key), part))
            sp.blockSignals(False)
        self.sp_alarm.blockSignals(True); self.sp_alarm.setValue(c.alarm_count); self.sp_alarm.blockSignals(False)
        self.sp_debounce.blockSignals(True); self.sp_debounce.setValue(c.debounce_frames); self.sp_debounce.blockSignals(False)
        self.sp_exp.blockSignals(True); self.sp_exp.setValue(c.exposure_us); self.sp_exp.blockSignals(False)
        self.sp_gain.blockSignals(True); self.sp_gain.setValue(int(c.gain_db * 10) / 10); self.sp_gain.blockSignals(False)
        self.sp_fps.blockSignals(True); self.sp_fps.setValue(c.target_fps); self.sp_fps.blockSignals(False)
        self.chk_serial.setChecked(c.use_serial)
        self.chk_mask.setChecked(c.show_mask)
        self.cmb_baud.setCurrentText(str(c.serial_baud))
        self._refresh_ports(preserve=c.serial_port)

    def _refresh_ports(self, preserve: str | None = None):
        cur = preserve or self.cfg.serial_port
        self.cmb_port.blockSignals(True)
        self.cmb_port.clear()
        ports = list_com_ports() or ["COM1"]
        self.cmb_port.addItems(ports)
        if cur in ports:
            self.cmb_port.setCurrentText(cur)
        elif cur:
            self.cmb_port.setCurrentText(cur)
        self.cmb_port.blockSignals(False)
        self.cfg.serial_port = self.cmb_port.currentText().strip()

    # ================= 事件 =================
    def log(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_view.appendPlainText(f"[{ts}] {msg}")

    def _on_roi_changed(self, x, y, w, h):
        for k, sp in self.sp_roi.items():
            sp.blockSignals(True)
            sp.setValue({"roi_x": x, "roi_y": y, "roi_w": w, "roi_h": h}[k])
            sp.blockSignals(False)
        self.log(f"ROI 已更新: x={x} y={y} w={w} h={h}")

    def _roi_spin_changed(self, _v):
        for k, sp in self.sp_roi.items():
            setattr(self.cfg, k, sp.value())

    def _set_roi(self, _x, _y, _w, _h):
        # 全幅:依赖下一帧的图幅,先置 0 由 worker 端 clamp 兜底;此处直接用当前图幅
        w, h = self.video._img_size
        self.cfg.roi_x, self.cfg.roi_y, self.cfg.roi_w, self.cfg.roi_h = 0, 0, w, h
        self._on_roi_changed(0, 0, w, h)
        self.video.update()

    def _win_changed(self, _v):
        for key, part in [("window_r", "lo"), ("window_r", "hi"), ("window_g", "lo"),
                          ("window_g", "hi"), ("window_b", "lo"), ("window_b", "hi")]:
            obj = getattr(self.cfg, key)
            setattr(obj, part, self.sp_win[f"{key}_{part}"].value())

    def _apply_preset(self, name: str):
        (blo, bhi), (glo, ghi), (rlo, rhi) = PRESETS[name]
        for key, (lo, hi) in [("window_r", (rlo, rhi)), ("window_g", (glo, ghi)),
                              ("window_b", (blo, bhi))]:
            obj = getattr(self.cfg, key)
            obj.lo, obj.hi = lo, hi
            for part, v in [("lo", lo), ("hi", hi)]:
                sp = self.sp_win[f"{key}_{part}"]
                sp.blockSignals(True)
                sp.setValue(v)
                sp.blockSignals(False)
        self.log(f"已应用预设『{name}』: R[{rlo},{rhi}] G[{glo},{ghi}] B[{blo},{bhi}]")

    def _on_source_changed(self):
        self.cfg.source = self.cmb_source.currentData()
        is_sim = self.cfg.source == "sim"
        self.cmb_defect.setEnabled(is_sim)
        self.log(f"信号源: {self.cmb_source.currentText()}")

    def _on_pick_image(self):
        p, _ = QFileDialog.getOpenFileName(self, "选择产品图片", "", "图片 (*.png *.jpg *.jpeg *.bmp)")
        if p:
            self.cfg.image_path = p
            self.ed_image.setText(p)
            self.log(f"图片: {p}")
            if self.cfg.source != "image":
                self.cmb_source.setCurrentIndex(self.cmb_source.findData("image"))

    def _on_save_frame(self):
        if self.worker is None:
            self.log("未在采集,无法保存")
            return
        # 通过 worker 属性取最近帧有跨线程风险,这里改由 worker 每帧缓存
        if getattr(self.worker, "_last_frame", None) is None:
            self.log("尚无画面")
            return
        os.makedirs(SNAPSHOT_DIR, exist_ok=True)
        f = os.path.join(SNAPSHOT_DIR, f"manual_{datetime.now():%Y%m%d_%H%M%S}.png")
        cv2.imwrite(f, self.worker._last_frame)
        self.log(f"已保存: {f}")

    # ================= 启停 =================
    def _on_start_stop(self):
        if self.worker is not None and self.worker.isRunning():
            self._stop_worker()
            return
        # 校验:图片源必须已选图;hik 源给出联调提示
        if self.cfg.source == "hik":
            self.log("提示: 海康源需要本机已装 MVS SDK 且相机出图;若报错请按 README 联调步骤检查。")
        self.worker = CaptureWorker(self.cfg)
        self.worker.frame_ready.connect(self.video.set_frame)
        self.worker.stats.connect(self._on_stats)
        self.worker.state_changed.connect(self._on_state)
        self.worker.alarm_fired.connect(lambda d: None)  # 已由日志体现
        self.worker.log.connect(self.log)
        self.worker.stopped.connect(self._on_worker_stopped)
        self.worker.start()
        self.btn_start.setText("■ 停止采集")
        self._set_lamp("#1565c0", "连接中…")
        self.log("启动采集线程…")

    def _stop_worker(self):
        if self.worker is not None:
            self.worker.stop()
        self.btn_start.setText("▶ 启动采集")
        self._set_lamp("#607d8b", "已停止")
        self.log("采集已停止")

    def _on_worker_stopped(self):
        self.btn_start.setText("▶ 启动采集")
        if self.worker:
            self.worker.deleteLater()
            self.worker = None
        self.btn_reset.setEnabled(False)
        self._set_lamp("#607d8b", "已停止")

    def _on_reset(self):
        if self.worker is not None and self.worker.isRunning():
            self.worker.request_reset()
            self.log("已请求复位报警")
        else:
            self._set_lamp("#2e7d32", "● 监视中")

    # ================= 状态回显 =================
    def _set_lamp(self, color: str, text: str):
        self.lamp.setText(text)
        self.lamp.setStyleSheet(self._lamp_style(color))

    def _on_state(self, state: str):
        if state == AlarmEngine.LATCHED:
            self._set_lamp("#c62828", "● 报警锁存")
            self.btn_reset.setEnabled(True)
        elif state == "PENDING":
            self._set_lamp("#ef6c00", "! 超阈值累计中…")
        else:
            self._set_lamp("#2e7d32", "● 监视中")
            self.btn_reset.setEnabled(False)

    def _on_stats(self, s: dict):
        self.lbl_n.setText(f"N = {s['n']:,}   (阈值 {self.cfg.alarm_count:,})")
        self.lbl_detail.setText(
            f"单通道: R {s['n_r']:,} | G {s['n_g']:,} | B {s['n_b']:,}"
            f"   | ROI {s['roi_pixels']:,}px | 占比 {s['ratio']}% | {s['fps']} fps")

    # ================= 关闭 =================
    def closeEvent(self, ev):
        if self.worker is not None and self.worker.isRunning():
            self.worker.stop()
        self.store.save(self.cfg)
        super().closeEvent(ev)


def run(cfg: AppConfig, store: ConfigStore):
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow(cfg, store)
    win.show()
    return app.exec()
