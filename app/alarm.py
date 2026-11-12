"""报警引擎:去抖 → 锁存 → 串口触发,以及串口/日志两种输出。

逻辑(与用户确认):
  连续 debounce_frames 帧 N >= alarm_count → 触发报警一次(发 07 00 00 A7),
  之后【锁存】:即便 N 回落也不再重复发,直到外部调用 reset()。
设计要点:
  - 纯逻辑无 GUI 依赖,便于自动化自检;
  - 串口打不开 / 未接报警盒时自动降级为日志输出并提示,不让整条检测线崩掉;
  - 发送与打开都加锁,线程安全。
"""
from __future__ import annotations

import threading
import time
from typing import Callable, Optional

import serial
from serial.tools import list_ports

from .config import ALARM_FRAME


class AlarmSink:
    """报警输出接口。"""

    def send(self, payload: bytes) -> bool:
        raise NotImplementedError

    def close(self) -> None:
        pass


class SerialSink(AlarmSink):
    """真实串口报警设备。打开失败时 last_error 有原因。"""

    def __init__(self, port: str, baud: int) -> None:
        self.port = port
        self.baud = baud
        self._ser: Optional[serial.Serial] = None
        self.last_error: str = ""

    def open(self) -> bool:
        try:
            self._ser = serial.Serial(self.port, self.baud, timeout=1)
            return True
        except Exception as e:
            self.last_error = f"串口 {self.port} 打开失败: {e}"
            return False

    def send(self, payload: bytes) -> bool:
        if self._ser is None:
            return False
        try:
            return self._ser.write(payload) == len(payload)
        except Exception as e:
            self.last_error = f"串口写入失败: {e}"
            return False

    def close(self) -> None:
        if self._ser is not None:
            try:
                self._ser.close()
            except Exception:
                pass
            self._ser = None


class LogSink(AlarmSink):
    """不接设备时的输出:打到 print(自检/演示用)。"""

    def __init__(self, label: str = "LOG") -> None:
        self.label = label

    def send(self, payload: bytes) -> bool:
        print(f"[alarm:{self.label}] 触发帧: {' '.join(f'{b:02X}' for b in payload)}")
        return True


def list_com_ports() -> list[str]:
    return [p.device for p in list_ports.comports()]


class AlarmEngine:
    """一帧一喂的状态机。"""

    IDLE, PENDING, LATCHED = "IDLE", "PENDING", "LATCHED"

    def __init__(self,
                 alarm_count: int = 800,
                 debounce_frames: int = 3,
                 use_serial: bool = True,
                 port: str = "COM2",
                 baud: int = 9600,
                 on_triggered: Optional[Callable[[int, int], None]] = None,
                 on_state: Optional[Callable[[str], None]] = None) -> None:
        self.alarm_count = int(alarm_count)
        self.debounce = max(1, int(debounce_frames))
        self._lock = threading.Lock()
        self.state = self.IDLE
        self._streak = 0
        self.trigger_count = 0
        self.frame_idx = 0
        self.sink: AlarmSink = LogSink("dev")
        self.sink_note = "仅日志(未用串口)"
        self.on_triggered = on_triggered
        self.on_state = on_state
        if use_serial:
            self._open_serial(port, baud)

    def _open_serial(self, port: str, baud: int) -> None:
        s = SerialSink(port, baud)
        if s.open():
            self.sink = s
            self.sink_note = f"串口 {port} @ {baud}"
        else:
            self.sink = LogSink("fallback")
            self.sink_note = f"串口不可用,已降级为日志: {s.last_error}"

    # ---------------- 参数运行时可改(线程安全) ----------------
    def update_params(self, alarm_count: int, debounce_frames: int) -> None:
        with self._lock:
            self.alarm_count = int(alarm_count)
            self.debounce = max(1, int(debounce_frames))

    # ---------------- 每帧喂入 ----------------
    def feed(self, n: int) -> str:
        """喂一帧的目标像素计数 N,返回当前状态;超阈值推进状态机,达标即触发。"""
        with self._lock:
            self.frame_idx += 1
            if self.state == self.LATCHED:
                return self.state
            if n >= self.alarm_count:
                self._streak += 1
                if self._streak >= self.debounce:
                    self.state = self.LATCHED
                    self.trigger_count += 1
                    self._fire(n)
            else:
                self._streak = 0
            return self.state

    def _fire(self, n: int) -> None:
        ok = False
        try:
            ok = self.sink.send(ALARM_FRAME)
        except Exception as e:
            print(f"[alarm] 发送异常: {e}")
        self._emit_state()
        if self.on_triggered:
            try:
                self.on_triggered(n, self.frame_idx)
            except Exception:
                pass
        if not ok:
            print(f"[alarm] 帧发送失败/被降级({self.sink_note})")
        else:
            print(f"[alarm] 已发送触发帧 07 00 00 A7 (N={n}, sink={self.sink_note})")

    def is_over(self, n: int) -> bool:
        """当前帧是否超阈值(纯查询,界面指示灯用,不推进状态)。"""
        return n >= self.alarm_count

    def reset(self) -> None:
        with self._lock:
            self.state = self.IDLE
            self._streak = 0
        self._emit_state()
        print("[alarm] 已复位,解除锁存")

    def close(self) -> None:
        try:
            self.sink.close()
        except Exception:
            pass

    def _emit_state(self) -> None:
        if self.on_state:
            try:
                self.on_state(self.state)
            except Exception:
                pass

    # 供 UI 显示的描述
    @property
    def state_text(self) -> str:
        return {"IDLE": "监视中", "PENDING": "累计超阈…", "LATCHED": "报警锁存"}.get(self.state, self.state)
