"""视觉检查系统入口。

用法:
  python main.py           启动 GUI(默认信号源=模拟器,可离线演示全流程)
  python main.py --selfcheck   无界面自检:验证 计数→去抖→报警→锁存→复位 全链路
"""
from __future__ import annotations

import argparse
import os
import sys

from app.config import AppConfig, ConfigStore, ChannelWindow, PRESETS, ALARM_FRAME
from app.processor import count_highlight

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(PROJECT_DIR, "config.json")


def _check(name: str, ok: bool, extra: str = "") -> bool:
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {extra}")
    return ok


def selfcheck() -> int:
    """无 GUI 全链路自检(模拟源 + 日志输出),返回进程退出码。"""
    import numpy as np

    from app.alarm import AlarmEngine, LogSink
    from app.sources import SimSource

    results: list[bool] = []
    win_white = [ChannelWindow(180, 255)] * 3
    full = (0, 0, 1280, 1024)

    # ---- 1. 无缺陷:不应报警 ----
    src = SimSource()
    src.open()
    eng = AlarmEngine(use_serial=False)          # 日志输出
    eng.sink = LogSink("selfcheck")
    src.set_defect("none")
    ns = []
    for _ in range(10):
        frame = src.read()
        r = count_highlight(frame, full, *win_white)
        ns.append(r.n)
        eng.feed(r.n)
    ok1 = max(ns) == 0 and eng.state == eng.IDLE
    results.append(_check("无缺陷 10 帧 N=0 且不报警", ok1, f"(N={max(ns)}, state={eng.state})"))

    # ---- 2. 白色异物:连续3帧超阈值 → 报警锁存 ----
    eng2 = AlarmEngine(alarm_count=800, debounce_frames=3, use_serial=False)
    sends: list[bytes] = []

    class _Cnt(LogSink):
        def send(self, payload: bytes) -> bool:
            sends.append(payload)
            return super().send(payload)

    eng2.sink = _Cnt("selfcheck")
    src.set_defect("white")
    src.set_defect_size(60, 40)
    states = []
    for _ in range(10):
        frame = src.read()
        r = count_highlight(frame, full, *win_white)
        states.append(eng2.feed(r.n))
    n_white = count_highlight(src.read(), full, *win_white).n
    ok2 = eng2.state == eng2.LATCHED and len(sends) == 1 and sends[0] == ALARM_FRAME
    results.append(_check(f"白色异物(60x40)触发报警,只发一次 {' '.join(f'{b:02X}' for b in ALARM_FRAME)}",
                          ok2, f"(N={n_white}, 发送次数={len(sends)}, state={eng2.state})"))

    # ---- 3. 锁存期间缺陷移除,N 回落也不解除 ----
    src.set_defect("none")
    for _ in range(5):
        frame = src.read()
        r = count_highlight(frame, full, *win_white)
        eng2.feed(r.n)
    ok3 = eng2.state == eng2.LATCHED and len(sends) == 1
    results.append(_check("锁存保持:异物消失后仍锁存且不重复发送", ok3))

    # ---- 4. 复位 → 恢复监视 ----
    eng2.reset()
    for _ in range(3):
        frame = src.read()
        r = count_highlight(frame, full, *win_white)
        eng2.feed(r.n)
    ok4 = eng2.state == eng2.IDLE
    results.append(_check("复位后恢复 IDLE,可再次累计", ok4))

    # ---- 5. 三通道窗口排除干扰色:红色异物在"白色窗口"下不计入 ----
    src.set_defect("red")
    src.set_defect_size(60, 40)
    rs = []
    for _ in range(3):
        frame = src.read()
        r = count_highlight(frame, full, *win_white)
        rs.append(r.n)
    red_n = max(rs)
    ok5 = red_n == 0
    results.append(_check("纯红异物 60x40 在白色窗口[R,G,B∈180..255]下 N=0(不误报)",
                          ok5, f"(N={red_n})"))

    # ---- 6. 换红色窗口预设 → 红色异物被精确命中 ----
    win_red = [ChannelWindow(0, 120), ChannelWindow(0, 120), ChannelWindow(180, 255)]
    rr = count_highlight(src.read(), full, *win_red)
    ok6 = rr.n >= 60 * 40 * 0.7
    results.append(_check("改用红色窗口[R∈180..255, G,B∈0..120]后 N 显著>0",
                          ok6, f"(N={rr.n})"))

    # ---- 7. 局部 ROI 统计(只框缺陷所在的一半) ----
    src.set_defect("white")
    src.defect_xy = (200, 400)   # 固定落在左半，避免随机位置跨中线导致两侧都有值
    roi_left = (0, 0, 640, 1024)
    roi_right = (640, 0, 640, 1024)
    nl, nr_ = 0, 0
    for _ in range(3):
        fl = src.read()
        rl = count_highlight(fl, roi_left, *win_white)
        rr_ = count_highlight(fl, roi_right, *win_white)
        nl, nr_ = max(nl, rl.n), max(nr_, rr_.n)
    ok7 = (nl > 0) != (nr_ > 0)  # 缺陷只落在一侧 → 两侧结果相反
    results.append(_check("ROI 局部统计生效(缺陷只被框住的一侧计入)",
                          ok7, f"(左半N={nl}, 右半N={nr_})"))

    src.close()
    passed = sum(results)
    print(f"\n自检结果: {passed}/{len(results)} 通过")
    return 0 if passed == len(results) else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="视觉检查系统(彩色高亮像素计数 + 串口报警)")
    ap.add_argument("--selfcheck", action="store_true", help="无界面自检全链路")
    ap.add_argument("--config", default=CONFIG_PATH, help="config.json 路径")
    args = ap.parse_args()

    if args.selfcheck:
        return selfcheck()

    from app.ui_main import run

    store = ConfigStore(args.config)
    cfg = store.load()
    # 首次运行把 ROI 默认成全幅(与实际图幅一致由 worker 端 clamp 保证)
    return run(cfg, store)


if __name__ == "__main__":
    sys.exit(main())
