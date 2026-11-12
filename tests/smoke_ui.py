"""GUI 离屏冒烟测试(不开真实窗口):验证 主窗口构建 → 启动采集线程 →
模拟缺陷出现 → 报警锁存(灯红/复位可用) → 复位恢复 → 停止。

运行: .venv/Scripts/python.exe tests/smoke_ui.py
"""
from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PySide6.QtWidgets import QApplication  # noqa: E402

from app.config import AppConfig, ConfigStore  # noqa: E402
from app.ui_main import MainWindow  # noqa: E402

ok = []


def pump(app: QApplication, secs: float):
    t0 = time.time()
    while time.time() - t0 < secs:
        app.processEvents()
        time.sleep(0.01)


def main() -> int:
    app = QApplication(sys.argv)
    cfg = AppConfig()
    cfg.source = "sim"
    store = ConfigStore(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                     "config_smoke.json"))
    win = MainWindow(cfg, store)
    win.show()

    # 启动模拟采集
    win._on_start_stop()
    assert win.worker is not None and win.worker.isRunning(), "worker 未启动"
    pump(app, 2.0)

    r1 = win.worker._last_frame is not None and "N =" in win.lbl_n.text()
    ok.append(("有画面且 N 标签更新", r1))
    print(f"[{'PASS' if r1 else 'FAIL'}] 有画面且 N 标签更新  (N标签='{win.lbl_n.text()}')")

    # 让白色异物出现 → 应触发报警锁存
    cfg.sim_defect = "white"
    pump(app, 1.5)
    lamp_text = win.lamp.text()
    r2 = "报警" in lamp_text and win.btn_reset.isEnabled()
    ok.append(("报警锁存:红灯+复位可用", r2))
    print(f"[{'PASS' if r2 else 'FAIL'}] 报警锁存 (灯='{lamp_text}', 复位可用={win.btn_reset.isEnabled()})")

    # 复位 → 恢复监视(先撤走缺陷,否则 3 帧内会正确地再次触发)
    cfg.sim_defect = "none"
    pump(app, 0.6)          # 让缺陷帧流出,状态回到累计清零路径
    win._on_reset()
    pump(app, 0.6)
    r3 = "监视中" in win.lamp.text() and not win.btn_reset.isEnabled()
    ok.append(("复位后恢复监视", r3))
    print(f"[{'PASS' if r3 else 'FAIL'}] 复位恢复 (灯='{win.lamp.text()}')")

    # 停采集
    win._on_start_stop()
    pump(app, 0.5)
    r4 = win.worker is None or not win.worker.isRunning()
    ok.append(("停止采集干净退出", r4))
    print(f"[{'PASS' if r4 else 'FAIL'}] 停止采集")

    try:
        os.remove(store.path)
    except OSError:
        pass
    passed = sum(1 for _, v in ok if v)
    print(f"\nGUI 冒烟: {passed}/{len(ok)} 通过")
    return 0 if passed == len(ok) else 1


if __name__ == "__main__":
    sys.exit(main())
