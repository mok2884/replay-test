#!/usr/bin/env python3
"""リプレイ用テストソフト。

確認すること
  - そのPCで起動できるか
  - 入出力デバイス（USBインターフェース、Danteなど）を認識できるか
  - Enterを押してから音が出るまでの速さ
"""
import datetime
import os
import sys
import time
import traceback
from pathlib import Path

# sounddevice の Windows 版は、この環境変数があると ASIO 対応版の PortAudio を読み込む。
os.environ.setdefault("SD_ENABLE_ASIO", "1")

import numpy as np
import sounddevice as sd
import tkinter as tk
from tkinter import font as tkfont, ttk

APP_NAME = "リプレイ テスト"
VERSION = "0.1"
APP_DIR = Path(sys.executable if getattr(sys, "frozen", False) else __file__).resolve().parent
NONE_LABEL = "（使わない）"
TONE_SECONDS = 1.0
TONE_HZ = 1000
TONE_DB = -20
API_ORDER = ("ASIO", "WASAPI", "WDM-KS", "DirectSound", "MME")

BG = "#14171c"
PANEL = "#1f242c"
TEXT = "#eef1f5"
MUTED = "#98a2b0"
RED = "#e5343f"
GREEN = "#2fbf71"
AMBER = "#e6a817"


def log_exception(*_args):
    try:
        with open(APP_DIR / "error.log", "a", encoding="utf-8") as f:
            f.write(f"--- {datetime.datetime.now():%Y-%m-%d %H:%M:%S}\n{traceback.format_exc()}\n")
    except OSError:
        pass


def make_tone(rate):
    n = int(rate * TONE_SECONDS)
    t = np.arange(n) / rate
    tone = 10 ** (TONE_DB / 20) * np.sin(2 * np.pi * TONE_HZ * t)
    fade = min(int(rate * 0.005), n // 2)
    if fade:  # 終わりのクリック音だけを防ぐ。頭は即座に音を出す
        tone[-fade:] *= np.linspace(1, 0, fade)
    return tone.astype(np.float32)


class Engine:
    def __init__(self):
        self.streams = []
        self.in_channels = ()
        self.out_channels = ()
        self.peaks = [0.0, 0.0]
        self.out_peak = 0.0
        self.tone = None
        self.pos = 0
        self.playing = False
        self.press_time = None
        self.key_to_buffer_ms = []
        self.output_latency_ms = None
        self.problems = 0
        self.rate = 48000

    def _on_input(self, indata, status):
        if status.input_overflow or status.input_underflow:
            self.problems += 1
        for i, ch in enumerate(self.in_channels):
            peak = float(np.abs(indata[:, ch]).max()) if len(indata) else 0.0
            if peak > self.peaks[i]:
                self.peaks[i] = peak

    def _on_output(self, outdata, frames, status):
        outdata.fill(0)
        if status.output_underflow or status.output_overflow:
            self.problems += 1
        if not self.playing or self.tone is None:
            return
        press = self.press_time
        if press is not None:
            self.press_time = None
            self.key_to_buffer_ms.append((time.perf_counter() - press) * 1000)
        block = self.tone[self.pos:self.pos + frames]
        for ch in self.out_channels:
            outdata[:len(block), ch] = block
        self.out_peak = max(self.out_peak, float(np.abs(block).max()) if len(block) else 0.0)
        self.pos += len(block)
        if self.pos >= len(self.tone):
            self.playing = False

    def _duplex(self, indata, outdata, frames, _t, status):
        try:
            self._on_input(indata, status)
            self._on_output(outdata, frames, status)
        except Exception:
            self.problems += 1

    def _input_only(self, indata, _frames, _t, status):
        try:
            self._on_input(indata, status)
        except Exception:
            self.problems += 1

    def _output_only(self, outdata, frames, _t, status):
        try:
            self._on_output(outdata, frames, status)
        except Exception:
            self.problems += 1

    def open(self, in_dev, out_dev, in_channels, out_channels, rate, latency):
        self.close()
        self.rate = rate
        self.tone = make_tone(rate)
        self.in_channels = tuple(in_channels) if in_dev is not None else ()
        self.out_channels = tuple(out_channels)
        self.key_to_buffer_ms = []
        self.problems = 0
        n_out = max(self.out_channels) + 1
        common = dict(samplerate=rate, dtype="float32", latency=latency)
        try:
            if in_dev is None:
                self.streams = [sd.OutputStream(device=out_dev, channels=n_out,
                                                callback=self._output_only, **common)]
            elif in_dev == out_dev:
                self.streams = [sd.Stream(device=(in_dev, out_dev), channels=(max(self.in_channels) + 1, n_out),
                                          callback=self._duplex, **common)]
            else:
                self.streams = [
                    sd.InputStream(device=in_dev, channels=max(self.in_channels) + 1,
                                   callback=self._input_only, **common),
                    sd.OutputStream(device=out_dev, channels=n_out, callback=self._output_only, **common),
                ]
            for s in self.streams:
                s.start()
            latency_value = self.streams[-1].latency
            output = latency_value[1] if isinstance(latency_value, (tuple, list)) else latency_value
            self.output_latency_ms = output * 1000
        except Exception:
            self.close()
            raise

    def close(self):
        for s in self.streams:
            try:
                s.stop()
                s.close()
            except Exception:
                pass
        self.streams = []
        self.playing = False

    def fire(self):
        if self.playing or not self.streams:
            return False
        self.pos = 0
        self.press_time = time.perf_counter()
        self.playing = True
        return True

    def stop(self):
        self.playing = False
        self.press_time = None


def list_devices():
    """[(番号, 表示名, 入力ch数, 出力ch数, API名)]。ASIOとWASAPIを先頭に並べる。"""
    result = []
    for i, d in enumerate(sd.query_devices()):
        api = sd.query_hostapis(d["hostapi"])["name"].replace("Windows ", "")
        result.append((i, f"[{api}] {d['name']}", d["max_input_channels"], d["max_output_channels"], api))

    def order(item):
        api = item[4]
        return (API_ORDER.index(api) if api in API_ORDER else len(API_ORDER), item[0])

    return sorted(result, key=order)


def rescan_devices():
    try:
        sd._terminate()
        sd._initialize()
    except Exception:
        log_exception()


class Meter(tk.Canvas):
    def __init__(self, parent, height=20):
        super().__init__(parent, height=height, bg="#0c0e11", highlightthickness=0)
        self.value = 0.0

    def set(self, peak):
        db = 20 * np.log10(peak) if peak > 1e-6 else -120.0
        target = min(1.0, max(0.0, (db + 60) / 60))
        self.value = target if target > self.value else max(target, self.value - 0.03)
        w, h = self.winfo_width(), self.winfo_height()
        self.delete("all")
        x = int(w * self.value)
        for start, end, color in ((0.0, 0.70, GREEN), (0.70, 0.90, AMBER), (0.90, 1.0, RED)):
            a, b = int(w * start), min(x, int(w * end))
            if b > a:
                self.create_rectangle(a, 0, b, h, fill=color, width=0)
        for mark in (-40, -20, -6):
            mx = int(w * (mark + 60) / 60)
            self.create_line(mx, 0, mx, h, fill="#000000")


class App:
    def __init__(self, root):
        self.root = root
        self.engine = Engine()
        self.devices = []
        self.enter_down = False
        self.was_playing = False
        self.reopening = False
        self.has_focus = True
        self.error = ""

        families = set(tkfont.families())
        family = next((f for f in ("Yu Gothic UI", "Meiryo UI", "Noto Sans CJK JP") if f in families), "TkDefaultFont")
        self.f_big = (family, 24, "bold")
        self.f_mid = (family, 14, "bold")
        self.f_body = (family, 11)
        self.f_small = (family, 10)

        root.title(f"{APP_NAME} {VERSION}")
        root.configure(bg=BG)
        root.geometry("760x560")
        root.minsize(680, 520)
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.build()
        root.bind("<KeyPress-Return>", self.on_enter)
        root.bind("<KeyRelease-Return>", self.on_enter_release)
        root.bind("<KeyPress-Escape>", lambda _e: self.engine.stop())
        self.load_devices()
        self.reopen()
        root.focus_force()
        self.tick()

    def build(self):
        style = ttk.Style(self.root)
        style.theme_use("clam")

        self.focus_bar = tk.Label(self.root, text="", font=self.f_mid, fg="#ffffff", bg=BG, pady=6)
        self.focus_bar.pack(fill="x")

        form = tk.Frame(self.root, bg=PANEL, padx=14, pady=10)
        form.pack(fill="x", padx=12)
        form.columnconfigure(1, weight=1)
        self.v_in = tk.StringVar()
        self.v_out = tk.StringVar()
        self.v_in_a = tk.StringVar(value="1")
        self.v_in_b = tk.StringVar(value="2")
        self.v_out_l = tk.StringVar(value="1")
        self.v_out_r = tk.StringVar(value="2")
        self.v_rate = tk.StringVar(value="48000")
        self.v_latency = tk.StringVar(value="low")

        self.c_in = self.combo(form, 0, "入力デバイス", self.v_in, width=52)
        self.c_out = self.combo(form, 1, "出力デバイス", self.v_out, width=52)
        rows = tk.Frame(form, bg=PANEL)
        rows.grid(row=2, column=0, columnspan=2, sticky="w", pady=(6, 0))
        self.c_in_a = self.small_combo(rows, "入力ch", self.v_in_a)
        self.c_in_b = self.small_combo(rows, "/", self.v_in_b)
        self.c_out_l = self.small_combo(rows, "   出力ch", self.v_out_l)
        self.c_out_r = self.small_combo(rows, "/", self.v_out_r)
        self.c_rate = self.small_combo(rows, "   サンプルレート", self.v_rate, ["44100", "48000", "96000"], width=7)
        self.c_latency = self.small_combo(rows, "   遅延", self.v_latency, ["low", "high"], width=5)

        fire = tk.Frame(self.root, bg=PANEL, padx=14, pady=14)
        fire.pack(fill="x", padx=12, pady=8)
        self.fire_label = tk.Label(fire, text="Enter で音を出す", font=self.f_big, fg=TEXT, bg=PANEL)
        self.fire_label.pack()
        self.latency_label = tk.Label(fire, text="", font=self.f_mid, fg=TEXT, bg=PANEL)
        self.latency_label.pack(pady=(4, 0))
        self.detail_label = tk.Label(fire, text="Esc で止める", font=self.f_small, fg=MUTED, bg=PANEL, justify="center")
        self.detail_label.pack(pady=(2, 0))

        meters = tk.Frame(self.root, bg=PANEL, padx=14, pady=10)
        meters.pack(fill="x", padx=12)
        meters.columnconfigure(1, weight=1)
        self.meters = []
        for i, text in enumerate(("入力 A", "入力 B", "出力")):
            tk.Label(meters, text=text, font=self.f_small, fg=MUTED, bg=PANEL).grid(row=i, column=0, sticky="w", padx=(0, 10), pady=3)
            m = Meter(meters)
            m.grid(row=i, column=1, sticky="ew", pady=3)
            self.meters.append(m)

        bottom = tk.Frame(self.root, bg=BG)
        bottom.pack(fill="x", padx=12, pady=8)
        self.status = tk.Label(bottom, text="", font=self.f_small, fg=MUTED, bg=BG, anchor="w", justify="left", wraplength=480)
        self.status.pack(side="left", fill="x", expand=True)
        # ボタンにフォーカスが残るとキー操作の妨げになるので takefocus を切る
        for text, cmd in (("結果をコピー", self.copy_report), ("デバイス再読込", self.rescan)):
            tk.Button(bottom, text=text, font=self.f_small, fg=TEXT, bg="#2a303a", activebackground="#3a424e",
                      activeforeground=TEXT, relief="flat", padx=10, takefocus=0, command=cmd).pack(side="right", padx=(6, 0))

    def combo(self, parent, row, label, var, width):
        tk.Label(parent, text=label, font=self.f_small, fg=TEXT, bg=PANEL).grid(row=row, column=0, sticky="w", padx=(0, 12), pady=3)
        box = ttk.Combobox(parent, textvariable=var, state="readonly", width=width, takefocus=0)
        box.grid(row=row, column=1, sticky="ew", pady=3)
        box.bind("<<ComboboxSelected>>", self.on_change)
        return box

    def small_combo(self, parent, label, var, values=(), width=3):
        tk.Label(parent, text=label, font=self.f_small, fg=TEXT, bg=PANEL).pack(side="left", padx=(0, 4))
        box = ttk.Combobox(parent, textvariable=var, values=list(values), state="readonly", width=width, takefocus=0)
        box.pack(side="left", padx=(0, 4))
        box.bind("<<ComboboxSelected>>", self.on_change)
        return box

    # ---- デバイス ----
    def load_devices(self):
        self.devices = list_devices()
        ins = [NONE_LABEL] + [d[1] for d in self.devices if d[2] > 0]
        outs = [d[1] for d in self.devices if d[3] > 0]
        self.c_in.configure(values=ins)
        self.c_out.configure(values=outs)
        if self.v_in.get() not in ins:
            self.v_in.set(ins[1] if len(ins) > 1 else NONE_LABEL)
        if self.v_out.get() not in outs:
            self.v_out.set(outs[0] if outs else "")
        # 同じAPI同士を初期値にする（ASIOは入出力を同じデバイスにする必要がある）
        if self.v_in.get() != NONE_LABEL and self.v_in.get() in outs:
            self.v_out.set(self.v_in.get())

    def device_by_label(self, label):
        return next((d for d in self.devices if d[1] == label), None)

    def update_channels(self):
        in_dev = self.device_by_label(self.v_in.get())
        out_dev = self.device_by_label(self.v_out.get())
        in_values = [str(i + 1) for i in range(in_dev[2])] if in_dev else ["1"]
        out_values = [str(i + 1) for i in range(out_dev[3])] if out_dev else ["1"]
        for box, var, values, fallback in ((self.c_in_a, self.v_in_a, in_values, 0), (self.c_in_b, self.v_in_b, in_values, 1),
                                           (self.c_out_l, self.v_out_l, out_values, 0), (self.c_out_r, self.v_out_r, out_values, 1)):
            box.configure(values=values)
            if var.get() not in values:
                var.set(values[min(fallback, len(values) - 1)])

    def reopen(self):
        self.update_channels()
        in_dev = self.device_by_label(self.v_in.get())
        out_dev = self.device_by_label(self.v_out.get())
        self.error = ""
        if out_dev is None:
            self.engine.close()
            self.error = "出力デバイスが見つかりません"
            return
        try:
            self.engine.open(
                in_dev[0] if in_dev else None, out_dev[0],
                (int(self.v_in_a.get()) - 1, int(self.v_in_b.get()) - 1),
                (int(self.v_out_l.get()) - 1, int(self.v_out_r.get()) - 1),
                int(self.v_rate.get()), self.v_latency.get(),
            )
        except Exception as e:
            log_exception()
            self.error = f"開けません: {e}"

    def on_change(self, _event=None):
        self.reopen()
        self.root.focus_force()

    def rescan(self):
        self.engine.close()
        rescan_devices()
        self.load_devices()
        self.reopen()
        self.root.focus_force()

    # ---- キー ----
    def on_enter(self, _event=None):
        if self.enter_down:  # 長押しのリピートでは反応しない
            return
        self.enter_down = True
        self.engine.fire()

    def on_enter_release(self, _event=None):
        self.enter_down = False

    # ---- 定期更新 ----
    def tick(self):
        try:
            self._tick()
        except Exception:
            log_exception()
        self.root.after(40, self.tick)

    def _tick(self):
        e = self.engine
        for i, meter in enumerate(self.meters[:2]):
            meter.set(e.peaks[i])
            e.peaks[i] = 0.0
        self.meters[2].set(e.out_peak)
        e.out_peak = 0.0

        self.has_focus = self.window_has_focus()
        if self.has_focus:
            self.focus_bar.configure(text="", bg=BG)
        else:
            self.focus_bar.configure(text="このソフトが選ばれていません（Enterが効きません）", bg=RED)

        if self.error:
            self.fire_label.configure(text="音を出せません", fg=RED)
        elif e.playing:
            self.fire_label.configure(text="音が出ています", fg=GREEN)
        else:
            self.fire_label.configure(text="Enter で音を出す", fg=TEXT)

        if e.key_to_buffer_ms and e.output_latency_ms is not None:
            last = e.key_to_buffer_ms[-1]
            samples = e.key_to_buffer_ms
            self.latency_label.configure(text=f"Enterから音まで 約 {last + e.output_latency_ms:.0f} ms")
            self.detail_label.configure(
                text=(f"ソフト内 {last:.1f} ms ＋ デバイスの申告値 {e.output_latency_ms:.1f} ms"
                      f"（{len(samples)} 回、最大 {max(samples) + e.output_latency_ms:.0f} ms）\nEsc で止める"))

        if self.error:
            self.status.configure(text=self.error, fg=RED)
        elif e.problems:
            self.status.configure(text=f"音声の取りこぼし {e.problems} 回", fg=AMBER)
        else:
            self.status.configure(text=f"動作中  /  {e.rate / 1000:g} kHz", fg=MUTED)

    def window_has_focus(self):
        # focus コマンドは、このソフトにフォーカスがなければ空を返す
        return bool(str(self.root.tk.call("focus")))

    def report_text(self):
        e = self.engine
        lines = [
            f"{APP_NAME} {VERSION}",
            f"Windows: {sys.getwindowsversion().major}.{sys.getwindowsversion().build}" if sys.platform == "win32" else sys.platform,
            f"PortAudio: {sd.get_portaudio_version()[1]}",
            f"入力: {self.v_in.get()}  ch {self.v_in_a.get()}/{self.v_in_b.get()}",
            f"出力: {self.v_out.get()}  ch {self.v_out_l.get()}/{self.v_out_r.get()}",
            f"サンプルレート: {self.v_rate.get()}  遅延設定: {self.v_latency.get()}",
            f"デバイス申告の出力遅延: {e.output_latency_ms}",
            f"エラー: {self.error or 'なし'}",
            f"取りこぼし: {e.problems}",
            "Enterからバッファまで(ms): " + ", ".join(f"{v:.1f}" for v in e.key_to_buffer_ms[-20:]),
            "",
            "デバイス一覧:",
        ]
        lines += [f"  {d[1]}  入力{d[2]} 出力{d[3]}" for d in self.devices]
        return "\n".join(lines)

    def copy_report(self):
        self.root.clipboard_clear()
        self.root.clipboard_append(self.report_text())
        self.status.configure(text="コピーしました")
        self.root.focus_force()

    def on_close(self):
        self.engine.close()
        self.root.destroy()


def self_test():
    """自動ビルドで、exeが部品込みで起動できることを確かめる。"""
    report = APP_DIR / "selftest.txt"
    try:
        apis = [a["name"] for a in sd.query_hostapis()]
        if "ASIO" not in apis:
            raise RuntimeError(f"ASIO対応のPortAudioが読み込まれていません: {apis}")
        root = tk.Tk()
        root.withdraw()
        root.update()
        root.destroy()
        report.write_text(f"OK\n{sd.get_portaudio_version()[1]}\n{apis}\n{len(make_tone(48000))}\n", encoding="utf-8")
        return 0
    except Exception:
        report.write_text(f"NG\n{traceback.format_exc()}", encoding="utf-8")
        return 1


def tune_windows():
    if sys.platform != "win32":
        return
    import ctypes
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass


def main():
    if "--self-test" in sys.argv:
        sys.exit(self_test())
    tune_windows()
    root = tk.Tk()
    root.report_callback_exception = log_exception
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
