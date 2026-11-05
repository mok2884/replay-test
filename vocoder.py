#!/usr/bin/env python3
"""ボコーダー — しゃべった声を、鍵盤で決めた音程の声に変えるソフト。

  声（入力）→ 帯域ごとの音量を調べる
  鍵盤の音（のこぎり波）→ その音量で帯域ごとに整形 → 出力

オーディオの処理はコールバック内で完結させ、画面側は「鳴らす音の集合」を渡すだけにする。
"""
import ctypes
import datetime
import json
import os
import sys
import threading
import time
import traceback
from pathlib import Path

os.environ.setdefault("SD_ENABLE_ASIO", "1")

import numpy as np
import sounddevice as sd
import tkinter as tk
from tkinter import font as tkfont, simpledialog, ttk

APP_NAME = "ボコーダー"
VERSION = "0.1"
APP_DIR = Path(sys.executable if getattr(sys, "frozen", False) else __file__).resolve().parent
CONFIG_PATH = APP_DIR / "vocoder_config.json"
API_ORDER = ("ASIO", "WASAPI", "WDM-KS", "DirectSound", "MME")

FRAME = 512
HOP = 128
BANDS = 22
BAND_LOW = 120.0
BAND_HIGH = 12000.0
CONSONANT_FROM = 3000.0
ATTACK_S = 0.003
RELEASE_S = 0.040
NOTE_ATTACK_S = 0.004
NOTE_RELEASE_S = 0.030
MAX_VOICES = 10
TABLE_SIZE = 2048

NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")

DEFAULT_CHORDS = [
    [36],                # 低い1音（ロボット）
    [48],                # 中くらいの1音
    [72],                # 高い1音
    [48, 55, 60, 64],    # C
    [45, 52, 57, 60],    # Am
    [41, 48, 53, 57],    # F
    [43, 50, 55, 59],    # G
    [38, 45, 50, 53],    # Dm
]

DEFAULT_CONFIG = {
    "input_device": "",
    "output_device": "",
    "samplerate": 48000,
    "input_channel": 1,
    "output_left": 1,
    "output_right": 2,
    "latency": "low",
    "input_gain_db": 0.0,
    "output_gain_db": 0.0,
    "consonant": 0.4,
    "dry": 0.0,
    "gate_db": -55.0,
    "chords": DEFAULT_CHORDS,
}

# PCキーボードの配置（ピアノと同じ並び）。数字は基準の C からの半音数
KEY_NOTES = {
    "a": 0, "w": 1, "s": 2, "e": 3, "d": 4, "f": 5, "t": 6, "g": 7, "y": 8, "h": 9,
    "u": 10, "j": 11, "k": 12, "o": 13, "l": 14, "p": 15,
}
# Windowsの仮想キーコード（日本語入力の状態に左右されない）
VK_TO_LETTER = {ord(c.upper()): c for c in KEY_NOTES}


def note_name(note):
    return f"{NOTE_NAMES[note % 12]}{note // 12 - 1}"


def chord_label(notes):
    if len(notes) == 1:
        return note_name(notes[0])
    return " ".join(note_name(n) for n in notes)


def log_exception(*_args):
    try:
        with open(APP_DIR / "error.log", "a", encoding="utf-8") as f:
            f.write(f"--- {datetime.datetime.now():%Y-%m-%d %H:%M:%S}\n{traceback.format_exc()}\n")
    except OSError:
        pass


# ----------------------------------------------------------- 音の処理 -------
class Vocoder:
    """入力サンプルを受け取り、ボコーダーの出力を返す。オーディオスレッド専用。"""

    def __init__(self, rate):
        self.rate = rate
        self.window = np.sin(np.pi * (np.arange(FRAME) + 0.5) / FRAME).astype(np.float32)
        self.ola_scale = 2.0 * HOP / FRAME  # sin窓を2回かけて75%重ねると合計が FRAME/(2*HOP) になる
        freqs = np.fft.rfftfreq(FRAME, 1 / rate)
        self.nbins = len(freqs)
        edges = np.geomspace(BAND_LOW, min(BAND_HIGH, rate * 0.45), BANDS + 1)
        band_of_bin = np.clip(np.searchsorted(edges, freqs, side="right") - 1, 0, BANDS - 1)
        # 低域は1帯域に1ビン以上を保証する（ビンのない帯域を詰める）
        used = np.unique(band_of_bin)
        self.nbands = len(used)
        self.band_of_bin = np.searchsorted(used, band_of_bin)
        self.bins_per_band = np.bincount(self.band_of_bin, minlength=self.nbands).astype(np.float64)
        self.band_center = np.array([np.exp(np.log(np.maximum(freqs[self.band_of_bin == b], 1.0)).mean())
                                     for b in range(self.nbands)])
        self.log_freqs = np.log(np.maximum(freqs, 1.0))
        self.log_centers = np.log(self.band_center)
        self.consonant_mask = np.clip((freqs - CONSONANT_FROM) / 1500.0, 0.0, 1.0).astype(np.float32)
        hop_s = HOP / rate
        self.attack = np.exp(-hop_s / ATTACK_S)
        self.release = np.exp(-hop_s / RELEASE_S)
        self.note_step_on = HOP / (NOTE_ATTACK_S * rate)
        self.note_step_off = HOP / (NOTE_RELEASE_S * rate)
        self.tables = self._make_tables()
        self.input_frame = np.zeros(FRAME, dtype=np.float32)
        self.carrier_frame = np.zeros(FRAME, dtype=np.float32)
        self.noise_frame = np.zeros(FRAME, dtype=np.float32)
        self.ola = np.zeros(FRAME, dtype=np.float32)
        self.env = np.zeros(self.nbands)
        self.pending = np.zeros(0, dtype=np.float32)
        self.voices = {}  # note -> [phase, gain, sounding]
        self.rng = np.random.default_rng()
        self.notes = frozenset()
        self.consonant = 0.4
        self.dry = 0.0
        self.gate = 10 ** (-55.0 / 20)
        self.gate_gain = 0.0

    def _make_tables(self):
        """帯域制限したのこぎり波を、音域ごとに作っておく（折り返しノイズ防止）。"""
        tables = {}
        phase = 2 * np.pi * np.arange(TABLE_SIZE) / TABLE_SIZE
        for octave in range(0, 11):
            top_freq = 440.0 * 2 ** ((12 * (octave + 1) - 69) / 12)
            harmonics = max(1, int(self.rate * 0.45 / top_freq))
            harmonics = min(harmonics, TABLE_SIZE // 2 - 1)
            k = np.arange(1, harmonics + 1)[:, None]
            table = (np.sin(k * phase) / k).sum(axis=0)
            tables[octave] = (table / np.abs(table).max()).astype(np.float32)
        return tables

    @staticmethod
    def note_freq(note):
        return 440.0 * 2 ** ((note - 69) / 12)

    def _carrier(self, n):
        """n サンプル分の鍵盤の音を作る。戻り値は (音, 鳴っている度合い)。"""
        notes = self.notes
        for note in notes:
            if note not in self.voices and len(self.voices) < MAX_VOICES:
                self.voices[note] = [self.rng.random(), 0.0, True]
        out = np.zeros(n, dtype=np.float32)
        total = 0.0
        finished = []
        for note, voice in self.voices.items():
            phase, gain, _ = voice
            sounding = note in notes
            target = 1.0 if sounding else 0.0
            step = self.note_step_on if sounding else self.note_step_off
            new_gain = min(target, gain + step) if sounding else max(target, gain - step)
            freq = self.note_freq(note)
            inc = freq / self.rate
            table = self.tables[min(10, max(0, note // 12))]
            pos = (phase + inc * np.arange(n)) % 1.0 * TABLE_SIZE
            i0 = pos.astype(np.int32)
            frac = (pos - i0).astype(np.float32)
            wave = table[i0] + (table[(i0 + 1) % TABLE_SIZE] - table[i0]) * frac
            out += wave * np.linspace(gain, new_gain, n, dtype=np.float32)
            voice[0] = (phase + inc * n) % 1.0
            voice[1] = new_gain
            total += new_gain
            if not sounding and new_gain <= 0.0:
                finished.append(note)
        for note in finished:
            del self.voices[note]
        return out, min(1.0, total)

    def _process_hop(self, samples):
        self.input_frame[:-HOP] = self.input_frame[HOP:]
        self.input_frame[-HOP:] = samples
        carrier, active = self._carrier(HOP)
        self.carrier_frame[:-HOP] = self.carrier_frame[HOP:]
        self.carrier_frame[-HOP:] = carrier
        self.noise_frame[:-HOP] = self.noise_frame[HOP:]
        self.noise_frame[-HOP:] = (self.rng.standard_normal(HOP) * 0.3).astype(np.float32) * active

        win = self.window
        m_spec = np.fft.rfft(self.input_frame * win)
        c_spec = np.fft.rfft(self.carrier_frame * win)
        if self.consonant > 0:
            c_spec = c_spec + np.fft.rfft(self.noise_frame * win) * self.consonant_mask * self.consonant * 3.0

        m_power = np.bincount(self.band_of_bin, (m_spec.real ** 2 + m_spec.imag ** 2), self.nbands) / self.bins_per_band
        c_power = np.bincount(self.band_of_bin, (c_spec.real ** 2 + c_spec.imag ** 2), self.nbands) / self.bins_per_band
        m_level = np.sqrt(m_power)

        # 声の大きさに追従（立ち上がりは速く、消え方はゆっくり）
        coef = np.where(m_level > self.env, self.attack, self.release)
        self.env = coef * self.env + (1 - coef) * m_level

        # 声が入っていないときは閉じて、鍵盤を押していても無音にする
        rms = np.sqrt(np.mean((self.input_frame * win) ** 2) * 2)
        gate_target = 1.0 if rms > self.gate else 0.0
        self.gate_gain += (gate_target - self.gate_gain) * (0.5 if gate_target else 0.08)

        floor = max(1e-4, np.sqrt(c_power.max()) * 1e-3) if len(c_power) else 1e-4
        band_gain = self.env / np.maximum(np.sqrt(c_power), floor)
        band_gain = np.minimum(band_gain, 30.0) * self.gate_gain
        # 帯域の境目で音が段にならないよう、対数周波数で滑らかにつなぐ
        bin_gain = np.interp(self.log_freqs, self.log_centers, band_gain)
        spec = c_spec * bin_gain
        if self.dry > 0:
            spec = spec + m_spec * self.dry
        frame = np.fft.irfft(spec, FRAME).astype(np.float32) * win * self.ola_scale

        self.ola += frame
        out = self.ola[:HOP].copy()
        self.ola[:-HOP] = self.ola[HOP:]
        self.ola[-HOP:] = 0.0
        return out

    def process(self, samples):
        """入力を受け取り、計算できた分の出力を返す（入力と同じ長さとは限らない）。"""
        data = np.concatenate((self.pending, samples)) if len(self.pending) else samples
        count = len(data) // HOP
        if count == 0:
            self.pending = data.copy()
            return np.zeros(0, dtype=np.float32)
        out = np.empty(count * HOP, dtype=np.float32)
        for i in range(count):
            out[i * HOP:(i + 1) * HOP] = self._process_hop(data[i * HOP:(i + 1) * HOP])
        self.pending = data[count * HOP:].copy()
        return out


class Engine:
    def __init__(self):
        self.rate = 48000
        self.voc = Vocoder(self.rate)
        self.streams = []
        self.in_channel = 0
        self.out_left = 0
        self.out_right = 1
        self.input_gain = 1.0
        self.output_gain = 1.0
        self.limit_gain = 1.0
        self.fifo_lock = threading.Lock()
        self.in_fifo = []
        self.in_count = 0
        self.out_fifo = np.zeros(0, dtype=np.float32)
        self.primed = False
        self.separate = False
        self.in_block = 0
        self.problems = 0
        self.underruns = 0
        self.peak_in = 0.0
        self.peak_out = 0.0
        self.last_in = 0.0
        self.last_out = 0.0
        self.max_callback = 0.0
        self.output_latency_ms = None
        self.has_stream = False

    def set_notes(self, notes):
        self.voc.notes = frozenset(notes)

    # ---- オーディオスレッド ----
    def _on_input(self, indata, status):
        self.last_in = time.monotonic()
        if status.input_overflow or status.input_underflow:
            self.problems += 1
        x = indata[:, self.in_channel]
        if len(x):
            peak = float(np.abs(x).max()) * self.input_gain
            if peak > self.peak_in:
                self.peak_in = peak
        if len(x) > self.in_block:
            self.in_block = len(x)
        with self.fifo_lock:
            self.in_fifo.append(x.copy())
            self.in_count += len(x)
            # 別々のストリームで時計がずれても、遅れが溜まらないように捨てる
            limit = max(self.rate // 10, len(x) * 6)
            while self.in_count > limit and len(self.in_fifo) > 1:
                self.in_count -= len(self.in_fifo.pop(0))

    def _on_output(self, outdata, frames, status):
        started = time.perf_counter()
        self.last_out = time.monotonic()
        outdata.fill(0)
        if status.output_underflow or status.output_overflow:
            self.problems += 1
        with self.fifo_lock:
            chunks, self.in_fifo = self.in_fifo, []
            self.in_count = 0
        if chunks:
            samples = chunks[0] if len(chunks) == 1 else np.concatenate(chunks)
            # 異常な値が1度でも入ると処理全体が壊れるので、ここで止める
            clean = np.nan_to_num(samples * self.input_gain, nan=0.0, posinf=0.0, neginf=0.0)
            wet = self.voc.process(np.clip(clean, -4.0, 4.0))
            self.out_fifo = np.concatenate((self.out_fifo, wet)) if len(self.out_fifo) else wet
        # 入力と出力が別のときは、入力のかたまりの分だけ余裕を持って鳴らし始める
        cushion = frames + (self.in_block + HOP if self.separate else 0)
        if not self.primed:
            if len(self.out_fifo) < cushion:
                return
            self.primed = True
        n = min(frames, len(self.out_fifo))
        if n < frames:
            self.underruns += 1
        if n:
            block = self.out_fifo[:n] * self.output_gain
            self.out_fifo = self.out_fifo[n:]
            # リミッター（割れ防止。かかるときは素早く、戻りはゆっくり）
            peak = float(np.abs(block).max())
            target = min(1.0, 0.89 / peak) if peak > 0 else 1.0
            new_gain = min(target, self.limit_gain + (1 - self.limit_gain) * 0.05)
            ramp = np.full(n, new_gain, dtype=np.float32)
            edge = min(32, n)  # 下げるときはブロックの頭で素早く下げる
            ramp[:edge] = np.linspace(self.limit_gain, new_gain, edge, dtype=np.float32)
            block = np.clip(block * ramp, -0.99, 0.99)
            self.limit_gain = new_gain
            outdata[:n, self.out_left] = block
            outdata[:n, self.out_right] = block
            peak = float(np.abs(block).max())
            if peak > self.peak_out:
                self.peak_out = peak
        if n < frames:
            self.primed = False
        # 余分に溜まった分は遅延になるので捨てる
        excess = len(self.out_fifo) - (cushion + 2 * HOP)
        if excess > max(frames, self.in_block):
            self.out_fifo = self.out_fifo[excess:]
        took = time.perf_counter() - started
        if took > self.max_callback:
            self.max_callback = took

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

    # ---- 操作 ----
    def open(self, cfg):
        self.close()
        in_dev = find_device(cfg["input_device"], "in")
        out_dev = find_device(cfg["output_device"], "out")
        if in_dev is None:
            raise RuntimeError(f"入力デバイスが見つかりません: {cfg['input_device'] or '未設定'}")
        if out_dev is None:
            raise RuntimeError(f"出力デバイスが見つかりません: {cfg['output_device'] or '未設定'}")
        rate = int(cfg["samplerate"])
        old_notes = self.voc.notes
        if rate != self.rate:
            self.rate = rate
            self.voc = Vocoder(rate)
            self.voc.notes = old_notes
        self.apply_mix(cfg)
        self.in_channel = int(cfg["input_channel"]) - 1
        self.out_left = int(cfg["output_left"]) - 1
        self.out_right = int(cfg["output_right"]) - 1
        n_in = self.in_channel + 1
        n_out = max(self.out_left, self.out_right) + 1
        common = dict(samplerate=rate, dtype="float32", latency=cfg["latency"])
        self.in_fifo, self.in_count = [], 0
        self.out_fifo = np.zeros(0, dtype=np.float32)
        self.primed = False
        self.in_block = 0
        self.problems = self.underruns = 0
        self.last_in = self.last_out = time.monotonic()
        self.separate = in_dev != out_dev
        try:
            if not self.separate:
                self.streams = [sd.Stream(device=(in_dev, out_dev), channels=(n_in, n_out),
                                          callback=self._duplex, **common)]
                latency = self.streams[0].latency
                self.output_latency_ms = (latency[0] + latency[1]) * 1000
            else:
                self.streams = [
                    sd.InputStream(device=in_dev, channels=n_in, callback=self._input_only, **common),
                    sd.OutputStream(device=out_dev, channels=n_out, callback=self._output_only, **common),
                ]
                self.output_latency_ms = (self.streams[0].latency + self.streams[1].latency) * 1000
            for s in self.streams:
                s.start()
        except Exception:
            self.close()
            raise
        self.has_stream = True

    def apply_mix(self, cfg):
        self.input_gain = 10 ** (float(cfg["input_gain_db"]) / 20)
        self.output_gain = 10 ** (float(cfg["output_gain_db"]) / 20)
        self.voc.consonant = float(cfg["consonant"])
        self.voc.dry = float(cfg["dry"])
        self.voc.gate = 10 ** (float(cfg["gate_db"]) / 20)

    def close(self):
        for s in self.streams:
            try:
                s.stop()
                s.close()
            except Exception:
                pass
        self.streams = []
        self.has_stream = False

    def healthy(self):
        if not self.has_stream:
            return False
        try:
            if not all(s.active for s in self.streams):
                return False
        except Exception:
            return False
        now = time.monotonic()
        return now - self.last_in < 2.0 and now - self.last_out < 2.0

    def latency_estimate_ms(self):
        """声を入れてから出るまでの目安。処理分＋機器の申告値。"""
        processing = (FRAME - HOP / 2) / self.rate * 1000
        device = self.output_latency_ms or 0.0
        waiting = len(self.out_fifo) / self.rate * 1000  # 入出力が別のときの待ち時間
        return processing + device + waiting


# ---------------------------------------------------------- デバイス -------
def device_label(info):
    api = sd.query_hostapis(info["hostapi"])["name"].replace("Windows ", "")
    return f"[{api}] {info['name']}"


def api_rank(label):
    for i, api in enumerate(API_ORDER):
        if label.startswith(f"[{api}]"):
            return i
    return len(API_ORDER)


def list_devices(kind):
    key = "max_input_channels" if kind == "in" else "max_output_channels"
    items = [(i, device_label(d), d[key]) for i, d in enumerate(sd.query_devices()) if d[key] > 0]
    return sorted(items, key=lambda it: (api_rank(it[1]), it[0]))


def find_device(label, kind):
    for i, lab, _ in list_devices(kind):
        if lab == label:
            return i
    return None


def rescan_devices():
    try:
        sd._terminate()
        sd._initialize()
    except Exception:
        log_exception()


def load_config():
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    try:
        cfg.update(json.loads(CONFIG_PATH.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        pass
    return cfg


def save_config(cfg):
    try:
        CONFIG_PATH.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        log_exception()


def tune_windows():
    if sys.platform != "win32":
        return
    w = ctypes.windll
    for call in (
        lambda: w.shcore.SetProcessDpiAwareness(1),
        lambda: w.imm32.ImmDisableIME(-1),
        lambda: w.winmm.timeBeginPeriod(1),
        lambda: w.kernel32.SetPriorityClass(w.kernel32.GetCurrentProcess(), 0x00000080),
        lambda: w.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001),
    ):
        try:
            call()
        except Exception:
            pass


# -------------------------------------------------------------- 画面 -------
BG = "#14171c"
PANEL = "#1f242c"
TEXT = "#eef1f5"
MUTED = "#98a2b0"
RED = "#e5343f"
GREEN = "#2fbf71"
AMBER = "#e6a817"
BUTTON = "#2f3641"
BUTTON_ON = "#2fbf71"
KEY_WHITE = "#e9ecf1"
KEY_BLACK = "#20242b"
KEY_ON = "#2fbf71"
KEY_LATCH = "#e6a817"


class Meter(tk.Canvas):
    def __init__(self, parent, height=14):
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


class Piano(tk.Canvas):
    """2オクターブ半の鍵盤。マウスのドラッグで音を移動できる。"""

    WHITE = (0, 2, 4, 5, 7, 9, 11)
    SPAN = 29

    def __init__(self, app, parent):
        super().__init__(parent, height=190, bg=BG, highlightthickness=0)
        self.app = app
        self.mouse_note = None
        self.rects = {}
        self.bind("<Configure>", lambda _e: self.redraw())
        self.bind("<Button-1>", self.on_press)
        self.bind("<B1-Motion>", self.on_drag)
        self.bind("<ButtonRelease-1>", self.on_release)

    def layout(self):
        base = self.app.base_note
        notes = range(base, base + self.SPAN)
        whites = [n for n in notes if n % 12 in self.WHITE]
        w = self.winfo_width() / max(1, len(whites))
        h = self.winfo_height()
        rects = {}
        for i, n in enumerate(whites):
            rects[n] = (i * w, 0, (i + 1) * w, h, True)
        for n in notes:
            if n % 12 not in self.WHITE and n - 1 in rects:
                x = rects[n - 1][2]
                rects[n] = (x - w * 0.3, 0, x + w * 0.3, h * 0.62, False)
        return rects

    def redraw(self):
        self.delete("all")
        self.rects = self.layout()
        live = self.app.live_notes()
        latched = self.app.latched
        for black in (False, True):
            for n, (x0, y0, x1, y1, is_white) in self.rects.items():
                if is_white == black:
                    continue
                base = KEY_WHITE if is_white else KEY_BLACK
                fill = KEY_ON if n in live else KEY_LATCH if n in latched else base
                self.create_rectangle(x0, y0, x1, y1, fill=fill, outline="#000000")
                if is_white:
                    letter = self.app.letter_for_note(n)
                    if n % 12 == 0:
                        self.create_text((x0 + x1) / 2, y1 - 14, text=note_name(n), fill="#333a44", font=self.app.f_small)
                    if letter:
                        self.create_text((x0 + x1) / 2, y1 - 34, text=letter.upper(), fill="#7c8592", font=self.app.f_small)
                else:
                    letter = self.app.letter_for_note(n)
                    if letter:
                        self.create_text((x0 + x1) / 2, y1 - 14, text=letter.upper(), fill="#98a2b0", font=self.app.f_small)

    def note_at(self, x, y):
        for black in (True, False):
            for n, (x0, y0, x1, y1, is_white) in self.rects.items():
                if is_white != black and x0 <= x < x1 and y0 <= y < y1:
                    return n
        return None

    def on_press(self, event):
        self.mouse_note = self.note_at(event.x, event.y)
        if self.mouse_note is not None:
            self.app.note_on(("mouse",), self.mouse_note)

    def on_drag(self, event):
        note = self.note_at(event.x, event.y)
        if note != self.mouse_note and note is not None:
            self.app.note_off(("mouse",))
            self.mouse_note = note
            self.app.note_on(("mouse",), note)

    def on_release(self, _event):
        self.mouse_note = None
        self.app.note_off(("mouse",))


class App:
    def __init__(self, root):
        self.root = root
        self.cfg = load_config()
        self.engine = Engine()
        self.base_note = 48  # C3
        self.held = {}  # 押しているもの -> 音の集合
        self.latched = set()
        self.keep = False
        self.down = set()
        self.engine_error = ""
        self.tick_count = 0
        self.dialog_open = False
        self.piano_dirty = True

        families = set(tkfont.families())
        family = next((f for f in ("Yu Gothic UI", "Meiryo UI", "Noto Sans CJK JP") if f in families), "TkDefaultFont")
        self.f_big = (family, 18, "bold")
        self.f_mid = (family, 12, "bold")
        self.f_small = (family, 10)
        self.f_chord = (family, 13, "bold")

        root.title(f"{APP_NAME} {VERSION}")
        root.configure(bg=BG)
        root.geometry("1040x640")
        root.minsize(860, 560)
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.build()
        root.bind("<KeyPress>", self.on_key)
        root.bind("<KeyRelease>", self.on_key_release)
        root.bind("<FocusOut>", self.on_focus_out)
        self.start_audio(first_run=True)
        root.focus_force()
        self.tick()

    # ---- 組み立て ----
    def button(self, parent, text, command, font=None, **kw):
        return tk.Button(parent, text=text, command=command, font=font or self.f_small, fg=TEXT, bg=BUTTON,
                         activebackground="#3d4653", activeforeground=TEXT, relief="flat", takefocus=0, **kw)

    def build(self):
        style = ttk.Style(self.root)
        style.theme_use("clam")

        self.banner = tk.Label(self.root, text="", font=self.f_big, fg="#ffffff", bg=PANEL, pady=8)
        self.banner.pack(fill="x")

        chords = tk.Frame(self.root, bg=BG)
        chords.pack(fill="x", padx=12, pady=(10, 4))
        tk.Label(chords, text="コード（数字キー。右クリックで今の音を登録）", font=self.f_small, fg=MUTED, bg=BG).grid(row=0, column=0, columnspan=8, sticky="w")
        self.chord_buttons = []
        for i in range(8):
            chords.columnconfigure(i, weight=1, uniform="chord")
            b = tk.Label(chords, text="", font=self.f_chord, fg=TEXT, bg=BUTTON, height=2, wraplength=120, cursor="hand2")
            b.grid(row=1, column=i, sticky="ew", padx=2, pady=(4, 0))
            b.bind("<ButtonPress-1>", lambda _e, k=i: self.chord_on(k, ("chordmouse",)))
            b.bind("<ButtonRelease-1>", lambda _e: self.note_off(("chordmouse",)))
            b.bind("<Button-3>", lambda _e, k=i: self.register_chord(k))
            self.chord_buttons.append(b)
        self.refresh_chord_labels()

        ctrl = tk.Frame(self.root, bg=BG)
        ctrl.pack(fill="x", padx=12, pady=8)
        self.keep_button = tk.Button(ctrl, text="キープ（Space）", font=self.f_mid, fg=TEXT, bg=BUTTON,
                                     activebackground="#3d4653", activeforeground=TEXT, relief="flat",
                                     takefocus=0, width=16, command=self.toggle_keep)
        self.keep_button.pack(side="left", ipady=8)
        self.button(ctrl, "全部止める（Esc）", self.all_off, font=self.f_mid, width=16).pack(side="left", padx=8, ipady=8)
        self.button(ctrl, "オクターブ下（Z）", lambda: self.shift_octave(-1), font=self.f_mid, width=14).pack(side="left", padx=(8, 4), ipady=8)
        self.button(ctrl, "上（X）", lambda: self.shift_octave(1), font=self.f_mid, width=7).pack(side="left", ipady=8)
        self.octave_label = tk.Label(ctrl, text="", font=self.f_mid, fg=TEXT, bg=BG)
        self.octave_label.pack(side="left", padx=10)
        self.button(ctrl, "設定", self.open_settings, padx=12).pack(side="right")
        self.button(ctrl, "再接続", self.reconnect, padx=12).pack(side="right", padx=6)

        bottom = tk.Frame(self.root, bg=BG)
        bottom.pack(side="bottom", fill="x", padx=12, pady=(4, 8))
        self.status = tk.Label(bottom, text="", font=self.f_small, fg=MUTED, bg=BG, anchor="w")
        self.status.pack(fill="x")

        mix = tk.Frame(self.root, bg=PANEL, padx=12, pady=8)
        mix.pack(side="bottom", fill="x", padx=12, pady=(8, 2))
        for c in (1, 3):
            mix.columnconfigure(c, weight=1)
        self.scales = {}
        for row, col, label, key, lo, hi in ((0, 0, "入力の音量", "input_gain_db", -24, 24),
                                             (1, 0, "出力の音量", "output_gain_db", -24, 12),
                                             (0, 2, "子音の強さ", "consonant", 0, 1),
                                             (1, 2, "元の声を混ぜる", "dry", 0, 1)):
            tk.Label(mix, text=label, font=self.f_small, fg=MUTED, bg=PANEL).grid(row=row, column=col, sticky="w", padx=(0 if col == 0 else 16, 8), pady=3)
            var = tk.DoubleVar(value=float(self.cfg[key]))
            s = ttk.Scale(mix, from_=lo, to=hi, variable=var, takefocus=0,
                          command=lambda _v, k=key, vr=var: self.on_mix(k, vr))
            s.grid(row=row, column=col + 1, sticky="ew", pady=3)
            self.scales[key] = var
        meters = tk.Frame(mix, bg=PANEL)
        meters.grid(row=0, column=4, rowspan=2, sticky="ew", padx=(16, 0))
        mix.columnconfigure(4, weight=1)
        meters.columnconfigure(1, weight=1)
        self.meter_in = Meter(meters)
        self.meter_out = Meter(meters)
        for i, (text, m) in enumerate((("声", self.meter_in), ("出力", self.meter_out))):
            tk.Label(meters, text=text, font=self.f_small, fg=MUTED, bg=PANEL).grid(row=i, column=0, sticky="w", padx=(0, 8), pady=3)
            m.grid(row=i, column=1, sticky="ew", pady=3)

        self.piano = Piano(self, self.root)
        self.piano.pack(fill="both", expand=True, padx=12, pady=4)

    # ---- オーディオ ----
    def start_audio(self, first_run=False):
        if not self.cfg["input_device"] or not self.cfg["output_device"]:
            self.engine_error = "デバイスが未設定です。「設定」から選んでください"
            if first_run:
                self.root.after(300, self.open_settings)
            return
        try:
            self.engine.open(self.cfg)
            self.engine_error = ""
        except Exception as e:
            log_exception()
            self.engine_error = f"オーディオを開けません: {e}"

    def reconnect(self):
        self.engine.close()
        rescan_devices()
        self.start_audio()

    def open_settings(self):
        if self.dialog_open:
            return
        self.all_off()
        self.engine.close()
        rescan_devices()
        SettingsDialog(self)

    def apply_settings(self, cfg):
        self.cfg = cfg
        save_config(cfg)
        self.start_audio()

    def on_mix(self, key, var):
        self.cfg[key] = round(float(var.get()), 3)
        self.engine.apply_mix(self.cfg)
        self.root.focus_set()

    # ---- 音の管理 ----
    def live_notes(self):
        out = set()
        for notes in self.held.values():
            out |= notes
        return out

    def update_notes(self):
        self.engine.set_notes(self.live_notes() | self.latched)
        self.piano_dirty = True

    def note_on(self, source, notes):
        notes = {notes} if isinstance(notes, int) else set(notes)
        was_idle = not self.held
        self.held[source] = notes
        # キープ中に新しく押したら、保持している音を置き換える
        if self.keep:
            self.latched = set(notes) if was_idle else self.latched | notes
        self.update_notes()

    def note_off(self, source):
        if self.held.pop(source, None) is not None:
            self.update_notes()

    def toggle_keep(self):
        if self.keep:
            self.keep = False
            self.latched = set()
        else:
            self.keep = True
            self.latched = self.live_notes()
        self.update_notes()
        self.root.focus_set()

    def all_off(self):
        self.held.clear()
        self.latched = set()
        self.keep = False
        self.update_notes()

    def chord_on(self, index, source):
        chords = self.cfg["chords"]
        if index < len(chords) and chords[index]:
            self.note_on(source, chords[index])

    def register_chord(self, index):
        notes = sorted(self.live_notes() | self.latched)
        if notes:
            self.cfg["chords"][index] = notes
            save_config(self.cfg)
            self.refresh_chord_labels()
            self.show_status(f"コード {index + 1} に {chord_label(notes)} を登録しました")
        else:
            self.show_status("音を鳴らしたまま右クリックすると登録できます")

    def refresh_chord_labels(self):
        for i, b in enumerate(self.chord_buttons):
            notes = self.cfg["chords"][i] if i < len(self.cfg["chords"]) else []
            b.configure(text=f"{i + 1}\n{chord_label(notes)}" if notes else f"{i + 1}")

    def shift_octave(self, delta):
        new = self.base_note + delta * 12
        if 24 <= new <= 72:
            self.base_note = new
            self.piano_dirty = True
        self.root.focus_set()

    def letter_for_note(self, note):
        offset = note - self.base_note
        for letter, value in KEY_NOTES.items():
            if value == offset:
                return letter
        return None

    # ---- キー ----
    def letter_of(self, event):
        ks = event.keysym
        if len(ks) == 1 and ks.isascii() and ks.isalpha():
            return ks.lower()
        if sys.platform == "win32":
            return VK_TO_LETTER.get(event.keycode) or {90: "z", 88: "x"}.get(event.keycode)
        return None

    def on_key(self, event):
        if self.dialog_open:
            return
        code = event.keycode
        if code in self.down:
            return
        self.down.add(code)
        ks = event.keysym
        if ks == "space":
            self.toggle_keep()
            return
        if ks == "Escape":
            self.all_off()
            return
        if len(ks) == 1 and ks in "12345678":
            self.chord_on(int(ks) - 1, ("key", code))
            return
        letter = self.letter_of(event)
        if letter == "z":
            self.shift_octave(-1)
        elif letter == "x":
            self.shift_octave(1)
        elif letter in KEY_NOTES:
            self.note_on(("key", code), self.base_note + KEY_NOTES[letter])

    def on_key_release(self, event):
        code = event.keycode
        self.down.discard(code)
        self.note_off(("key", code))

    def on_focus_out(self, event):
        if event.widget is self.root:  # 離した通知が来なくなるので、押しっぱなしを防ぐ
            self.down.clear()
            self.held.clear()
            self.update_notes()

    # ---- 更新 ----
    def show_status(self, text):
        self.status.configure(text=text, fg=AMBER)
        self.status_hold = time.monotonic() + 3

    status_hold = 0.0

    def tick(self):
        try:
            self._tick()
        except Exception:
            log_exception()
        self.root.after(40, self.tick)

    def _tick(self):
        self.tick_count += 1
        e = self.engine
        self.meter_in.set(e.peak_in)
        self.meter_out.set(e.peak_out)
        e.peak_in = e.peak_out = 0.0
        if self.piano_dirty:
            self.piano_dirty = False
            self.piano.redraw()
            sounding = sorted(self.live_notes() | self.latched)
            for i, b in enumerate(self.chord_buttons):
                notes = self.cfg["chords"][i] if i < len(self.cfg["chords"]) else []
                b.configure(bg=BUTTON_ON if notes and set(notes) == set(sounding) else BUTTON)
            self.keep_button.configure(bg=AMBER if self.keep else BUTTON, fg="#000000" if self.keep else TEXT)
            self.octave_label.configure(text=f"鍵盤 {note_name(self.base_note)}〜")
            if not sounding:
                self.banner_text = "鍵盤を押しながらしゃべる"
            else:
                head = "キープ中" if self.keep else "音"
                self.banner_text = f"{head}：{chord_label(sounding)}"
        has_focus = bool(str(self.root.tk.call("focus")))
        if not has_focus and not self.dialog_open:
            self.banner.configure(text="このソフトが選ばれていません（キーが効きません）", bg=RED)
        else:
            sounding = self.live_notes() | self.latched
            self.banner.configure(text=self.banner_text, bg=("#7a6410" if self.keep and sounding else "#1b6b45" if sounding else PANEL))
        if self.tick_count % 12 == 0 and time.monotonic() >= self.status_hold:
            if not e.healthy():
                text = self.engine_error or "オーディオが止まっています。機器を確認して「再接続」を押してください"
                self.status.configure(text=text, fg=RED)
            else:
                text = f"{self.cfg['input_device']} → {self.cfg['output_device']}  /  遅延の目安 約 {e.latency_estimate_ms():.0f} ms"
                problems = e.problems + e.underruns
                if problems:
                    text += f"  /  音の途切れ {problems} 回（設定の遅延を high にすると減ります）"
                self.status.configure(text=text, fg=AMBER if problems else MUTED)

    banner_text = "鍵盤を押しながらしゃべる"

    def on_close(self):
        save_config(self.cfg)
        self.engine.close()
        self.root.destroy()


class SettingsDialog(tk.Toplevel):
    def __init__(self, app):
        super().__init__(app.root)
        self.app = app
        app.dialog_open = True
        self.title("設定")
        self.configure(bg=BG, padx=16, pady=14)
        self.transient(app.root)
        self.protocol("WM_DELETE_WINDOW", self.cancel)
        cfg = app.cfg
        self.ins = list_devices("in")
        self.outs = list_devices("out")
        self.v_in = tk.StringVar(value=cfg["input_device"])
        self.v_out = tk.StringVar(value=cfg["output_device"])
        self.v_in_ch = tk.StringVar(value=str(cfg["input_channel"]))
        self.v_l = tk.StringVar(value=str(cfg["output_left"]))
        self.v_r = tk.StringVar(value=str(cfg["output_right"]))
        self.v_rate = tk.StringVar(value=str(cfg["samplerate"]))
        self.v_latency = tk.StringVar(value=cfg["latency"])
        self.v_gate = tk.StringVar(value=f"{float(cfg['gate_db']):g}")
        self.c_in = self.row(0, "入力デバイス", self.v_in, [d[1] for d in self.ins], 58)
        self.c_in_ch = self.row(1, "入力チャンネル", self.v_in_ch, [])
        self.c_out = self.row(2, "出力デバイス", self.v_out, [d[1] for d in self.outs], 58)
        self.c_l = self.row(3, "出力チャンネル L", self.v_l, [])
        self.c_r = self.row(4, "出力チャンネル R", self.v_r, [])
        self.row(5, "サンプルレート", self.v_rate, ["44100", "48000", "96000"])
        self.row(6, "遅延", self.v_latency, ["low", "high"])
        self.row(7, "声がないとみなす音量（dB）", self.v_gate, ["-70", "-65", "-60", "-55", "-50", "-45", "-40"])
        buttons = tk.Frame(self, bg=BG)
        buttons.grid(row=8, column=0, columnspan=2, sticky="e", pady=(12, 0))
        tk.Button(buttons, text="キャンセル", font=app.f_small, command=self.cancel, padx=12).pack(side="right")
        tk.Button(buttons, text="OK", font=app.f_small, command=self.ok, padx=20).pack(side="right", padx=(0, 6))
        self.c_in.bind("<<ComboboxSelected>>", lambda _e: self.update_channels())
        self.c_out.bind("<<ComboboxSelected>>", lambda _e: self.update_channels())
        self.update_channels()
        self.grab_set()

    def row(self, r, label, var, values, width=10):
        tk.Label(self, text=label, font=self.app.f_small, fg=TEXT, bg=BG).grid(row=r, column=0, sticky="w", padx=(0, 12), pady=4)
        box = ttk.Combobox(self, textvariable=var, values=values, state="readonly", width=width)
        box.grid(row=r, column=1, sticky="w")
        return box

    def update_channels(self):
        n_in = next((d[2] for d in self.ins if d[1] == self.v_in.get()), 1)
        n_out = next((d[2] for d in self.outs if d[1] == self.v_out.get()), 2)
        in_values = [str(i + 1) for i in range(n_in)]
        out_values = [str(i + 1) for i in range(n_out)]
        for box, var, values, fallback in ((self.c_in_ch, self.v_in_ch, in_values, 0),
                                           (self.c_l, self.v_l, out_values, 0),
                                           (self.c_r, self.v_r, out_values, 1)):
            box.configure(values=values)
            if var.get() not in values:
                var.set(values[min(fallback, len(values) - 1)])

    def close_dialog(self):
        self.app.dialog_open = False
        self.grab_release()
        self.destroy()
        self.app.root.focus_force()

    def ok(self):
        cfg = dict(self.app.cfg)
        cfg.update(input_device=self.v_in.get(), output_device=self.v_out.get(),
                   input_channel=int(self.v_in_ch.get()), output_left=int(self.v_l.get()),
                   output_right=int(self.v_r.get()), samplerate=int(self.v_rate.get()),
                   latency=self.v_latency.get(), gate_db=float(self.v_gate.get()))
        self.close_dialog()
        self.app.apply_settings(cfg)

    def cancel(self):
        self.close_dialog()
        self.app.start_audio()


def self_test():
    """自動ビルドで、exeが部品込みで動くことを確かめる。"""
    report = APP_DIR / "selftest.txt"
    try:
        apis = [a["name"] for a in sd.query_hostapis()]
        if "ASIO" not in apis:
            raise RuntimeError(f"ASIO対応のPortAudioが読み込まれていません: {apis}")
        voc = Vocoder(48000)
        voc.notes = frozenset({48})
        t = np.arange(48000) / 48000
        out = voc.process((0.2 * np.sin(2 * np.pi * 200 * t)).astype(np.float32))
        if not np.any(np.abs(out) > 1e-4):
            raise RuntimeError("ボコーダーの音が出ません")
        root = tk.Tk()
        root.withdraw()
        root.update()
        root.destroy()
        report.write_text(f"OK\n{sd.get_portaudio_version()[1]}\n{apis}\n", encoding="utf-8")
        return 0
    except Exception:
        report.write_text(f"NG\n{traceback.format_exc()}", encoding="utf-8")
        return 1


def main():
    if "--self-test" in sys.argv:
        sys.exit(self_test())
    tune_windows()
    sys.setswitchinterval(0.002)
    root = tk.Tk()
    root.report_callback_exception = log_exception
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
