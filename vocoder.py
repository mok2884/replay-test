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
from tkinter import font as tkfont, ttk

APP_NAME = "VOCODER"
VERSION = "0.2"
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

CHORD_NAMES = ["LOW", "MID", "HIGH", "C", "Am", "F", "G", "Dm"]

DEFAULT_EQ = {
    "on": True,
    "bands": [
        {"f": 120.0, "g": 0.0, "q": 1.0},
        {"f": 450.0, "g": 0.0, "q": 1.0},
        {"f": 2400.0, "g": 0.0, "q": 1.0},
        {"f": 8000.0, "g": 0.0, "q": 1.0},
    ],
}

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
    "eq": DEFAULT_EQ,
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
def eq_response(rate, freqs, bands):
    """パラメトリックEQ（ピーキング）の振幅特性を返す。ゲイン0のバンドは計算しない。"""
    freqs = np.asarray(freqs, dtype=np.float64)
    out = np.ones(len(freqs))
    w = 2 * np.pi * np.minimum(freqs, rate / 2) / rate
    z1 = np.exp(-1j * w)
    z2 = z1 * z1
    for band in bands:
        gain = float(band["g"])
        if gain == 0.0:
            continue
        a = 10 ** (gain / 40)
        w0 = 2 * np.pi * min(float(band["f"]), rate * 0.49) / rate
        alpha = np.sin(w0) / (2 * float(band["q"]))
        cos = np.cos(w0)
        num = (1 + alpha * a) + (-2 * cos) * z1 + (1 - alpha * a) * z2
        den = (1 + alpha / a) + (-2 * cos) * z1 + (1 - alpha / a) * z2
        out *= np.abs(num / den)
    return out


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
        self.eq_curve = None  # None のときはEQの計算を丸ごと飛ばす
        self.freqs = freqs

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
        eq_curve = self.eq_curve
        if eq_curve is not None:
            spec = spec * eq_curve
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
        self.eq_settings = None

    def apply_eq(self, eq):
        self.eq_settings = eq
        active = eq["on"] and any(float(b["g"]) != 0.0 for b in eq["bands"])
        self.voc.eq_curve = eq_response(self.rate, self.voc.freqs, eq["bands"]).astype(np.float32) if active else None

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
            if self.eq_settings is not None:
                self.apply_eq(self.eq_settings)
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
BG = "#141517"
PANEL = "#1f2023"
RAISED = "#26282c"
RAISED_HI = "#33353a"
LINE = "#3b3d42"
TRACK = "#35373c"
TEXT = "#f1efe9"
MUTED = "#b5b3ad"
ACCENT = "#f2a93b"
ON_ACCENT = "#211603"
TEAL = "#55c2b0"
RED = "#e86a58"
YELLOW = "#f2d13c"
LCD = "#0d0f0e"
KEY_WHITE = "#ece8df"
KEY_WHITE_TEXT = "#4c4d51"
KEY_BLACK = "#1d1f22"
SEG_OFF = "#2d2f33"

EQ_F_MIN, EQ_F_MAX = 20.0, 20000.0
EQ_G_MAX = 18.0
EQ_Q_MIN, EQ_Q_MAX = 0.3, 9.0
BAND_NAMES = ("LOW", "LOW MID", "HIGH MID", "HIGH")


def round_rect(canvas, x0, y0, x1, y1, r, **kw):
    r = max(0.0, min(r, (x1 - x0) / 2, (y1 - y0) / 2))
    pts = (x0 + r, y0, x1 - r, y0, x1, y0, x1, y0 + r, x1, y1 - r, x1, y1,
           x1 - r, y1, x0 + r, y1, x0, y1, x0, y1 - r, x0, y0 + r, x0, y0)
    return canvas.create_polygon(pts, smooth=True, splinesteps=12, **kw)


class RoundPanel(tk.Canvas):
    """角丸のパネル。中身は .inner に置く。"""

    def __init__(self, app, parent, fill=PANEL, pad=12, radius=16, parent_bg=BG, min_height=0):
        super().__init__(parent, bg=parent_bg, highlightthickness=0, takefocus=0)
        self.pad = app.px(pad)
        self.radius = app.px(radius)
        self.fill = fill
        self.min_height = app.px(min_height)
        self.inner = tk.Frame(self, bg=fill)
        self.window = self.create_window(self.pad, self.pad, anchor="nw", window=self.inner)
        self.bind("<Configure>", self._layout)
        self.inner.bind("<Configure>", self._fit)

    def _fit(self, _event=None):
        need = max(self.inner.winfo_reqheight() + 2 * self.pad, self.min_height)
        if int(self.cget("height")) != need:
            self.configure(height=need)

    def _layout(self, _event=None):
        w, h = self.winfo_width(), self.winfo_height()
        self.delete("bg")
        round_rect(self, 0, 0, w - 1, h - 1, self.radius, fill=self.fill, outline="", tags="bg")
        self.tag_lower("bg")
        self.itemconfigure(self.window, width=max(1, w - 2 * self.pad), height=max(1, h - 2 * self.pad))


class RoundButton(tk.Canvas):
    def __init__(self, app, parent, text="", sub="", on_press=None, on_release=None, height=44, radius=22,
                 font=None, sub_font=None, width=None, parent_bg=BG, fill=RAISED, outline=LINE):
        super().__init__(parent, height=app.px(height), bg=parent_bg, highlightthickness=0, takefocus=0, cursor="hand2")
        if width:
            self.configure(width=app.px(width))
        self.app = app
        self.text = text
        self.sub = sub
        self.radius = app.px(radius)
        self.font = font or app.f_small
        self.sub_font = sub_font or app.f_mono_small
        self.fill = fill
        self.outline = outline
        self.fg = TEXT
        self.sub_fg = MUTED
        self.on_press = on_press
        self.on_release = on_release
        self.bind("<Configure>", lambda _e: self.redraw())
        self.bind("<ButtonPress-1>", self._press)
        self.bind("<ButtonRelease-1>", self._release)

    def _press(self, _event):
        if self.on_press:
            self.on_press()

    def _release(self, _event):
        if self.on_release:
            self.on_release()

    def set_style(self, fill=None, outline=None, fg=None, sub_fg=None, text=None, sub=None):
        changed = False
        for name, value in (("fill", fill), ("outline", outline), ("fg", fg), ("sub_fg", sub_fg), ("text", text), ("sub", sub)):
            if value is not None and getattr(self, name) != value:
                setattr(self, name, value)
                changed = True
        if changed:
            self.redraw()

    def redraw(self):
        self.delete("all")
        w, h = self.winfo_width(), self.winfo_height()
        if w < 4 or h < 4:
            return
        round_rect(self, 1, 1, w - 1, h - 1, self.radius, fill=self.fill, outline=self.outline)
        if self.sub:
            self.create_text(w / 2, h * 0.40, text=self.text, fill=self.fg, font=self.font)
            self.create_text(w / 2, h * 0.72, text=self.sub, fill=self.sub_fg, font=self.sub_font)
        else:
            self.create_text(w / 2, h / 2, text=self.text, fill=self.fg, font=self.font)


class ChordPad(tk.Canvas):
    def __init__(self, app, parent, index):
        super().__init__(parent, height=app.px(66), bg=BG, highlightthickness=0, takefocus=0, cursor="hand2")
        self.app = app
        self.index = index
        self.active = False
        self.bind("<Configure>", lambda _e: self.redraw())
        self.bind("<ButtonPress-1>", lambda _e: app.chord_on(index, ("chordmouse",)))
        self.bind("<ButtonRelease-1>", lambda _e: app.note_off(("chordmouse",)))
        self.bind("<Button-3>", lambda _e: app.register_chord(index))

    def set_active(self, active):
        if active != self.active:
            self.active = active
            self.redraw()

    def redraw(self):
        self.delete("all")
        w, h = self.winfo_width(), self.winfo_height()
        if w < 4:
            return
        app = self.app
        fill = ACCENT if self.active else RAISED
        fg = ON_ACCENT if self.active else TEXT
        sub = ON_ACCENT if self.active else MUTED
        round_rect(self, 1, 1, w - 1, h - 1, app.px(14), fill=fill, outline=ACCENT if self.active else LINE)
        cx, cy, r = app.px(24), h / 2, app.px(13)
        self.create_oval(cx - r, cy - r, cx + r, cy + r, fill="#17181a", outline="")
        self.create_text(cx, cy, text=str(self.index + 1), fill="#d8d6cf", font=app.f_mono_small)
        name, notes = app.chord_text(self.index)
        x = cx + r + app.px(9)
        self.create_text(x, h * 0.36, text=name, anchor="w", fill=fg, font=app.f_mid)
        self.create_text(x, h * 0.68, text=notes, anchor="w", fill=sub, font=app.f_tiny)


class Knob(tk.Canvas):
    """上下ドラッグかホイールで回す。ダブルクリックで初期値に戻る。"""

    def __init__(self, app, parent, label, get, set_, fmt, default, value_color=ACCENT, arc_color=ACCENT, parent_bg=PANEL):
        super().__init__(parent, width=app.px(82), height=app.px(104), bg=parent_bg, highlightthickness=0, takefocus=0, cursor="sb_v_double_arrow")
        self.app = app
        self.label = label
        self.get = get
        self.set_ = set_
        self.fmt = fmt
        self.default = default
        self.value_color = value_color
        self.arc_color = arc_color
        self.last_y = 0
        self.bind("<ButtonPress-1>", self._press)
        self.bind("<B1-Motion>", self._drag)
        self.bind("<Double-Button-1>", lambda _e: self._apply(self.default))
        self.bind("<MouseWheel>", lambda e: self._apply(self.get() + (0.02 if e.delta > 0 else -0.02)))
        self.bind("<Button-4>", lambda _e: self._apply(self.get() + 0.02))
        self.bind("<Button-5>", lambda _e: self._apply(self.get() - 0.02))
        self.redraw()

    def _press(self, event):
        self.last_y = event.y

    def _drag(self, event):
        delta = (self.last_y - event.y) / self.app.px(180)
        self.last_y = event.y
        self._apply(self.get() + delta)

    def _apply(self, value):
        if self.app.settings_open:
            return
        self.set_(min(1.0, max(0.0, value)))
        self.redraw()

    def redraw(self):
        self.delete("all")
        app = self.app
        v = self.get()
        size = app.px(56)
        pad = app.px(6)
        cx = (app.px(82)) / 2
        x0, y0 = cx - size / 2, pad
        x1, y1 = x0 + size, y0 + size
        width = app.px(5)
        self.create_arc(x0, y0, x1, y1, start=225, extent=-270, style="arc", outline=TRACK, width=width)
        if v > 0.003:
            self.create_arc(x0, y0, x1, y1, start=225, extent=-270 * v, style="arc", outline=self.arc_color, width=width)
        cy = y0 + size / 2
        r = app.px(19)
        self.create_oval(cx - r, cy - r, cx + r, cy + r, fill="#2d2f33", outline="")
        angle = np.radians(225 - 270 * v)
        self.create_line(cx + np.cos(angle) * r * 0.25, cy - np.sin(angle) * r * 0.25,
                         cx + np.cos(angle) * r * 0.82, cy - np.sin(angle) * r * 0.82,
                         fill=TEXT, width=app.px(3), capstyle="round")
        self.create_text(cx, y1 + app.px(12), text=self.label, fill=TEXT, font=app.f_tiny)
        self.create_text(cx, y1 + app.px(28), text=self.fmt(), fill=self.value_color, font=app.f_mono_tiny)


class SegmentMeter(tk.Canvas):
    COUNT = 24

    def __init__(self, app, parent):
        super().__init__(parent, height=app.px(14), bg=PANEL, highlightthickness=0, takefocus=0)
        self.app = app
        self.value = 0.0
        self.items = []
        self.lit = -1
        self.bind("<Configure>", lambda _e: self.build())

    def build(self):
        self.delete("all")
        self.items = []
        w, h = self.winfo_width(), self.winfo_height()
        gap = self.app.px(3)
        seg = (w - gap * (self.COUNT - 1)) / self.COUNT
        for i in range(self.COUNT):
            x = i * (seg + gap)
            self.items.append(round_rect(self, x, 0, x + seg, h, h / 2, fill=SEG_OFF, outline=""))
        self.lit = -1

    def set(self, peak):
        db = 20 * np.log10(peak) if peak > 1e-6 else -120.0
        target = min(1.0, max(0.0, (db + 60) / 60))
        self.value = target if target > self.value else max(target, self.value - 0.03)
        lit = int(round(self.value * self.COUNT))
        if lit == self.lit or not self.items:
            return
        for i, item in enumerate(self.items):
            zone = TEAL if i < 16 else ACCENT if i < 21 else RED
            self.itemconfigure(item, fill=zone if i < lit else SEG_OFF)
        self.lit = lit


class EqGraph(tk.Canvas):
    def __init__(self, app, parent):
        super().__init__(parent, bg=LCD, highlightthickness=0, takefocus=0, height=app.px(150), width=app.px(300))
        self.app = app
        self.dragging = False
        self.bind("<Configure>", lambda _e: self.redraw())
        self.bind("<ButtonPress-1>", self._press)
        self.bind("<B1-Motion>", self._drag)
        self.bind("<ButtonRelease-1>", lambda _e: setattr(self, "dragging", False))

    def x_of(self, f):
        return np.log10(f / EQ_F_MIN) / np.log10(EQ_F_MAX / EQ_F_MIN) * self.winfo_width()

    def y_of(self, g):
        h = self.winfo_height()
        return h / 2 - g / EQ_G_MAX * (h / 2 - self.app.px(8))

    def _press(self, event):
        bands = self.app.cfg["eq"]["bands"]
        best = min(range(len(bands)), key=lambda i: (self.x_of(bands[i]["f"]) - event.x) ** 2 + (self.y_of(bands[i]["g"]) - event.y) ** 2)
        self.app.select_band(best)
        self.dragging = True

    def _drag(self, event):
        if not self.dragging:
            return
        w, h = self.winfo_width(), self.winfo_height()
        f = EQ_F_MIN * (EQ_F_MAX / EQ_F_MIN) ** (min(w, max(0, event.x)) / max(1, w))
        g = (h / 2 - event.y) / max(1, (h / 2 - self.app.px(8))) * EQ_G_MAX
        self.app.set_band(f=f, g=min(EQ_G_MAX, max(-EQ_G_MAX, g)))

    def redraw(self):
        self.delete("all")
        w, h = self.winfo_width(), self.winfo_height()
        if w < 10:
            return
        app = self.app
        for f, label in ((100, "100"), (1000, "1k"), (10000, "10k")):
            x = self.x_of(f)
            self.create_line(x, 0, x, h, fill=RAISED)
            self.create_text(x + app.px(4), h - app.px(4), text=label, anchor="sw", fill="#a9a8a2", font=app.f_mono_tiny)
        self.create_line(0, h / 2, w, h / 2, fill="#45474c")
        eq = app.cfg["eq"]
        xs = np.arange(0, w + 3, 3)
        freqs = EQ_F_MIN * (EQ_F_MAX / EQ_F_MIN) ** (xs / max(1, w))
        curve = eq_response(app.engine.rate, freqs, eq["bands"])
        db = 20 * np.log10(np.maximum(curve, 1e-4)) if eq["on"] else np.zeros_like(xs, dtype=np.float64)
        ys = h / 2 - np.clip(db, -EQ_G_MAX, EQ_G_MAX) / EQ_G_MAX * (h / 2 - app.px(8))
        coords = np.empty(len(xs) * 2)
        coords[0::2] = xs
        coords[1::2] = ys
        self.create_line(coords.tolist(), fill=ACCENT if eq["on"] else "#6d6c67", width=app.px(2))
        for i, band in enumerate(eq["bands"]):
            x, y = self.x_of(band["f"]), self.y_of(band["g"])
            selected = i == app.selected_band
            if selected:
                r = app.px(10)
                self.create_oval(x - r, y - r, x + r, y + r, outline=TEXT)
            r = app.px(5)
            self.create_oval(x - r, y - r, x + r, y + r, fill=ACCENT if selected else "#d8d6cf", outline="")
            label_y = y + app.px(17) if band["g"] >= 0 else y - app.px(17)
            self.create_text(x, label_y, text=str(i + 1), fill=TEXT, font=app.f_mono_tiny)


class Piano(tk.Canvas):
    WHITE = (0, 2, 4, 5, 7, 9, 11)
    SPAN = 29

    def __init__(self, app, parent):
        super().__init__(parent, bg="#0d0e10", highlightthickness=0, takefocus=0)
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
                rects[n] = (x - w * 0.31, 0, x + w * 0.31, h * 0.60, False)
        return rects

    def redraw(self):
        self.delete("all")
        self.rects = self.layout()
        app = self.app
        sounding = app.live_notes() | app.latched
        r_white, r_black = app.px(11), app.px(8)
        for n, (x0, y0, x1, y1, is_white) in self.rects.items():
            if not is_white:
                continue
            on = n in sounding
            round_rect(self, x0 + 1.5, y0 - r_white, x1 - 1.5, y1, r_white, fill=ACCENT if on else KEY_WHITE, outline="")
            fg = ON_ACCENT if on else KEY_WHITE_TEXT
            if n % 12 == 0:
                self.create_text((x0 + x1) / 2, y1 - app.px(13), text=note_name(n), fill=fg, font=app.f_tiny)
            letter = app.letter_for_note(n)
            if letter:
                self.create_text((x0 + x1) / 2, y1 - app.px(32), text=letter.upper(), fill=fg, font=app.f_mono_small)
        for n, (x0, y0, x1, y1, is_white) in self.rects.items():
            if is_white:
                continue
            on = n in sounding
            round_rect(self, x0, y0 - r_black, x1, y1, r_black, fill=ACCENT if on else KEY_BLACK, outline="#000000")
            letter = app.letter_for_note(n)
            if letter:
                self.create_text((x0 + x1) / 2, y1 - app.px(14), text=letter.upper(), fill=ON_ACCENT if on else "#d8d6cf", font=app.f_mono_small)

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
        if self.app.hold:  # HOLD中のドラッグで音が増減しないようにする
            return
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
        self.base_note = 48
        self.held = {}
        self.latched = set()
        self.hold = False
        self.down = set()
        self.engine_error = ""
        self.tick_count = 0
        self.settings_open = False
        self.dirty = True
        self.selected_band = 2
        self.status_hold = 0.0
        self.scale = max(1.0, root.winfo_fpixels("1i") / 96)

        families = set(tkfont.families())
        family = next((f for f in ("Yu Gothic UI", "Meiryo UI", "Noto Sans CJK JP") if f in families), "TkDefaultFont")
        mono = next((f for f in ("Consolas", "DejaVu Sans Mono") if f in families), family)
        self.f_title = (family, -self.px(20), "bold")
        self.f_lcd = (mono, -self.px(34), "bold")
        self.f_lcd_small = (mono, -self.px(13))
        self.f_button = (family, -self.px(19), "bold")
        self.f_mid = (family, -self.px(15), "bold")
        self.f_small = (family, -self.px(13))
        self.f_tiny = (family, -self.px(11))
        self.f_mono_small = (mono, -self.px(12))
        self.f_mono_tiny = (mono, -self.px(11))

        root.title(f"VOCODER {VERSION}")
        root.configure(bg=BG)
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        root.geometry(f"{min(self.px(1200), sw - 40)}x{min(self.px(900), sh - 90)}")
        root.minsize(self.px(960), min(self.px(680), sh - 90))
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.build_style()
        self.main_view = tk.Frame(root, bg=BG)
        self.build_main(self.main_view)
        self.main_view.pack(fill="both", expand=True, padx=self.px(16), pady=self.px(12))
        self.settings_view = None
        root.bind("<KeyPress>", self.on_key)
        root.bind("<KeyRelease>", self.on_key_release)
        root.bind("<FocusOut>", self.on_focus_out)
        self.engine.apply_eq(self.cfg["eq"])
        self.start_audio(first_run=True)
        root.focus_force()
        self.tick()

    def px(self, n):
        return int(round(n * getattr(self, "scale", 1.0)))

    def build_style(self):
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("Dark.TCombobox", fieldbackground=BG, background=RAISED, foreground=TEXT, arrowcolor=TEXT,
                        bordercolor=LINE, lightcolor=LINE, darkcolor=LINE, padding=self.px(8))
        style.map("Dark.TCombobox", fieldbackground=[("readonly", BG)], foreground=[("readonly", TEXT)],
                  selectbackground=[("readonly", BG)], selectforeground=[("readonly", TEXT)])
        self.root.option_add("*TCombobox*Listbox.background", RAISED)
        self.root.option_add("*TCombobox*Listbox.foreground", TEXT)
        self.root.option_add("*TCombobox*Listbox.selectBackground", ACCENT)
        self.root.option_add("*TCombobox*Listbox.selectForeground", ON_ACCENT)

    # ---- メイン画面 ----
    def build_main(self, parent):
        px = self.px
        header = tk.Frame(parent, bg=BG)
        header.pack(fill="x", pady=(0, px(10)))
        logo = tk.Canvas(header, width=px(30), height=px(30), bg=BG, highlightthickness=0)
        logo.create_oval(1, 1, px(29), px(29), fill=RAISED, outline="")
        for x, half in ((11, 4), (15, 8), (19, 4), (23, 1)):
            logo.create_line(px(x), px(15 - half), px(x), px(15 + half), fill=ACCENT, width=px(2), capstyle="round")
        logo.pack(side="left", padx=(0, px(10)))
        tk.Label(header, text="VOCODER", font=self.f_title, fg=TEXT, bg=BG).pack(side="left")
        self.not_active = tk.Canvas(header, width=px(102), height=px(26), bg=BG, highlightthickness=0)
        round_rect(self.not_active, 1, 1, px(101), px(25), px(13), fill=BG, outline=YELLOW)
        self.not_active.create_text(px(51), px(13), text="NOT ACTIVE", fill=YELLOW, font=self.f_mono_small)
        self.not_active_shown = False

        self.settings_button = RoundButton(self, header, text="設定", on_release=self.open_settings, width=88, font=self.f_small)
        self.settings_button.pack(side="right")
        self.reconnect_button = RoundButton(self, header, text="RECONNECT", on_release=self.reconnect, width=112, font=self.f_small)
        self.reconnect_shown = False
        self.latency_label = tk.Label(header, text="", font=self.f_mono_small, fg=MUTED, bg=BG)
        self.latency_label.pack(side="right", padx=px(16))
        self.device_label = tk.Label(header, text="", font=self.f_small, fg=MUTED, bg=BG)
        self.device_label.pack(side="right")
        self.led = tk.Canvas(header, width=px(10), height=px(10), bg=BG, highlightthickness=0)
        self.led_item = self.led.create_oval(0, 0, px(9), px(9), fill=TEAL, outline="")
        self.led.pack(side="right", padx=(0, px(8)))

        top = tk.Frame(parent, bg=BG)
        top.pack(fill="x", pady=(0, px(10)))
        top.columnconfigure(0, weight=3, uniform="top")
        top.columnconfigure(1, weight=1, uniform="top")
        top.columnconfigure(2, weight=1, uniform="top")
        self.lcd = tk.Canvas(top, height=px(88), bg=BG, highlightthickness=0)
        self.lcd.grid(row=0, column=0, sticky="ew", padx=(0, px(12)))
        self.lcd.bind("<Configure>", lambda _e: self.draw_lcd())
        self.hold_button = RoundButton(self, top, text="HOLD", sub="Space", on_press=self.toggle_hold, height=88, radius=16, font=self.f_button)
        self.hold_button.grid(row=0, column=1, sticky="ew", padx=(0, px(12)))
        self.off_button = RoundButton(self, top, text="ALL OFF", sub="Esc", on_press=self.all_off, height=88, radius=16, font=self.f_button)
        self.off_button.grid(row=0, column=2, sticky="ew")

        chords = tk.Frame(parent, bg=BG)
        chords.pack(fill="x", pady=(0, px(8)))
        self.chord_pads = []
        for i in range(8):
            chords.columnconfigure(i, weight=1, uniform="chord")
            pad = ChordPad(self, chords, i)
            pad.grid(row=0, column=i, sticky="ew", padx=(0 if i == 0 else px(5), 0))
            self.chord_pads.append(pad)

        bottom = tk.Frame(parent, bg=BG)
        bottom.pack(side="bottom", fill="x", pady=(px(10), 0))
        bottom.columnconfigure(0, weight=2, uniform="bottom")
        bottom.columnconfigure(1, weight=3, uniform="bottom")
        left = tk.Frame(bottom, bg=BG)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, px(12)))
        self.build_mix(left)
        eq = RoundPanel(self, bottom, pad=12)
        eq.grid(row=0, column=1, sticky="nsew")
        self.build_eq(eq.inner)

        octave = tk.Frame(parent, bg=BG)
        octave.pack(fill="x", pady=(0, px(6)))
        RoundButton(self, octave, text="OCT ＋  X", on_press=lambda: self.shift_octave(1), width=104, font=self.f_small).pack(side="right")
        RoundButton(self, octave, text="OCT −  Z", on_press=lambda: self.shift_octave(-1), width=104, font=self.f_small).pack(side="right", padx=px(8))
        self.octave_label = tk.Label(octave, text="", font=self.f_mono_small, fg=MUTED, bg=BG)
        self.octave_label.pack(side="right", padx=px(4))
        self.status = tk.Label(octave, text="", font=self.f_small, fg=MUTED, bg=BG, anchor="w")
        self.status.pack(side="left", fill="x", expand=True)

        self.piano_frame = RoundPanel(self, parent, fill="#0d0e10", pad=9, min_height=170)
        self.piano_frame.pack(fill="both", expand=True)
        self.piano = Piano(self, self.piano_frame.inner)
        self.piano.configure(height=px(150))
        self.piano.pack(fill="both", expand=True)

    def mix_knob(self, parent, label, key, lo, hi, fmt, color):
        def get():
            return (float(self.cfg[key]) - lo) / (hi - lo)

        def set_(v):
            self.cfg[key] = round(lo + v * (hi - lo), 3)
            self.engine.apply_mix(self.cfg)

        default = (float(DEFAULT_CONFIG[key]) - lo) / (hi - lo)
        return Knob(self, parent, label, get, set_, lambda: fmt(float(self.cfg[key])), default, arc_color=color, value_color=color)

    def build_mix(self, parent):
        px = self.px
        knobs_panel = RoundPanel(self, parent, pad=8)
        knobs_panel.pack(fill="both", expand=True)
        knobs = knobs_panel.inner
        knobs.rowconfigure(0, weight=1)
        db = lambda v: f"{v:+.1f} dB"
        pct = lambda v: f"{v * 100:.0f}%"
        self.mix_knobs = [
            self.mix_knob(knobs, "INPUT", "input_gain_db", -24, 24, db, TEAL),
            self.mix_knob(knobs, "OUTPUT", "output_gain_db", -24, 12, db, TEAL),
            self.mix_knob(knobs, "SIBILANCE", "consonant", 0, 1, pct, ACCENT),
            self.mix_knob(knobs, "DRY", "dry", 0, 1, pct, ACCENT),
        ]
        for i, k in enumerate(self.mix_knobs):
            knobs.columnconfigure(i, weight=1)
            k.grid(row=0, column=i)
        meters_panel = RoundPanel(self, parent, pad=12)
        meters_panel.pack(fill="x", pady=(px(10), 0))
        meters = meters_panel.inner
        meters.columnconfigure(1, weight=1)
        self.meter_in = SegmentMeter(self, meters)
        self.meter_out = SegmentMeter(self, meters)
        for i, (text, m) in enumerate((("MIC", self.meter_in), ("OUTPUT", self.meter_out))):
            tk.Label(meters, text=text, font=self.f_mono_small, fg=MUTED, bg=PANEL).grid(row=i, column=0, sticky="w", padx=(0, px(10)), pady=px(5))
            m.grid(row=i, column=1, sticky="ew")

    def build_eq(self, parent):
        px = self.px
        head = tk.Frame(parent, bg=PANEL)
        head.pack(fill="x", pady=(0, px(6)))
        tk.Label(head, text="EQ", font=self.f_mid, fg=TEXT, bg=PANEL).pack(side="left")
        self.eq_button = RoundButton(self, head, text="ON", on_press=self.toggle_eq, width=64, height=34, radius=17, parent_bg=PANEL, fill=PANEL, outline=TEAL, font=self.f_small)
        self.eq_button.pack(side="right")
        middle = tk.Frame(parent, bg=PANEL)
        middle.pack(fill="both", expand=True)
        self.eq_graph = EqGraph(self, middle)
        self.eq_graph.pack(side="left", fill="both", expand=True, padx=(0, px(8)))

        def fget():
            return np.log10(self.band()["f"] / EQ_F_MIN) / np.log10(EQ_F_MAX / EQ_F_MIN)

        def gget():
            return (self.band()["g"] + EQ_G_MAX) / (2 * EQ_G_MAX)

        def qget():
            return np.log10(self.band()["q"] / EQ_Q_MIN) / np.log10(EQ_Q_MAX / EQ_Q_MIN)

        self.eq_knobs = [
            Knob(self, middle, "FREQ", fget, lambda v: self.set_band(f=EQ_F_MIN * (EQ_F_MAX / EQ_F_MIN) ** v), lambda: fmt_freq(self.band()["f"]), fget()),
            Knob(self, middle, "GAIN", gget, lambda v: self.set_band(g=v * 2 * EQ_G_MAX - EQ_G_MAX), lambda: fmt_gain(self.band()["g"]), 0.5),
            Knob(self, middle, "Q", qget, lambda v: self.set_band(q=EQ_Q_MIN * (EQ_Q_MAX / EQ_Q_MIN) ** v), lambda: f"{self.band()['q']:.1f}", qget()),
        ]
        for k in self.eq_knobs:
            k.pack(side="left")
        chips = tk.Frame(parent, bg=PANEL)
        chips.pack(fill="x", pady=(px(6), 0))
        self.band_chips = []
        for i in range(4):
            chips.columnconfigure(i, weight=1, uniform="chip")
            chip = RoundButton(self, chips, text=f"{i + 1}  {BAND_NAMES[i]}", sub="", on_press=lambda k=i: self.select_band(k),
                               height=52, radius=12, parent_bg=PANEL, font=self.f_small, sub_font=self.f_mono_tiny)
            chip.grid(row=0, column=i, sticky="ew", padx=(0 if i == 0 else px(5), 0))
            self.band_chips.append(chip)
        self.refresh_eq()

    # ---- EQ ----
    def band(self):
        return self.cfg["eq"]["bands"][self.selected_band]

    def select_band(self, index):
        self.selected_band = index
        self.refresh_eq()
        self.root.focus_set()

    def set_band(self, f=None, g=None, q=None):
        band = self.band()
        if f is not None:
            band["f"] = float(min(EQ_F_MAX, max(EQ_F_MIN, f)))
        if g is not None:
            band["g"] = 0.0 if abs(g) < 0.15 else float(g)  # 0付近は吸着させて、処理も止まるようにする
        if q is not None:
            band["q"] = float(min(EQ_Q_MAX, max(EQ_Q_MIN, q)))
        self.engine.apply_eq(self.cfg["eq"])
        self.refresh_eq()

    def toggle_eq(self):
        self.cfg["eq"]["on"] = not self.cfg["eq"]["on"]
        self.engine.apply_eq(self.cfg["eq"])
        self.refresh_eq()
        self.root.focus_set()

    def refresh_eq(self):
        eq = self.cfg["eq"]
        self.eq_button.set_style(text="ON" if eq["on"] else "OFF", outline=TEAL if eq["on"] else LINE, fg="#8fe0d2" if eq["on"] else MUTED)
        for i, chip in enumerate(self.band_chips):
            band = eq["bands"][i]
            selected = i == self.selected_band
            chip.set_style(fill="#33291a" if selected else RAISED, outline=ACCENT if selected else LINE,
                           sub=f"{fmt_freq(band['f'])} {fmt_gain(band['g'])} Q{band['q']:.1f}", sub_fg="#cfcdc7")
        for k in self.eq_knobs:
            k.redraw()
        self.eq_graph.redraw()

    # ---- オーディオ ----
    def start_audio(self, first_run=False):
        if not self.cfg["input_device"] or not self.cfg["output_device"]:
            self.engine_error = "SELECT A デバイス IN 設定"
            if first_run:
                self.root.after(200, self.open_settings)
            return
        try:
            self.engine.open(self.cfg)
            self.engine_error = ""
        except Exception as e:
            log_exception()
            self.engine_error = f"CANNOT OPEN AUDIO: {e}"

    def reconnect(self):
        self.engine.close()
        rescan_devices()
        self.start_audio()
        self.root.focus_set()

    def open_settings(self):
        if self.settings_open:
            return
        self.all_off()
        self.engine.close()
        rescan_devices()
        self.settings_open = True
        self.main_view.pack_forget()
        self.settings_view = SettingsView(self)
        self.settings_view.pack(fill="both", expand=True, padx=self.px(16), pady=self.px(12))
        self.root.focus_set()

    def close_settings(self, cfg=None):
        self.settings_open = False
        self.settings_view.destroy()
        self.settings_view = None
        self.main_view.pack(fill="both", expand=True, padx=self.px(16), pady=self.px(12))
        if cfg is not None:
            self.cfg = cfg
            save_config(cfg)
        self.start_audio()
        self.root.focus_force()
        self.dirty = True

    # ---- 音の管理 ----
    def live_notes(self):
        out = set()
        for notes in self.held.values():
            out |= notes
        return out

    def update_notes(self):
        self.engine.set_notes(self.live_notes() | self.latched)
        self.dirty = True

    def note_on(self, source, notes, chord=False):
        notes = {notes} if isinstance(notes, int) else set(notes)
        if self.hold:
            # HOLD中：鍵盤は押すたびに足す／もう一度で外す。コードは丸ごと置き換える
            if chord:
                self.latched = notes
            else:
                self.latched = self.latched ^ notes
        else:
            self.held[source] = notes
        self.update_notes()

    def note_off(self, source):
        if self.held.pop(source, None) is not None:
            self.update_notes()

    def toggle_hold(self):
        if self.hold:
            self.hold = False
            self.latched = set()
        else:
            self.hold = True
            self.latched = self.live_notes()
        self.update_notes()
        self.root.focus_set()

    def all_off(self):
        self.held.clear()
        self.latched = set()
        self.hold = False
        self.update_notes()

    def chord_on(self, index, source):
        chords = self.cfg["chords"]
        if index < len(chords) and chords[index]:
            self.note_on(source, chords[index], chord=True)

    def chord_text(self, index):
        notes = self.cfg["chords"][index] if index < len(self.cfg["chords"]) else []
        default = DEFAULT_CHORDS[index] if index < len(DEFAULT_CHORDS) else None
        name = CHORD_NAMES[index] if notes == default else "USER"
        return name, chord_label(notes)

    def register_chord(self, index):
        notes = sorted(self.live_notes() | self.latched)
        if notes:
            self.cfg["chords"][index] = notes
            save_config(self.cfg)
            self.chord_pads[index].redraw()
            self.show_status(f"CHORD {index + 1} = {chord_label(notes)}")
        else:
            self.show_status("PLAY NOTES, THEN RIGHT-CLICK A CHORD")

    def shift_octave(self, delta):
        new = self.base_note + delta * 12
        if 24 <= new <= 72:
            self.base_note = new
            self.dirty = True
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
        ks = event.keysym
        if self.settings_open:
            if ks == "Escape" and self.settings_view is not None:
                self.settings_view.back()
            return
        code = event.keycode
        if code in self.down:
            return
        self.down.add(code)
        if ks == "space":
            self.toggle_hold()
        elif ks == "Escape":
            self.all_off()
        elif len(ks) == 1 and ks in "12345678":
            self.chord_on(int(ks) - 1, ("key", code))
        else:
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
        if event.widget is self.root:
            self.down.clear()
            self.held.clear()
            self.update_notes()

    # ---- 更新 ----
    def show_status(self, text):
        self.status.configure(text=text, fg=ACCENT)
        self.status_hold = time.monotonic() + 3

    def draw_lcd(self):
        c = self.lcd
        c.delete("all")
        w, h = c.winfo_width(), c.winfo_height()
        if w < 10:
            return
        round_rect(c, 1, 1, w - 1, h - 1, self.px(16), fill=LCD, outline="#33352f")
        sounding = sorted(self.live_notes() | self.latched)
        head = "HOLD" if self.hold else "NOTES"
        x = self.px(22)
        c.create_text(x, h * 0.28, text=head, anchor="w", fill=ACCENT if (self.hold or sounding) else "#7d7c76", font=self.f_lcd_small)
        text = "  ".join(note_name(n) for n in sounding[:7]) if sounding else "—"
        c.create_text(x, h * 0.65, text=text, anchor="w", fill=ACCENT if sounding else "#5d5c57", font=self.f_lcd)

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
        if self.settings_open:
            return
        if self.dirty:
            self.dirty = False
            self.piano.redraw()
            self.draw_lcd()
            sounding = self.live_notes() | self.latched
            for i, pad in enumerate(self.chord_pads):
                notes = self.cfg["chords"][i] if i < len(self.cfg["chords"]) else []
                pad.set_active(bool(notes) and set(notes) == sounding)
            if self.hold:
                self.hold_button.set_style(fill=ACCENT, outline=ACCENT, fg=ON_ACCENT, sub_fg=ON_ACCENT)
            else:
                self.hold_button.set_style(fill=RAISED, outline=LINE, fg=TEXT, sub_fg=MUTED)
            self.octave_label.configure(text=f"KEYS {note_name(self.base_note)}–")
        focused = self.root_has_focus()
        if focused == self.not_active_shown:
            self.not_active_shown = not focused
            if self.not_active_shown:
                self.not_active.pack(side="left", padx=self.px(12))
            else:
                self.not_active.pack_forget()
        if self.tick_count % 12 == 0:
            ok = e.healthy()
            self.led.itemconfigure(self.led_item, fill=TEAL if ok else RED)
            if ok:
                self.device_label.configure(text=short_device(self.cfg["output_device"]))
                self.latency_label.configure(text=f"LATENCY {e.latency_estimate_ms():.0f} ms")
            else:
                self.device_label.configure(text="NOT CONNECTED")
                self.latency_label.configure(text="")
            if ok == self.reconnect_shown:
                self.reconnect_shown = not ok
                if self.reconnect_shown:
                    self.reconnect_button.pack(side="right", padx=(0, self.px(8)), before=self.settings_button)
                else:
                    self.reconnect_button.pack_forget()
            if time.monotonic() >= self.status_hold:
                problems = e.problems + e.underruns
                if not ok:
                    self.status.configure(text=self.engine_error or "AUDIO STOPPED", fg=YELLOW)
                elif problems:
                    self.status.configure(text=f"DROPOUTS {problems}", fg=YELLOW)
                else:
                    self.status.configure(text="", fg=MUTED)

    def root_has_focus(self):
        return bool(str(self.root.tk.call("focus")))

    def on_close(self):
        save_config(self.cfg)
        self.engine.close()
        self.root.destroy()


def fmt_freq(f):
    return f"{f / 1000:.1f} kHz" if f >= 1000 else f"{f:.0f} Hz"


def fmt_gain(g):
    return f"{g:+.1f} dB"


def short_device(label):
    name = label.split("] ", 1)[-1]
    return name if len(name) <= 34 else name[:33] + "…"


class SettingsView(tk.Frame):
    """ポップアップではなく、同じウィンドウの中に開く設定画面。"""

    def __init__(self, app):
        super().__init__(app.root, bg=BG)
        self.app = app
        px = app.px
        cfg = app.cfg
        self.v_in = tk.StringVar(value=cfg["input_device"])
        self.v_out = tk.StringVar(value=cfg["output_device"])
        self.v_in_ch = tk.StringVar(value=str(cfg["input_channel"]))
        self.v_l = tk.StringVar(value=str(cfg["output_left"]))
        self.v_r = tk.StringVar(value=str(cfg["output_right"]))
        self.v_rate = tk.StringVar(value=str(cfg["samplerate"]))
        self.v_latency = tk.StringVar(value=str(cfg["latency"]).upper())
        self.v_gate = tk.StringVar(value=f"{float(cfg['gate_db']):g}")
        self.load_devices()

        header = tk.Frame(self, bg=BG)
        header.pack(fill="x", pady=(0, px(14)))
        tk.Label(header, text="設定", font=app.f_title, fg=TEXT, bg=BG).pack(side="left")
        RoundButton(app, header, text="BACK  Esc", on_release=self.back, width=112, font=app.f_small).pack(side="right")

        panels = tk.Frame(self, bg=BG)
        panels.pack(fill="both", expand=True)
        panels.columnconfigure((0, 1), weight=1, uniform="panel")
        left_panel = RoundPanel(app, panels, pad=20)
        left_panel.grid(row=0, column=0, sticky="new", padx=(0, px(7)))
        right_panel = RoundPanel(app, panels, pad=20)
        right_panel.grid(row=0, column=1, sticky="new", padx=(px(7), 0))
        left, right = left_panel.inner, right_panel.inner
        tk.Label(left, text="INPUT", font=app.f_mid, fg=TEXT, bg=PANEL).pack(anchor="w")
        tk.Label(right, text="OUTPUT", font=app.f_mid, fg=TEXT, bg=PANEL).pack(anchor="w")
        self.c_in = self.field(left, "デバイス", self.v_in, [d[1] for d in self.ins])
        self.c_in_ch = self.field(left, "CH", self.v_in_ch, [])
        self.field(left, "GATE (dB)", self.v_gate, ["-70", "-65", "-60", "-55", "-50", "-45", "-40"])
        self.c_out = self.field(right, "デバイス", self.v_out, [d[1] for d in self.outs])
        self.c_l = self.field(right, "CH L", self.v_l, [])
        self.c_r = self.field(right, "CH R", self.v_r, [])
        self.field(right, "SAMPLE RATE", self.v_rate, ["44100", "48000", "96000"])
        self.field(right, "LATENCY", self.v_latency, ["LOW", "HIGH"])
        self.c_in.bind("<<ComboboxSelected>>", lambda _e: self.update_channels(), add="+")
        self.c_out.bind("<<ComboboxSelected>>", lambda _e: self.update_channels(), add="+")

        footer = tk.Frame(self, bg=BG)
        footer.pack(fill="x", pady=(px(14), 0))
        RoundButton(app, footer, text="APPLY", on_release=self.apply, width=132, height=48, radius=24, fill=ACCENT, outline=ACCENT, font=app.f_mid).pack(side="right")
        footer.winfo_children()[0].set_style(fg=ON_ACCENT)
        RoundButton(app, footer, text="RESCAN", on_release=self.rescan, width=112, height=48, radius=24, font=app.f_small).pack(side="right", padx=px(10))
        self.update_channels()

    def load_devices(self):
        self.ins = list_devices("in")
        self.outs = list_devices("out")

    def field(self, parent, label, var, values):
        tk.Label(parent, text=label, font=self.app.f_small, fg="#c9c7c1", bg=PANEL).pack(anchor="w", pady=(self.app.px(12), self.app.px(4)))
        box = ttk.Combobox(parent, textvariable=var, values=values, state="readonly", style="Dark.TCombobox", font=self.app.f_small)
        box.pack(fill="x")
        box.bind("<<ComboboxSelected>>", lambda _e: self.focus_set())
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

    def rescan(self):
        rescan_devices()
        self.load_devices()
        self.c_in.configure(values=[d[1] for d in self.ins])
        self.c_out.configure(values=[d[1] for d in self.outs])
        self.update_channels()

    def back(self):
        self.app.close_settings()

    def apply(self):
        cfg = dict(self.app.cfg)
        cfg.update(input_device=self.v_in.get(), output_device=self.v_out.get(),
                   input_channel=int(self.v_in_ch.get()), output_left=int(self.v_l.get()),
                   output_right=int(self.v_r.get()), samplerate=int(self.v_rate.get()),
                   latency=self.v_latency.get().lower(), gate_db=float(self.v_gate.get()))
        self.app.close_settings(cfg)


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
