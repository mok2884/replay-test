#!/usr/bin/env python3
"""リプレイ録音機 — 生放送を録音し続け、言い間違いをすぐ聞き返して本線に出すソフト。

音声の流れ
  入力 → Timeline（メモリ上の録音全体）→ 本線(L) / 検聴(R)
       └→ DiskWriter（WAVへ保存。別スレッドなのでディスクが詰まっても音は止まらない）

  本線はオーディオスレッドが Timeline を直接読むので、Enterからすぐ音が出る。
  検聴は早回しの計算が必要なので、CueRenderer のスレッドが少し先まで作っておく。
"""
import bisect
import ctypes
import datetime
import json
import os
import queue
import shutil
import struct
import subprocess
import sys
import threading
import time
import traceback
from collections import deque
from pathlib import Path

# sounddevice の Windows 版は、この環境変数があると ASIO 対応版の PortAudio を読み込む。
os.environ.setdefault("SD_ENABLE_ASIO", "1")

import numpy as np
import sounddevice as sd
import tkinter as tk
from tkinter import filedialog, font as tkfont, messagebox, ttk

APP_NAME = "リプレイ録音機"
VERSION = "0.1"
APP_DIR = Path(sys.executable if getattr(sys, "frozen", False) else __file__).resolve().parent
CONFIG_PATH = APP_DIR / "replay_config.json"
API_ORDER = ("ASIO", "WASAPI", "WDM-KS", "DirectSound", "MME")

JUMP_SECONDS = 2.0
ENTER_GUARD_SECONDS = 0.5  # 本線に出した直後は止める操作を無視する
HEAD_MARGIN_SECONDS = 0.05  # 録音の先端の手前で再生を止める余裕
STALL_SECONDS = 2.0
MAX_DATA_BYTES = 4_000_000_000
BYTES_PER_SAMPLE = 3

LIVE, SEARCH, ONLINE = "live", "search", "online"

DEFAULT_CONFIG = {
    "input_device": "",
    "output_device": "",
    "samplerate": 48000,
    "input_channel": 1,
    "main_channel": 1,
    "cue_channel": 2,
    "latency": "low",
    "down_offset": 0.0,
    "save_dir": str(APP_DIR / "録音"),
}


def log_exception(*_args):
    try:
        with open(APP_DIR / "error.log", "a", encoding="utf-8") as f:
            f.write(f"--- {datetime.datetime.now():%Y-%m-%d %H:%M:%S}\n{traceback.format_exc()}\n")
    except OSError:
        pass


def fmt_clock(frames, rate):
    total = max(0, frames) / rate
    h = int(total // 3600)
    m = int(total % 3600 // 60)
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:05.2f}"


# ------------------------------------------------------------ 録音データ -------
class Timeline:
    """録音全体をメモリに持つ。書き込みはオーディオスレッドのみ。"""

    def __init__(self, rate):
        self.rate = rate
        self.chunk = rate * 10
        self.chunks = []
        self.head = 0
        self.seams = []
        self.reserve()

    def reserve(self, spare=2):
        """オーディオスレッドで確保せずに済むよう、画面スレッドから先に確保しておく。"""
        need = self.head // self.chunk + 1 + spare
        while len(self.chunks) < need:
            a = np.empty(self.chunk, dtype=np.float32)
            a.fill(0)
            self.chunks.append(a)

    def write(self, x):
        h = self.head
        i, n = 0, len(x)
        while i < n:
            c, o = divmod(h + i, self.chunk)
            if c >= len(self.chunks):
                self.chunks.append(np.zeros(self.chunk, dtype=np.float32))
            k = min(n - i, self.chunk - o)
            self.chunks[c][o:o + k] = x[i:i + k]
            i += k
        self.head = h + n

    def read(self, start, n):
        start = max(0, int(start))
        end = min(start + int(n), self.head)
        if end <= start:
            return np.zeros(0, dtype=np.float32)
        out = np.empty(end - start, dtype=np.float32)
        i, total = 0, end - start
        while i < total:
            c, o = divmod(start + i, self.chunk)
            k = min(total - i, self.chunk - o)
            out[i:i + k] = self.chunks[c][o:o + k]
            i += k
        return out

    def read_padded(self, start, n):
        """範囲外は無音で埋めて、必ず n サンプル返す。"""
        start, n = int(start), int(n)
        out = np.zeros(n, dtype=np.float32)
        skip = max(0, -start)
        if skip < n:
            data = self.read(start + skip, n - skip)
            out[skip:skip + len(data)] = data
        return out


class Peaks:
    """波形表示用の最小値・最大値。画面スレッドだけが触る。"""

    BIN0, BIN1 = 256, 4096

    def __init__(self, timeline):
        self.t = timeline
        self.done = 0
        self.n0 = 0
        self.n1 = 0
        self.min0 = np.zeros(4096, dtype=np.float32)
        self.max0 = np.zeros(4096, dtype=np.float32)
        self.min1 = np.zeros(256, dtype=np.float32)
        self.max1 = np.zeros(256, dtype=np.float32)

    @staticmethod
    def _grow(arr, need):
        if need <= len(arr):
            return arr
        out = np.zeros(max(need, len(arr) * 2), dtype=np.float32)
        out[:len(arr)] = arr
        return out

    def update(self):
        bins = (self.t.head - self.done) // self.BIN0
        if bins <= 0:
            return
        data = self.t.read(self.done, bins * self.BIN0).reshape(bins, self.BIN0)
        self.min0 = self._grow(self.min0, self.n0 + bins)
        self.max0 = self._grow(self.max0, self.n0 + bins)
        self.min0[self.n0:self.n0 + bins] = data.min(axis=1)
        self.max0[self.n0:self.n0 + bins] = data.max(axis=1)
        self.n0 += bins
        self.done += bins * self.BIN0
        ratio = self.BIN1 // self.BIN0
        coarse = self.n0 // ratio - self.n1
        if coarse > 0:
            s = self.n1 * ratio
            mins = self.min0[s:s + coarse * ratio].reshape(coarse, ratio).min(axis=1)
            maxs = self.max0[s:s + coarse * ratio].reshape(coarse, ratio).max(axis=1)
            self.min1 = self._grow(self.min1, self.n1 + coarse)
            self.max1 = self._grow(self.max1, self.n1 + coarse)
            self.min1[self.n1:self.n1 + coarse] = mins
            self.max1[self.n1:self.n1 + coarse] = maxs
            self.n1 += coarse

    def column_range(self, view_start, spp, width):
        """画面の各列の (x, min, max) を返す。"""
        if spp >= self.BIN1 and self.n1:
            bin_size, lo_arr, hi_arr, available = self.BIN1, self.min1[:self.n1], self.max1[:self.n1], self.n1 * self.BIN1
        elif spp >= self.BIN0 and self.n0:
            bin_size, lo_arr, hi_arr, available = self.BIN0, self.min0[:self.n0], self.max0[:self.n0], self.n0 * self.BIN0
        else:
            bin_size, lo_arr, hi_arr, available = None, None, None, self.t.head
        first = max(0, int(np.ceil(-view_start / spp)))
        last = min(width, int(np.floor((available - view_start) / spp)))
        if last <= first:
            empty = np.zeros(0)
            return empty, empty, empty
        xs = np.arange(first, last)
        left = view_start + xs * spp
        if bin_size is None:  # 拡大時は生のサンプルから作る
            start = int(max(0, left[0]))
            raw = self.t.read(start, int(left[-1] + spp) - start + 1)
            if len(raw) == 0:
                empty = np.zeros(0)
                return empty, empty, empty
            idx = np.clip((left - start).astype(np.int64), 0, len(raw) - 1)
            return xs, np.minimum.reduceat(raw, idx), np.maximum.reduceat(raw, idx)
        idx = np.clip((left / bin_size).astype(np.int64), 0, len(lo_arr) - 1)
        end = min(len(lo_arr), int((left[-1] + spp) / bin_size) + 1)
        end = max(end, int(idx[-1]) + 1)
        return xs, np.minimum.reduceat(lo_arr[:end], idx), np.maximum.reduceat(hi_arr[:end], idx)


# ------------------------------------------------------------- 保存 -----------
def wav_header(rate, data_bytes):
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 36 + data_bytes, b"WAVE", b"fmt ", 16,
        1, 1, rate, rate * BYTES_PER_SAMPLE, BYTES_PER_SAMPLE, BYTES_PER_SAMPLE * 8,
        b"data", data_bytes,
    )


def to_pcm24(samples):
    scaled = np.round(np.asarray(samples, dtype=np.float64) * 8388608.0)
    ints = np.clip(scaled, -8388608, 8388607).astype("<i4")
    return ints.view(np.uint8).reshape(-1, 4)[:, :3].tobytes()


class DiskWriter:
    """番組全体を1つのWAVに保存する。録音を止めて再開しても同じファイルにつながる。"""

    def __init__(self, path, rate):
        self.path = Path(path)
        self.rate = rate
        self.error = None
        self.done = threading.Event()
        self._q = queue.SimpleQueue()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "wb")
        self._f.write(wav_header(rate, 0))
        self._f.flush()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def push(self, samples):
        self._q.put(samples)

    def close(self):
        self._q.put(None)

    def _patch(self, written):
        self._f.flush()
        self._f.seek(0)
        self._f.write(wav_header(self.rate, written))
        self._f.seek(0, os.SEEK_END)
        self._f.flush()
        os.fsync(self._f.fileno())

    def _run(self):
        written = 0
        last = time.monotonic()
        try:
            while True:
                try:
                    item = self._q.get(timeout=0.25)
                except queue.Empty:
                    item = False
                if item is None:
                    break
                if item is not False and self.error is None:
                    data = to_pcm24(item)
                    if written + len(data) > MAX_DATA_BYTES:
                        self.error = "ファイルの上限に達したため保存を止めました（聞き返しは続けられます）"
                    else:
                        self._f.write(data)
                        written += len(data)
                now = time.monotonic()
                if now - last >= 1.0 and self.error is None:
                    self._patch(written)
                    last = now
        except Exception as e:
            self.error = f"保存エラー: {e}（聞き返しは続けられます）"
            while self._q.get() is not None:
                pass
        finally:
            try:
                self._patch(written)
                self._f.close()
            except Exception as e:
                self.error = self.error or f"保存エラー: {e}"
            self.done.set()


# ------------------------------------------------------------- 検聴 -----------
class CueRenderer:
    """検聴用の音を、別スレッドで少し先まで作っておく。"""

    def __init__(self, timeline):
        self.t = timeline
        self.rate = timeline.rate
        self.speed = 1.5
        self.keep_pitch = True
        self.lock = threading.Lock()
        self.need = threading.Event()
        self.buffer = deque()
        self.buffered = 0
        self.offset = 0  # 先頭ブロックの消費済みサンプル数
        self.gen = 0
        self.pending = None
        self.active = False
        self.heard = 0.0
        self.target = self.rate // 10
        self.quit = False
        # 早回しの設定（WSOLA。話し声向けに30ms窓）
        self.n = int(self.rate * 0.03) // 2 * 2
        self.hop = self.n // 2
        self.tol = int(self.rate * 0.008)
        self.window = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(self.n) / self.n)).astype(np.float32)
        self._reset(0.0)
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    # ---- 画面スレッドから ----
    def play_from(self, pos):
        with self.lock:
            self.gen += 1
            self.buffer.clear()
            self.buffered = 0
            self.offset = 0
            self.pending = float(pos)
            self.active = True
            self.heard = float(pos)
        self.need.set()

    def stop(self):
        with self.lock:
            self.gen += 1
            self.buffer.clear()
            self.buffered = 0
            self.offset = 0
            self.pending = None
            self.active = False

    def locate(self, pos):
        with self.lock:
            self.gen += 1
            self.buffer.clear()
            self.buffered = 0
            self.offset = 0
            self.pending = None
            self.active = False
            self.heard = float(pos)

    def playing(self):
        with self.lock:
            return self.active or bool(self.buffer)

    def position(self):
        return self.heard

    # ---- オーディオスレッドから ----
    def pull(self, frames):
        with self.lock:
            if not self.buffer and not self.active:
                return None
            self.target = max(self.rate // 10, frames * 3)
            out = np.zeros(frames, dtype=np.float32)
            filled = 0
            while filled < frames and self.buffer:
                samples, src_start, src_end = self.buffer[0]
                k = min(frames - filled, len(samples) - self.offset)
                out[filled:filled + k] = samples[self.offset:self.offset + k]
                filled += k
                self.offset += k
                self.buffered -= k
                self.heard = src_start + (src_end - src_start) * self.offset / len(samples)
                if self.offset >= len(samples):
                    self.buffer.popleft()
                    self.offset = 0
        self.need.set()
        return out

    # ---- 計算スレッド ----
    def _reset(self, pos):
        self._pos = float(pos)
        self._prev = None
        self._ola = np.zeros(self.n, dtype=np.float32)
        self._first = True

    def _limit(self):
        return self.t.head - int(self.rate * HEAD_MARGIN_SECONDS)

    def _render_direct(self):
        n = 480
        if self._pos + n > self._limit():
            return None
        start = self._pos
        block = self.t.read(int(start), n)
        self._pos = start + n
        self._prev = None
        return block, start, self._pos

    def _render_resample(self, speed):
        n = 480
        span = n * speed
        if self._pos + span + 2 > self._limit():
            return None
        start = self._pos
        base = int(start)
        data = self.t.read(base, int(span) + 3)
        positions = start - base + np.arange(n) * speed
        block = np.interp(positions, np.arange(len(data)), data).astype(np.float32)
        self._pos = start + span
        self._prev = None
        return block, start, self._pos

    def _render_wsola(self, speed):
        n, hop, tol = self.n, self.hop, self.tol
        if self._pos + hop * speed + n + tol > self._limit():
            return None
        start = self._pos
        nominal = int(start)
        if self._prev is None:
            frame = self.t.read_padded(nominal, n)
            chosen = nominal
        else:
            seg = self.t.read_padded(nominal - tol, n + 2 * tol)
            target = self.t.read_padded(self._prev + hop, n)
            seg_d, target_d = seg[::4], target[::4]
            corr = np.correlate(seg_d, target_d, mode="valid")
            energy = np.cumsum(seg_d.astype(np.float64) ** 2)
            m = len(target_d)
            window_energy = energy[m - 1:].copy()
            window_energy[1:] -= energy[:-m]
            score = corr / np.sqrt(np.maximum(window_energy, 1e-9))
            offset = int(np.argmax(score)) * 4
            frame = seg[offset:offset + n]
            chosen = nominal - tol + offset
        self._ola += frame * self.window
        block = self._ola[:hop].copy()
        self._ola = np.concatenate((self._ola[hop:], np.zeros(hop, dtype=np.float32)))
        self._prev = chosen
        self._pos = start + hop * speed
        return block, start, self._pos

    def _render(self):
        speed = self.speed
        if abs(speed - 1.0) < 1e-3:
            block = self._render_direct()
        elif self.keep_pitch:
            block = self._render_wsola(speed)
        else:
            block = self._render_resample(speed)
        if block is not None and self._first:  # 飛んだ先の頭のクリック音を防ぐ
            self._first = False
            samples = block[0]
            fade = min(len(samples), int(self.rate * 0.003))
            samples[:fade] *= np.linspace(0, 1, fade, dtype=np.float32)
        return block

    def _run(self):
        while not self.quit:
            self.need.wait(0.05)
            self.need.clear()
            while not self.quit:
                with self.lock:
                    if self.pending is not None:
                        self._reset(self.pending)
                        self.pending = None
                    if not self.active or self.buffered >= self.target:
                        break
                    gen = self.gen
                try:
                    block = self._render()
                except Exception:
                    log_exception()
                    block = None
                with self.lock:
                    if gen != self.gen:
                        continue
                    if block is None:  # 録音の先端に追いついた
                        self.active = False
                        break
                    self.buffer.append(block)
                    self.buffered += len(block[0])


# ---------------------------------------------------------- オーディオ -------
class Engine:
    def __init__(self, rate=48000):
        self.rate = rate
        self.timeline = Timeline(rate)
        self.cue = CueRenderer(self.timeline)
        self.streams = []
        self.recording = False
        self.writer = None
        self.in_channel = 0
        self.main_channel = 0
        self.cue_channel = 1
        self.online = False
        self.online_pos = 0
        self.online_stop = False
        self.online_ended = False
        self.peak_in = 0.0
        self.peak_main = 0.0
        self.peak_cue = 0.0
        self.problems = 0
        self.last_in = 0.0
        self.last_out = 0.0
        self.has_stream = False

    def set_rate(self, rate):
        """録音がまだ空のときだけ呼ぶ。"""
        self.cue.quit = True
        self.cue.need.set()
        self.rate = rate
        self.timeline = Timeline(rate)
        self.cue = CueRenderer(self.timeline)

    # ---- オーディオスレッド ----
    def _on_input(self, indata, status):
        self.last_in = time.monotonic()
        if status.input_overflow or status.input_underflow:
            self.problems += 1
        x = indata[:, self.in_channel]
        if len(x):
            peak = float(np.abs(x).max())
            if peak > self.peak_in:
                self.peak_in = peak
        if self.recording:
            copy = x.copy()
            writer = self.writer
            if writer is not None:
                writer.push(copy)
            self.timeline.write(copy)

    def _on_output(self, outdata, frames, status):
        self.last_out = time.monotonic()
        outdata.fill(0)
        if status.output_underflow or status.output_overflow:
            self.problems += 1
        if self.online:
            block = self.timeline.read(self.online_pos, frames)
            if self.online_stop:  # クリック音を避けて、1ブロックで絞って止める
                if len(block):
                    block = block * np.linspace(1, 0, len(block), dtype=np.float32)
                    self.online_pos += len(block)
                self.online_stop = False
                self.online = False
            limit = self.timeline.head - int(self.rate * HEAD_MARGIN_SECONDS)
            if self.online and self.online_pos + len(block) >= limit:
                self.online = False
                self.online_ended = True
                if self.online_pos < limit:
                    block = block[:limit - self.online_pos]
                    self.online_pos = limit
                else:
                    block = block[:0]
            elif self.online:
                self.online_pos += len(block)
            if len(block):
                outdata[:len(block), self.main_channel] = block
                peak = float(np.abs(block).max())
                if peak > self.peak_main:
                    self.peak_main = peak
            return
        block = self.cue.pull(frames)
        if block is not None:
            outdata[:, self.cue_channel] = block
            peak = float(np.abs(block).max())
            if peak > self.peak_cue:
                self.peak_cue = peak

    def _duplex(self, indata, outdata, frames, _t, status):
        try:
            self._on_input(indata, status)
        except Exception:
            self.problems += 1
        try:
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
        self.in_channel = int(cfg["input_channel"]) - 1
        self.main_channel = int(cfg["main_channel"]) - 1
        self.cue_channel = int(cfg["cue_channel"]) - 1
        n_in = self.in_channel + 1
        n_out = max(self.main_channel, self.cue_channel) + 1
        common = dict(samplerate=self.rate, dtype="float32", latency=cfg["latency"])
        self.last_in = self.last_out = time.monotonic()
        try:
            if in_dev == out_dev:
                self.streams = [sd.Stream(device=(in_dev, out_dev), channels=(n_in, n_out),
                                          callback=self._duplex, **common)]
            else:
                self.streams = [
                    sd.InputStream(device=in_dev, channels=n_in, callback=self._input_only, **common),
                    sd.OutputStream(device=out_dev, channels=n_out, callback=self._output_only, **common),
                ]
            for s in self.streams:
                s.start()
        except Exception:
            self.close()
            raise
        self.has_stream = True

    def close(self):
        for s in self.streams:
            try:
                s.stop()
                s.close()
            except Exception:
                pass
        self.streams = []
        self.has_stream = False
        self.online = False
        self.cue.stop()

    def healthy(self):
        if not self.has_stream:
            return False
        try:
            if not all(s.active for s in self.streams):
                return False
        except Exception:
            return False
        now = time.monotonic()
        return now - self.last_in < STALL_SECONDS and now - self.last_out < STALL_SECONDS

    def start_recording(self, save_dir):
        if self.writer is None:
            name = f"replay_{datetime.datetime.now():%Y%m%d_%H%M%S}.wav"
            self.writer = DiskWriter(Path(save_dir) / name, self.rate)
        elif self.timeline.head and (not self.timeline.seams or self.timeline.seams[-1] != self.timeline.head):
            self.timeline.seams.append(self.timeline.head)
        self.recording = True

    def stop_recording(self):
        self.recording = False

    def go_online(self, pos):
        self.cue.stop()
        self.online_pos = int(pos)
        self.online_stop = False
        self.online_ended = False
        self.online = True

    def stop_online(self):
        if self.healthy():
            self.online_stop = True
        else:
            self.online = False

    def shutdown(self):
        self.recording = False
        self.close()
        self.cue.quit = True
        self.cue.need.set()
        if self.writer is not None:
            self.writer.close()
            self.writer.done.wait(10)


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
    cfg = dict(DEFAULT_CONFIG)
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
    """ウィンドウを作る前に呼ぶ。"""
    if sys.platform != "win32":
        return
    w = ctypes.windll
    for call in (
        lambda: w.shcore.SetProcessDpiAwareness(1),
        lambda: w.imm32.ImmDisableIME(-1),  # 日本語入力にキーを取られないようにする
        lambda: w.winmm.timeBeginPeriod(1),
        lambda: w.kernel32.SetPriorityClass(w.kernel32.GetCurrentProcess(), 0x00000080),  # HIGH_PRIORITY_CLASS
        lambda: w.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001),  # スリープ抑止
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
LINE = "#343b46"
RED = "#e5343f"
GREEN = "#2fbf71"
AMBER = "#e6a817"
YELLOW = "#f2d13c"
WAVE = "#6ea8ff"
WAVE_BG = "#0c0e11"
EMPTY_BG = "#191c22"
RULER_BG = "#262c35"
RULER_H = 28


class Meter(tk.Canvas):
    def __init__(self, parent, height=14):
        super().__init__(parent, height=height, bg=WAVE_BG, highlightthickness=0)
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


class App:
    def __init__(self, root):
        self.root = root
        self.cfg = load_config()
        self.engine = Engine(int(self.cfg["samplerate"]))
        self.peaks = Peaks(self.engine.timeline)
        self.mode = LIVE
        self.marks = []
        self.marks_file = None
        self.spp = float(self.engine.rate * 30 / 900)  # 表示は約30秒から
        self.v_gain = 1.0
        self.free_view = None
        self.online_started = 0.0
        self.down = set()
        self.notice = ""
        self.notice_until = 0.0
        self.engine_error = ""
        self.tick_count = 0
        self.dialog_open = False

        families = set(tkfont.families())
        family = next((f for f in ("Yu Gothic UI", "Meiryo UI", "Noto Sans CJK JP") if f in families), "TkDefaultFont")
        mono = next((f for f in ("Consolas", "DejaVu Sans Mono") if f in families), family)
        self.f_mode = (family, 22, "bold")
        self.f_clock = (mono, 26, "bold")
        self.f_mid = (family, 13, "bold")
        self.f_body = (family, 11)
        self.f_small = (family, 10)
        self.f_ruler = (mono, 9)

        root.title(f"{APP_NAME} {VERSION}")
        root.configure(bg=BG)
        root.geometry("1100x660")
        root.minsize(900, 560)
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.build()
        root.bind("<KeyPress>", self.on_key)
        root.bind("<KeyRelease>", self.on_key_release)
        root.bind("<FocusOut>", self.on_focus_out)
        self.start_audio(first_run=True)
        root.focus_force()
        self.tick()

    # ---- 組み立て ----
    def build(self):
        style = ttk.Style(self.root)
        style.theme_use("clam")

        top = tk.Frame(self.root, bg=PANEL, padx=12, pady=6)
        top.pack(fill="x")
        tk.Label(top, text="検聴の速度", font=self.f_small, fg=MUTED, bg=PANEL).pack(side="left")
        self.v_speed = tk.DoubleVar(value=1.5)
        self.speed_scale = ttk.Scale(top, from_=1.0, to=2.0, variable=self.v_speed, length=260,
                                     takefocus=0, command=self.on_speed)
        self.speed_scale.pack(side="left", padx=8)
        self.speed_label = tk.Label(top, text="1.5 倍", font=self.f_mid, fg=TEXT, bg=PANEL, width=6)
        self.speed_label.pack(side="left")
        self.v_keep_pitch = tk.BooleanVar(value=True)
        self.pitch_button = tk.Checkbutton(top, text="声の高さを保つ", variable=self.v_keep_pitch, font=self.f_small,
                                           fg=TEXT, bg=PANEL, selectcolor="#0c0e11", activebackground=PANEL,
                                           activeforeground=TEXT, takefocus=0, command=self.on_pitch)
        self.pitch_button.pack(side="left", padx=(16, 0))
        self.settings_button = tk.Button(top, text="設定", font=self.f_small, fg=TEXT, bg="#2a303a",
                                         activebackground="#3a424e", activeforeground=TEXT, relief="flat", padx=12,
                                         takefocus=0, command=self.open_settings)
        self.settings_button.pack(side="right")
        self.reconnect_button = tk.Button(top, text="再接続", font=self.f_small, fg=TEXT, bg="#2a303a",
                                          activebackground="#3a424e", activeforeground=TEXT, relief="flat", padx=12,
                                          takefocus=0, command=self.reconnect)
        self.reconnect_button.pack(side="right", padx=(0, 6))

        self.banner = tk.Label(self.root, text="", font=self.f_mode, fg="#ffffff", bg=BG, pady=4)
        self.banner.pack(fill="x")

        self.hint = tk.Label(self.root, text="", font=self.f_body, fg=MUTED, bg=BG, anchor="w", padx=12, pady=6)
        self.hint.pack(side="bottom", fill="x")
        self.status = tk.Label(self.root, text="", font=self.f_small, fg=MUTED, bg=BG, anchor="w", padx=12)
        self.status.pack(side="bottom", fill="x")

        panel = tk.Frame(self.root, bg=PANEL, padx=12, pady=8)
        panel.pack(side="bottom", fill="x", padx=12, pady=(8, 2))
        panel.columnconfigure(3, weight=1)
        self.rec_button = tk.Button(panel, text="●  録音開始", font=self.f_mid, fg="#ffffff", bg="#3a424e",
                                    activebackground="#4a5360", activeforeground="#ffffff",
                                    disabledforeground="#6d7683", relief="flat", width=12, takefocus=0,
                                    command=self.toggle_record)
        self.rec_button.grid(row=0, column=0, rowspan=2, ipady=8, padx=(0, 16))
        tk.Label(panel, text="現在位置", font=self.f_small, fg=MUTED, bg=PANEL).grid(row=0, column=1, sticky="w")
        self.pos_label = tk.Label(panel, text="00:00:00.00", font=self.f_clock, fg=YELLOW, bg="#000000", padx=8)
        self.pos_label.grid(row=1, column=1, padx=(0, 12))
        tk.Label(panel, text="録音の長さ", font=self.f_small, fg=MUTED, bg=PANEL).grid(row=0, column=2, sticky="w")
        self.head_label = tk.Label(panel, text="00:00:00.00", font=self.f_clock, fg=YELLOW, bg="#000000", padx=8)
        self.head_label.grid(row=1, column=2, padx=(0, 16))
        meters = tk.Frame(panel, bg=PANEL)
        meters.grid(row=0, column=3, rowspan=2, sticky="ew")
        meters.columnconfigure(1, weight=1)
        self.meters = {}
        for i, (key, text) in enumerate((("in", "入力"), ("main", "本線 L"), ("cue", "検聴 R"))):
            tk.Label(meters, text=text, font=self.f_small, fg=MUTED, bg=PANEL).grid(row=i, column=0, sticky="w", padx=(0, 8), pady=2)
            m = Meter(meters)
            m.grid(row=i, column=1, sticky="ew", pady=2)
            self.meters[key] = m

        self.wave = tk.Canvas(self.root, bg=WAVE_BG, highlightthickness=0)
        self.wave.pack(fill="both", expand=True, padx=12)
        self.wave.bind("<Button-1>", self.on_wave_click)
        # ホイールはウィンドウ全体で受けて、マウスが波形の上にあるときだけ反応する
        self.root.bind("<MouseWheel>", self.on_wheel)
        self.root.bind("<Button-4>", lambda e: self.on_wheel(e, 120))
        self.root.bind("<Button-5>", lambda e: self.on_wheel(e, -120))

    # ---- オーディオ起動 ----
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
        if self.mode == ONLINE or self.engine.recording:
            self.show_notice("録音を止めてから再接続してください")
            return
        self.leave_search()
        self.engine.close()
        rescan_devices()
        self.start_audio()

    def open_settings(self):
        if self.mode == ONLINE or self.engine.recording:
            self.show_notice("録音を止めてから設定を開いてください")
            return
        if self.dialog_open:
            return
        self.leave_search()
        self.engine.close()
        rescan_devices()
        SettingsDialog(self)

    def apply_settings(self, cfg):
        rate = int(cfg["samplerate"])
        if rate != self.engine.rate and self.engine.timeline.head == 0 and self.engine.writer is None:
            self.engine.set_rate(rate)
            self.peaks = Peaks(self.engine.timeline)
            self.spp = float(rate * 30 / 900)
            self.apply_cue_settings()
        self.cfg = cfg
        save_config(cfg)
        self.start_audio()

    def apply_cue_settings(self):
        self.engine.cue.speed = round(float(self.v_speed.get()), 1)
        self.engine.cue.keep_pitch = bool(self.v_keep_pitch.get())

    # ---- 上部 ----
    def on_speed(self, _value=None):
        speed = round(float(self.v_speed.get()), 1)
        self.speed_label.configure(text=f"{speed:.1f} 倍")
        self.engine.cue.speed = speed
        self.root.focus_set()

    def on_pitch(self):
        self.engine.cue.keep_pitch = bool(self.v_keep_pitch.get())
        self.root.focus_set()

    def show_notice(self, text, seconds=3.0):
        self.notice = text
        self.notice_until = time.monotonic() + seconds
        self.update_status()

    # ---- 録音 ----
    def toggle_record(self):
        if self.mode == ONLINE:
            return
        if self.engine.recording:
            self.engine.stop_recording()
            return
        if not self.engine.healthy():
            self.show_notice("オーディオが動いていないため録音できません")
            return
        try:
            self.engine.start_recording(self.cfg["save_dir"])
            if self.marks_file is None:
                self.marks_file = self.engine.writer.path.with_suffix(".marks.txt")
        except OSError as e:
            log_exception()
            self.show_notice(f"録音を開始できません: {e}", 6.0)
        self.root.focus_set()

    # ---- 位置 ----
    def position(self):
        e = self.engine
        if self.mode == ONLINE:
            return e.online_pos
        if self.mode == SEARCH:
            return int(e.cue.position())
        return e.timeline.head

    def leave_search(self):
        self.engine.cue.stop()
        self.free_view = None
        self.mode = LIVE

    def clamp(self, pos):
        return max(0, min(int(pos), self.engine.timeline.head))

    def play_at(self, pos):
        self.mode = SEARCH
        self.free_view = None
        self.engine.cue.play_from(self.clamp(pos))

    def jump_keep_state(self, pos):
        """マウスや Shift+矢印の移動。止まっていれば飛ぶだけ、再生中なら飛んだ先から続ける。"""
        cue = self.engine.cue
        was_playing = self.mode == SEARCH and cue.playing()
        self.mode = SEARCH
        self.free_view = None
        if was_playing:
            cue.play_from(self.clamp(pos))
        else:
            cue.locate(self.clamp(pos))

    # ---- キー ----
    @staticmethod
    def is_letter(event):
        ks = event.keysym
        if len(ks) == 1 and ks.isascii() and ks.isalpha():
            return True
        if sys.platform == "win32" and (65 <= event.keycode <= 90 or event.keycode == 229):
            return True  # 229 は日本語入力が取ったキー
        return False

    def on_key(self, event):
        code = event.keycode
        if code in self.down:  # 押しっぱなしのリピートには反応しない
            return
        self.down.add(code)
        ks = event.keysym
        is_enter = ks in ("Return", "KP_Enter")
        if self.mode == ONLINE:
            if is_enter and time.monotonic() - self.online_started >= ENTER_GUARD_SECONDS:
                self.engine.stop_online()
            return
        shift = bool(event.state & 0x0001)
        if is_enter:
            self.key_enter()
        elif ks == "Escape":
            if self.mode == SEARCH:
                self.leave_search()
        elif ks == "space":
            self.key_space()
        elif ks == "Down":
            self.key_down()
        elif ks == "Left":
            self.key_mark_jump(-1) if shift else self.key_jump(-1)
        elif ks == "Right":
            self.key_mark_jump(1) if shift else self.key_jump(1)
        elif self.is_letter(event):
            self.add_mark()

    def on_key_release(self, event):
        self.down.discard(event.keycode)

    def on_focus_out(self, event):
        if event.widget is self.root:
            self.down.clear()

    def key_enter(self):
        if self.mode != SEARCH:  # 聞かずに出すことはないので、普段の表示では何もしない
            return
        if not self.engine.healthy():
            self.show_notice("オーディオが動いていないため再生できません")
            return
        pos = self.position()
        limit = self.engine.timeline.head - int(self.engine.rate * HEAD_MARGIN_SECONDS)
        if pos >= limit:
            self.show_notice("録音の先端なので出せる音がありません")
            return
        self.free_view = None
        self.engine.go_online(pos)
        self.mode = ONLINE
        self.online_started = time.monotonic()

    def key_space(self):
        if self.mode != SEARCH:
            return
        cue = self.engine.cue
        if cue.playing():
            cue.stop()  # 止めた位置がスタンバイ位置になる
        else:
            self.play_at(cue.position())

    def key_down(self):
        if not self.marks:
            self.show_notice("マークがまだありません")
            return
        offset = float(self.cfg.get("down_offset", 0.0)) * self.engine.rate
        self.play_at(self.marks[-1] - offset)

    def key_jump(self, direction):
        step = int(JUMP_SECONDS * self.engine.rate)
        if self.mode == LIVE and direction > 0:
            return
        if not self.engine.timeline.head:
            return
        self.play_at(self.position() + direction * step)

    def key_mark_jump(self, direction):
        if not self.marks:
            self.show_notice("マークがまだありません")
            return
        pos = self.position()
        margin = int(self.engine.rate * 0.05)  # 再生中に同じマークへ戻り続けないように
        if direction < 0:
            i = bisect.bisect_left(self.marks, pos - margin) - 1
        else:
            i = bisect.bisect_right(self.marks, pos + margin)
        if 0 <= i < len(self.marks):
            self.jump_keep_state(self.marks[i])

    def add_mark(self):
        head = self.engine.timeline.head
        if head == 0:
            self.show_notice("録音を始めるとマークを打てます")
            return
        if self.marks and self.marks[-1] == head:
            return
        self.marks.append(head)
        self.show_notice(f"マーク {len(self.marks)}  ({fmt_clock(head, self.engine.rate)})", 1.5)
        if self.marks_file is not None:
            try:
                with open(self.marks_file, "a", encoding="utf-8") as f:
                    f.write(f"{fmt_clock(head, self.engine.rate)}\n")
            except OSError:
                log_exception()

    # ---- マウス ----
    def view_start(self, width):
        if self.free_view is not None:
            return self.free_view
        return self.position() - width / 3 * self.spp

    def on_wave_click(self, event):
        # 波形の上は何も起きない。目盛りのクリックだけ飛ぶ（BE-200と同じ）
        if self.mode == ONLINE or event.y > RULER_H or not self.engine.timeline.head:
            return
        width = self.wave.winfo_width()
        target = self.view_start(width) + event.x * self.spp
        self.jump_keep_state(target)

    def on_wheel(self, event, delta=None):
        if self.mode == ONLINE:
            return
        mouse_x = self.root.winfo_pointerx() - self.wave.winfo_rootx()
        mouse_y = self.root.winfo_pointery() - self.wave.winfo_rooty()
        if not (0 <= mouse_x < self.wave.winfo_width() and 0 <= mouse_y < self.wave.winfo_height()):
            return
        delta = event.delta if delta is None else delta
        up = delta > 0
        if event.state & 0x0004:  # Ctrl：時間軸
            old = self.spp
            new = old / 1.25 if up else old * 1.25
            new = min(max(new, 2.0), self.engine.rate * 3600 * 6 / max(1, self.wave.winfo_width()))
            width = self.wave.winfo_width()
            if self.mode == SEARCH and not self.engine.cue.playing():
                # 止まっているときはマウスの位置を中心にする
                t = self.view_start(width) + mouse_x * old
                self.free_view = t - mouse_x * new
            self.spp = new
        elif event.state & 0x0001:  # Shift：縦
            self.v_gain = min(64.0, self.v_gain * 1.25) if up else max(1.0, self.v_gain / 1.25)

    # ---- 定期更新 ----
    def tick(self):
        try:
            self._tick()
        except Exception:
            log_exception()
        self.root.after(40, self.tick)

    def _tick(self):
        self.tick_count += 1
        e = self.engine
        try:
            e.timeline.reserve()
        except MemoryError:
            log_exception()
            self.show_notice("メモリが不足しています", 5.0)
        self.peaks.update()
        if self.mode == ONLINE and not e.online:
            self.mode = SEARCH
            e.cue.locate(e.online_pos)
        if self.free_view is not None and self.mode == SEARCH and e.cue.playing():
            self.free_view = None
        self.draw_wave()
        self.update_labels()
        self.meters["in"].set(e.peak_in)
        self.meters["main"].set(e.peak_main)
        self.meters["cue"].set(e.peak_cue)
        e.peak_in = e.peak_main = e.peak_cue = 0.0
        if self.tick_count % 12 == 0:
            self.update_status()

    def draw_wave(self):
        c = self.wave
        w, h = c.winfo_width(), c.winfo_height()
        if w < 10 or h < 40:
            return
        e = self.engine
        rate = e.rate
        c.delete("all")
        spp = self.spp
        start = self.view_start(w)
        wave_top = RULER_H
        wave_h = h - wave_top
        mid = wave_top + wave_h / 2
        scale = wave_h / 2 * self.v_gain
        c.create_rectangle(0, 0, w, wave_top, fill=RULER_BG, width=0)
        head = e.timeline.head
        # 録音のない範囲は背景を変える
        x0 = (0 - start) / spp
        x1 = (head - start) / spp
        c.create_rectangle(0, wave_top, w, h, fill=EMPTY_BG, width=0)
        if x1 > 0 and x0 < w:
            c.create_rectangle(max(0, x0), wave_top, min(w, x1), h, fill=WAVE_BG, width=0)
        # 目盛り
        seconds_per_px = spp / rate
        step = next((s for s in (0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 1200, 1800, 3600, 7200)
                     if s / seconds_per_px >= 90), 14400)
        t = max(0.0, np.floor(start / rate / step) * step)
        while True:
            x = (t * rate - start) / spp
            if x > w:
                break
            if x >= 0:
                c.create_line(x, wave_top - 8, x, wave_top, fill=MUTED)
                c.create_line(x, wave_top, x, h, fill="#1c2027")
                total = int(round(t * 100))
                hh, rem = divmod(total, 360000)
                mm, rem = divmod(rem, 6000)
                ss = rem / 100
                text = f"{hh}:{mm:02d}:{ss:02.0f}" if step >= 1 else f"{hh}:{mm:02d}:{ss:04.1f}"
                c.create_text(x + 3, 2, text=text, anchor="nw", fill=TEXT, font=self.f_ruler)
            t += step
        c.create_line(0, mid, w, mid, fill=LINE)
        # 波形
        xs, mins, maxs = self.peaks.column_range(start, spp, w)
        if len(xs) >= 2:
            ytop = np.maximum(mid - maxs * scale, wave_top)
            ybot = np.minimum(mid - mins * scale, h)
            ybot = np.maximum(ybot, ytop + 1)
            coords = np.empty(len(xs) * 4)
            coords[0::4] = xs
            coords[1::4] = ytop
            coords[2::4] = xs
            coords[3::4] = ybot
            c.create_line(coords.tolist(), fill=WAVE)
        # つなぎ目
        for seam in e.timeline.seams:
            x = (seam - start) / spp
            if 0 <= x <= w:
                c.create_line(x, wave_top, x, h, fill="#7d8796", dash=(2, 4))
        # マーク
        left = bisect.bisect_left(self.marks, start)
        right = bisect.bisect_right(self.marks, start + w * spp)
        for m in self.marks[left:right]:
            x = (m - start) / spp
            c.create_line(x, wave_top, x, h, fill=AMBER)
            c.create_polygon(x - 5, wave_top, x + 5, wave_top, x, wave_top + 8, fill=AMBER, width=0)
        # カーソル
        pos = self.position()
        x = (pos - start) / spp
        color = {LIVE: "#ffffff", SEARCH: YELLOW, ONLINE: RED}[self.mode]
        if 0 <= x <= w:
            c.create_line(x, 0, x, h, fill=color, width=2)
            if self.mode == SEARCH and not e.cue.playing():
                c.create_polygon(x - 7, 0, x + 7, 0, x, 12, fill=YELLOW, width=0)
                c.create_text(x + 9, wave_top + 3, text="STANDBY", anchor="nw", fill=YELLOW, font=self.f_small)

    def update_labels(self):
        e = self.engine
        rate = e.rate
        self.pos_label.configure(text=fmt_clock(self.position(), rate))
        self.head_label.configure(text=fmt_clock(e.timeline.head, rate))
        if self.mode == ONLINE:
            text, bg = "ON AIR  本線に出しています", RED
            hint = "Enter：止める　　（他の操作は受け付けません）"
        elif self.mode == SEARCH:
            if e.cue.playing():
                text, bg = f"検聴中  {e.cue.speed:.1f} 倍", "#1b6b45"
                hint = "← →：2秒戻す／送る　　Space：止める　　Enter：本線に出す　　Shift+← →：マーク移動　　Esc：やめる"
            else:
                text, bg = "スタンバイ", "#7a6410"
                hint = "Enter：ここから本線に出す　　← →：2秒戻す／送る　　Space：聞く　　Shift+← →：マーク移動　　Esc：やめる"
        else:
            if e.recording:
                text, bg = f"録音中　マーク {len(self.marks)}", BG
            else:
                text, bg = f"録音停止中　マーク {len(self.marks)}", "#5a2a10"
            hint = "アルファベット：マーク　　↓：最後のマークから検聴　　←：2秒戻して検聴"
        if not self.root_has_focus() and not self.dialog_open:
            text, bg = "このソフトが選ばれていません（キーが効きません）", RED
        self.banner.configure(text=text, bg=bg)
        self.hint.configure(text=hint)
        if e.recording:
            self.rec_button.configure(text="■  録音停止", bg=RED, activebackground="#f25560")
        else:
            self.rec_button.configure(text="●  録音開始", bg="#3a424e", activebackground="#4a5360")
        online = self.mode == ONLINE
        state = "disabled" if online else "normal"
        self.rec_button.configure(state=state)
        self.settings_button.configure(state=state)
        self.reconnect_button.configure(state=state)
        self.pitch_button.configure(state=state)
        self.speed_scale.configure(state=state)
        now = time.monotonic()
        if self.notice and now >= self.notice_until:
            self.notice = ""

    def root_has_focus(self):
        # focus コマンドは、このソフトにフォーカスがなければ空を返す
        return bool(str(self.root.tk.call("focus")))

    def update_status(self):
        e = self.engine
        color = MUTED
        if self.notice:
            text, color = self.notice, AMBER
        elif e.writer is not None and e.writer.error:
            text, color = e.writer.error, RED
        elif not e.healthy():
            text = self.engine_error or "オーディオが止まっています。機器を確認して「再接続」を押してください"
            color = RED
        else:
            text = f"{self.cfg['input_device']}  →  {self.cfg['output_device']}  /  {e.rate / 1000:g} kHz"
            used = len(e.timeline.chunks) * e.timeline.chunk * 4 / 1e9
            text += f"  /  メモリ {used:.1f} GB"
            try:
                free = shutil.disk_usage(Path(self.cfg["save_dir"]).anchor or ".").free
                text += f"  /  ディスク空き {free / 1e9:.0f} GB"
                if free < 2e9:
                    color = AMBER
            except OSError:
                pass
            if e.problems:
                text += f"  /  音声の取りこぼし {e.problems} 回"
                color = AMBER
        self.status.configure(text=text, fg=color)

    def on_close(self):
        if self.mode == ONLINE:
            self.show_notice("本線に出している間は終了できません。Enterで止めてください")
            return
        if self.engine.recording and not messagebox.askokcancel(APP_NAME, "録音中です。止めて終了しますか？"):
            return
        self.engine.shutdown()
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
        self.v_main = tk.StringVar(value=str(cfg["main_channel"]))
        self.v_cue = tk.StringVar(value=str(cfg["cue_channel"]))
        self.v_rate = tk.StringVar(value=str(cfg["samplerate"]))
        self.v_latency = tk.StringVar(value=cfg["latency"])
        self.v_offset = tk.StringVar(value=f"{float(cfg.get('down_offset', 0.0)):g}")
        self.v_dir = tk.StringVar(value=cfg["save_dir"])

        self.c_in = self.row(0, "入力デバイス", self.v_in, [d[1] for d in self.ins], 58)
        self.c_in_ch = self.row(1, "入力チャンネル", self.v_in_ch, [])
        self.c_out = self.row(2, "出力デバイス", self.v_out, [d[1] for d in self.outs], 58)
        self.c_main = self.row(3, "本線のチャンネル", self.v_main, [])
        self.c_cue = self.row(4, "検聴のチャンネル", self.v_cue, [])
        rate_box = self.row(5, "サンプルレート", self.v_rate, ["44100", "48000", "96000"])
        if app.engine.timeline.head or app.engine.writer is not None:
            rate_box.configure(state="disabled")  # 録音済みの途中では変えられない
        self.row(6, "遅延", self.v_latency, ["low", "high"])
        self.row(7, "↓の開始位置（マークの何秒前）", self.v_offset, ["0", "1", "2", "3", "5", "10"])
        tk.Label(self, text="保存先", font=app.f_small, fg=TEXT, bg=BG).grid(row=8, column=0, sticky="w", pady=4)
        folder = tk.Frame(self, bg=BG)
        folder.grid(row=8, column=1, sticky="ew")
        tk.Entry(folder, textvariable=self.v_dir, font=app.f_small, width=48).pack(side="left", fill="x", expand=True)
        tk.Button(folder, text="選ぶ", font=app.f_small, command=self.pick_dir).pack(side="left", padx=(6, 0))
        buttons = tk.Frame(self, bg=BG)
        buttons.grid(row=9, column=0, columnspan=2, sticky="e", pady=(12, 0))
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
                                           (self.c_main, self.v_main, out_values, 0),
                                           (self.c_cue, self.v_cue, out_values, 1)):
            box.configure(values=values)
            if var.get() not in values:
                var.set(values[min(fallback, len(values) - 1)])

    def pick_dir(self):
        d = filedialog.askdirectory(parent=self, initialdir=self.v_dir.get())
        if d:
            self.v_dir.set(d)

    def close_dialog(self):
        self.app.dialog_open = False
        self.grab_release()
        self.destroy()
        self.app.root.focus_force()

    def ok(self):
        if self.v_main.get() == self.v_cue.get() and not messagebox.askokcancel(
                APP_NAME, "本線と検聴が同じチャンネルです。検聴の音が本線に出ます。このまま進めますか？", parent=self):
            return
        cfg = dict(self.app.cfg)
        cfg.update(input_device=self.v_in.get(), output_device=self.v_out.get(),
                   input_channel=int(self.v_in_ch.get()), main_channel=int(self.v_main.get()),
                   cue_channel=int(self.v_cue.get()), samplerate=int(self.v_rate.get()),
                   latency=self.v_latency.get(), down_offset=float(self.v_offset.get()),
                   save_dir=self.v_dir.get())
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
        timeline = Timeline(48000)
        timeline.write(np.sin(np.arange(48000 * 3) * 0.1).astype(np.float32) * 0.1)
        cue = CueRenderer(timeline)
        cue.play_from(0)
        got = 0
        deadline = time.monotonic() + 5
        while got < 4800 and time.monotonic() < deadline:
            block = cue.pull(480)
            if block is not None and np.any(block):
                got += 480
            time.sleep(0.001)
        cue.quit = True
        if got < 4800:
            raise RuntimeError("検聴の音を作れません")
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
