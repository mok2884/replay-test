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
import sys
import threading
import time
import traceback
from collections import deque
from pathlib import Path

import numpy as np
import sounddevice as sd
import tkinter as tk
from tkinter import filedialog, font as tkfont, messagebox, ttk

APP_NAME = "REPLAY"
VERSION = "0.2"
APP_DIR = Path(sys.executable if getattr(sys, "frozen", False) else __file__).resolve().parent
CONFIG_PATH = APP_DIR / "replay_config.json"
API_ORDER = ("WASAPI", "WDM-KS", "DirectSound", "MME")

JUMP_SECONDS = 2.0
ENTER_GUARD_SECONDS = 0.5  # 本線に出した直後は止める操作を無視する
HEAD_MARGIN_SECONDS = 0.05  # 録音の先端の手前で再生を止める余裕
STALL_SECONDS = 2.0
MAX_DATA_BYTES = 4_000_000_000
BYTES_PER_SAMPLE = 2  # 16bit

LIVE, SEARCH, ONLINE = "live", "search", "online"

DEFAULT_CONFIG = {
    "input_device": "",
    "output_device": "",
    "samplerate": 48000,
    "input_channel": 1,
    "main_channel": 1,
    "cue_channel": 2,
    "latency": "high",
    "exclusive": True,
    "mark_offset": 2.0,
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


def to_int16(samples):
    scaled = np.round(np.asarray(samples, dtype=np.float64) * 32768.0)
    return np.clip(scaled, -32768, 32767).astype(np.int16)


# ------------------------------------------------------------ 録音データ -------
class Timeline:
    """録音全体を16bitでメモリに持つ。書き込みはオーディオスレッドのみ。"""

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
            a = np.empty(self.chunk, dtype=np.int16)
            a.fill(0)
            self.chunks.append(a)

    def write(self, x):
        if x.dtype != np.int16:
            x = to_int16(x)
        h = self.head
        i, n = 0, len(x)
        while i < n:
            c, o = divmod(h + i, self.chunk)
            if c >= len(self.chunks):
                self.chunks.append(np.zeros(self.chunk, dtype=np.int16))
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
        return out / 32768.0

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
                    data = item.astype("<i2").tobytes()
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
        self.exclusive_failed = False
        self.latency_ms = 0.0

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
            ints = to_int16(x)
            writer = self.writer
            if writer is not None:
                writer.push(ints)
            self.timeline.write(ints)

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
        self.exclusive_failed = False
        # WASAPIは排他モードで開いて、通知音などが出力に混ざらないようにする
        exclusive = bool(cfg.get("exclusive", True)) and str(cfg["output_device"]).startswith("[WASAPI]")
        attempts = [True, False] if exclusive else [False]
        for use_exclusive in attempts:
            try:
                self._start_streams(in_dev, out_dev, n_in, n_out, common, use_exclusive)
                self.has_stream = True
                return
            except Exception:
                self.close()
                if use_exclusive:
                    self.exclusive_failed = True
                    continue
                raise

    def _start_streams(self, in_dev, out_dev, n_in, n_out, common, use_exclusive):
        settings = sd.WasapiSettings(exclusive=True) if use_exclusive else None
        if in_dev == out_dev:
            self.streams = [sd.Stream(device=(in_dev, out_dev), channels=(n_in, n_out), callback=self._duplex,
                                      extra_settings=(None, settings), **common)]
        else:
            self.streams = [
                sd.InputStream(device=in_dev, channels=n_in, callback=self._input_only, **common),
                sd.OutputStream(device=out_dev, channels=n_out, callback=self._output_only,
                                extra_settings=settings, **common),
            ]
        for s in self.streams:
            s.start()
        latency = self.streams[-1].latency
        self.latency_ms = (latency[1] if isinstance(latency, (tuple, list)) else latency) * 1000

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
BG = "#c9ccd0"
PANEL = "#dfe1e4"
RAISED = "#eeeff1"
LINE = "#8f949a"
TEXT = "#191c1f"
MUTED = "#40454b"
WAVE_BG = "#eff0f2"
EMPTY_BG = "#bcc0c5"
RULER_BG = "#b4b8bd"
GRID = "#dcdee1"
WAVE = "#174fb3"
MARK = "#a84f00"
HEAD = "#50565c"
SEG_OFF = "#bcc0c5"
SEG_A = "#0c6d62"
SEG_B = "#a87400"
SEG_C = "#b42f20"
RED = "#cf2a35"
RED_TEXT = "#a81c27"
GREEN = "#1a7f46"
YELLOW = "#f2d13c"
YELLOW_TEXT = "#241d00"
CUE_LINE = "#8a6800"
LCD = "#0d0f0e"
LCD_TEXT = "#f2a93b"
LED_OK = "#18805f"
NOTICE = "#6b5000"
RULER_H = 30


def round_rect(canvas, x0, y0, x1, y1, r, **kw):
    r = max(0.0, min(r, (x1 - x0) / 2, (y1 - y0) / 2))
    pts = (x0 + r, y0, x1 - r, y0, x1, y0, x1, y0 + r, x1, y1 - r, x1, y1,
           x1 - r, y1, x0 + r, y1, x0, y1, x0, y1 - r, x0, y0 + r, x0, y0)
    return canvas.create_polygon(pts, smooth=True, splinesteps=12, **kw)


class RoundPanel(tk.Canvas):
    def __init__(self, app, parent, fill=PANEL, pad=12, radius=14, parent_bg=BG, fit_width=False, outline=""):
        super().__init__(parent, bg=parent_bg, highlightthickness=0, takefocus=0)
        self.pad = app.px(pad)
        self.radius = app.px(radius)
        self.fill = fill
        self.outline = outline
        self.fit_width = fit_width
        self.inner = tk.Frame(self, bg=fill)
        self.window = self.create_window(self.pad, self.pad, anchor="nw", window=self.inner)
        self.bind("<Configure>", self._layout)
        self.inner.bind("<Configure>", self._fit)

    def _fit(self, _event=None):
        need = self.inner.winfo_reqheight() + 2 * self.pad
        if int(self.cget("height")) != need:
            self.configure(height=need)
        if self.fit_width:
            need_w = self.inner.winfo_reqwidth() + 2 * self.pad
            if int(self.cget("width")) != need_w:
                self.configure(width=need_w)

    def _layout(self, _event=None):
        w, h = self.winfo_width(), self.winfo_height()
        self.delete("bg")
        round_rect(self, 1, 1, w - 2, h - 2, self.radius, fill=self.fill, outline=self.outline, tags="bg")
        self.tag_lower("bg")
        self.itemconfigure(self.window, width=max(1, w - 2 * self.pad), height=max(1, h - 2 * self.pad))


class RoundButton(tk.Canvas):
    def __init__(self, app, parent, text="", on_click=None, height=44, radius=22, font=None, width=None,
                 parent_bg=BG, fill=RAISED, outline=LINE, fg=TEXT):
        super().__init__(parent, height=app.px(height), bg=parent_bg, highlightthickness=0, takefocus=0, cursor="hand2")
        if width:
            self.configure(width=app.px(width))
        self.app = app
        self.text = text
        self.radius = app.px(radius)
        self.font = font or app.f_small
        self.fill = fill
        self.outline = outline
        self.fg = fg
        self.on_click = on_click
        self.enabled = True
        self.bind("<Configure>", lambda _e: self.redraw())
        self.bind("<ButtonRelease-1>", self._release)

    def _release(self, _event):
        if self.enabled and self.on_click:
            self.on_click()

    def set_style(self, fill=None, outline=None, fg=None, text=None, enabled=None):
        changed = False
        for name, value in (("fill", fill), ("outline", outline), ("fg", fg), ("text", text), ("enabled", enabled)):
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
        round_rect(self, 1, 1, w - 2, h - 2, self.radius, fill=self.fill, outline=self.outline)
        self.create_text(w / 2, h / 2, text=self.text, fill=self.fg, font=self.font)


class KeyCap(tk.Canvas):
    def __init__(self, app, parent, text, width=80, height=40, color=TEXT, border=2, font=None, parent_bg=PANEL, radius=9):
        super().__init__(parent, width=app.px(width), height=app.px(height), bg=parent_bg, highlightthickness=0, takefocus=0)
        w, h = app.px(width), app.px(height)
        round_rect(self, 1, 1, w - 2, h - 2, app.px(radius), fill=RAISED, outline=color, width=app.px(border))
        self.create_text(w / 2, h / 2, text=text, fill=color, font=font or app.f_key)


class SegmentMeter(tk.Canvas):
    COUNT = 32

    def __init__(self, app, parent):
        super().__init__(parent, height=app.px(10), bg=PANEL, highlightthickness=0, takefocus=0)
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
            zone = SEG_A if i < 22 else SEG_B if i < 28 else SEG_C
            self.itemconfigure(item, fill=zone if i < lit else SEG_OFF)
        self.lit = lit


class TransportButton(tk.Canvas):
    """再生やマーク移動のボタン。記号と、必要なら下に短い文字を出す。"""

    def __init__(self, app, parent, kind, label="", on_click=None, width=60, height=56, round_shape=False):
        super().__init__(parent, width=app.px(width), height=app.px(height), bg=PANEL, highlightthickness=0, takefocus=0, cursor="hand2")
        self.app = app
        self.kind = kind
        self.label = label
        self.on_click = on_click
        self.round_shape = round_shape
        self.w, self.h = app.px(width), app.px(height)
        self.enabled = True
        self.bind("<ButtonRelease-1>", self._release)
        self.redraw()

    def _release(self, _event):
        if self.enabled and self.on_click:
            self.on_click()

    def set_kind(self, kind=None, enabled=None):
        changed = False
        if kind is not None and kind != self.kind:
            self.kind = kind
            changed = True
        if enabled is not None and enabled != self.enabled:
            self.enabled = enabled
            changed = True
        if changed:
            self.redraw()

    def redraw(self):
        self.delete("all")
        app, w, h = self.app, self.w, self.h
        fg = TEXT if self.enabled else "#8f949a"
        if self.round_shape:
            self.create_oval(1, 1, w - 2, h - 2, fill=TEXT if self.enabled else "#8f949a", outline="")
            fg = RAISED
        else:
            round_rect(self, 1, 1, w - 2, h - 2, app.px(12), fill=RAISED, outline=LINE)
        cx = w / 2
        cy = h / 2 if not self.label else h * 0.38
        s = app.px(9)

        def tri(x, direction):
            self.create_polygon(x, cy - s, x, cy + s, x + direction * s * 1.2, cy, fill=fg, width=0)

        if self.kind == "prev":
            tri(cx + s * 0.1, -1)
            tri(cx + s * 1.3, -1)
        elif self.kind == "next":
            tri(cx - s * 0.1, 1)
            tri(cx - s * 1.3, 1)
        elif self.kind == "live":
            tri(cx - s * 1.6, 1)
            tri(cx - s * 0.5, 1)
            self.create_rectangle(cx + s * 0.8, cy - s, cx + s * 1.15, cy + s, fill=fg, width=0)
        elif self.kind == "play":
            self.create_polygon(cx - s * 0.8, cy - s * 1.2, cx - s * 0.8, cy + s * 1.2, cx + s * 1.3, cy, fill=fg, width=0)
        elif self.kind == "stop":
            self.create_rectangle(cx - s, cy - s, cx + s, cy + s, fill=fg, width=0)
        elif self.kind:
            self.create_text(cx, cy, text=self.kind, fill=fg, font=app.f_transport)
        if self.label:
            self.create_text(cx, h * 0.78, text=self.label, fill=fg, font=app.f_mono)


class App:
    def __init__(self, root):
        self.root = root
        self.scale = max(1.0, root.winfo_fpixels("1i") / 96)
        self.cfg = load_config()
        self.engine = Engine(int(self.cfg["samplerate"]))
        self.peaks = Peaks(self.engine.timeline)
        self.mode = LIVE
        self.marks = []
        self.marks_file = None
        self.current_mark = None
        self.spp = float(self.engine.rate * 30 / 900)
        self.v_gain = 1.0
        self.free_view = None
        self.online_started = 0.0
        self.down = set()
        self.shift_held = False
        self.notice = ""
        self.notice_until = 0.0
        self.engine_error = ""
        self.tick_count = 0
        self.settings_open = False
        self.settings_view = None
        self.band_state = None

        families = set(tkfont.families())
        family = next((f for f in ("Yu Gothic UI", "Meiryo UI", "Noto Sans CJK JP") if f in families), "TkDefaultFont")
        mono = next((f for f in ("Consolas", "DejaVu Sans Mono") if f in families), family)
        self.f_title = (family, -self.px(16), "bold")
        self.f_band = (family, -self.px(34), "bold")
        self.f_clock = (mono, -self.px(30), "bold")
        self.f_guide_title = (family, -self.px(26), "bold")
        self.f_guide = (family, -self.px(14))
        self.f_key = (mono, -self.px(12), "bold")
        self.f_enter = (mono, -self.px(16), "bold")
        self.f_enter_label = (family, -self.px(13), "bold")
        self.f_mid = (family, -self.px(17), "bold")
        self.f_body = (family, -self.px(13))
        self.f_small = (family, -self.px(13))
        self.f_mono = (mono, -self.px(11))
        self.f_mono_big = (mono, -self.px(18))
        self.f_speed = (mono, -self.px(15), "bold")
        self.f_transport = (mono, -self.px(14), "bold")

        root.title(f"チャレンジREC {VERSION}")
        root.configure(bg=BG)
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        root.geometry(f"{min(self.px(1320), sw - 40)}x{min(self.px(860), sh - 90)}")
        root.minsize(self.px(1000), min(self.px(680), sh - 90))
        if sys.platform == "win32":
            try:
                root.state("zoomed")  # 全画面での使用が基本
            except tk.TclError:
                pass
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.build_style()
        self.build_statusbar()
        self.main_view = tk.Frame(root, bg=BG)
        self.build(self.main_view)
        self.main_view.pack(fill="both", expand=True, padx=self.px(16), pady=(self.px(6), self.px(9)))
        root.bind("<KeyPress>", self.on_key)
        root.bind("<KeyRelease>", self.on_key_release)
        root.bind("<FocusOut>", self.on_focus_out)
        root.bind("<MouseWheel>", self.on_wheel)
        root.bind("<Button-4>", lambda e: self.on_wheel(e, 120))
        root.bind("<Button-5>", lambda e: self.on_wheel(e, -120))
        self.start_audio(first_run=True)
        root.focus_force()
        self.tick()

    def px(self, n):
        return int(round(n * getattr(self, "scale", 1.0)))

    def build_style(self):
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("Horizontal.TScale", background=PANEL, troughcolor=SEG_OFF, bordercolor=LINE, lightcolor=PANEL, darkcolor=LINE)
        style.configure("Light.TCombobox", fieldbackground=RAISED, background=PANEL, foreground=TEXT, arrowcolor=TEXT,
                        bordercolor=LINE, lightcolor=LINE, darkcolor=LINE, padding=self.px(8))
        style.map("Light.TCombobox", fieldbackground=[("readonly", RAISED)], foreground=[("readonly", TEXT)],
                  selectbackground=[("readonly", RAISED)], selectforeground=[("readonly", TEXT)])
        self.root.option_add("*TCombobox*Listbox.background", RAISED)
        self.root.option_add("*TCombobox*Listbox.foreground", TEXT)
        self.root.option_add("*TCombobox*Listbox.selectBackground", WAVE)
        self.root.option_add("*TCombobox*Listbox.selectForeground", "#ffffff")

    # ---- 組み立て ----
    def build_statusbar(self):
        px = self.px
        bar = tk.Frame(self.root, bg=RULER_BG, highlightthickness=1, highlightbackground=LINE)
        bar.pack(side="bottom", fill="x")
        self.status_items = {}
        for key in ("state", "format", "latency", "dropouts", "memory", "disk"):
            label = tk.Label(bar, text="", font=self.f_mono, fg=TEXT, bg=RULER_BG)
            label.pack(side="left", padx=(px(16) if key == "state" else 0, px(20)), pady=px(3))
            self.status_items[key] = label
        self.status_file = tk.Label(bar, text="", font=self.f_mono, fg=TEXT, bg=RULER_BG)
        self.status_file.pack(side="right", padx=px(16))

    def build(self, parent):
        px = self.px
        header = tk.Frame(parent, bg=BG)
        header.pack(fill="x", pady=(0, px(8)))
        logo = tk.Canvas(header, width=px(22), height=px(22), bg=BG, highlightthickness=0)
        logo.create_oval(1, 1, px(21), px(21), fill=RAISED, outline=LINE)
        logo.create_arc(px(6), px(6), px(16), px(16), start=30, extent=300, style="arc", outline=WAVE, width=px(2))
        logo.pack(side="left", padx=(0, px(8)))
        tk.Label(header, text="チャレンジREC", font=self.f_title, fg=TEXT, bg=BG).pack(side="left")
        self.not_active = tk.Canvas(header, width=px(102), height=px(24), bg=BG, highlightthickness=0)
        round_rect(self.not_active, 1, 1, px(100), px(22), px(11), fill=YELLOW, outline="")
        self.not_active.create_text(px(51), px(12), text="NOT ACTIVE", fill=YELLOW_TEXT, font=self.f_mono)
        self.not_active_shown = False
        self.settings_button = RoundButton(self, header, text="設定", on_click=self.open_settings, width=72, height=30, radius=15)
        self.settings_button.pack(side="right")
        self.reconnect_button = RoundButton(self, header, text="RECONNECT", on_click=self.reconnect, width=108, height=30, radius=15)
        self.reconnect_shown = False
        self.status = tk.Label(header, text="", font=self.f_small, fg=MUTED, bg=BG, anchor="e")
        self.status.pack(side="right", padx=px(14), fill="x", expand=True)

        guide_panel = RoundPanel(self, parent, pad=8)
        guide_panel.pack(fill="x")
        guide = guide_panel.inner
        tk.Label(guide, text="操作説明", font=self.f_guide_title, fg=TEXT, bg=PANEL).pack(side="left", padx=(px(6), px(14)))
        tk.Frame(guide, bg=LINE, width=1).pack(side="left", fill="y", padx=(0, px(14)))
        keys = tk.Frame(guide, bg=PANEL)
        keys.pack(side="left", fill="x", expand=True)
        rows = ((("A–Z", "マーク"), ("↓", "前のマーク"), ("↑", "次のマーク"), ("ESC", "先端に戻る")),
                (("← →", "2秒"), ("SPACE", "再生 / 止める"), ("SHIFT", "押す間 等倍")))
        for c in range(4):
            keys.columnconfigure(c, weight=1, uniform="guide")
        for r, row in enumerate(rows):
            for c, (key, text) in enumerate(row):
                cell = tk.Frame(keys, bg=PANEL)
                cell.grid(row=r, column=c, sticky="w", pady=px(2))
                KeyCap(self, cell, key, width=54, height=26, color=MUTED, border=1, radius=6).pack(side="left", padx=(0, px(8)))
                tk.Label(cell, text=text, font=self.f_guide, fg=TEXT, bg=PANEL).pack(side="left")
        enter = tk.Frame(guide, bg=PANEL)
        enter.pack(side="right", padx=(px(14), px(4)))
        KeyCap(self, enter, "ENTER", width=130, height=36, color=RED_TEXT, border=2, font=self.f_enter, radius=8).pack()
        tk.Label(enter, text="ONLINE PLAY / STOP", font=self.f_enter_label, fg=RED_TEXT, bg=PANEL).pack(pady=(px(2), 0))

        self.band = tk.Canvas(parent, height=px(70), bg=BG, highlightthickness=0, takefocus=0)
        self.band.pack(fill="x", pady=px(8))
        self.band.bind("<Configure>", lambda _e: self.draw_band(force=True))

        row = tk.Frame(parent, bg=BG)
        row.pack(fill="x", pady=(0, px(8)))
        row.columnconfigure(1, weight=1, uniform="info")
        row.columnconfigure(2, weight=1, uniform="info")
        clock = RoundPanel(self, row, fill=LCD, pad=7, radius=10, fit_width=True)
        clock.grid(row=0, column=0, sticky="ns", padx=(0, px(8)))
        tk.Label(clock.inner, text="POSITION", font=self.f_mono, fg="#c2c0b9", bg=LCD).pack(side="left", padx=(px(6), px(10)))
        self.pos_label = tk.Label(clock.inner, text="00:00:00.00", font=self.f_clock, fg=LCD_TEXT, bg=LCD)
        self.pos_label.pack(side="left", padx=(0, px(6)))

        speed = RoundPanel(self, row, pad=7, radius=10)
        speed.grid(row=0, column=1, sticky="nsew", padx=(0, px(8)))
        line = tk.Frame(speed.inner, bg=PANEL)
        line.pack(fill="both", expand=True, padx=px(6))
        tk.Label(line, text="CUE SPEED", font=self.f_mono, fg=MUTED, bg=PANEL).pack(side="left")
        self.v_speed = tk.DoubleVar(value=1.5)
        self.speed_scale = ttk.Scale(line, from_=1.0, to=2.0, variable=self.v_speed, takefocus=0, command=self.on_speed)
        self.speed_scale.pack(side="left", fill="x", expand=True, padx=px(8))
        self.speed_label = tk.Label(line, text="1.5×", font=self.f_speed, fg=TEXT, bg=PANEL, width=4)
        self.speed_label.pack(side="left")
        self.v_keep_pitch = tk.BooleanVar(value=True)
        self.pitch_button = tk.Checkbutton(line, text="KEEP PITCH", variable=self.v_keep_pitch, font=self.f_mono, fg=MUTED,
                                           bg=PANEL, activebackground=PANEL, activeforeground=TEXT, selectcolor=RAISED, takefocus=0,
                                           highlightthickness=0, command=self.on_pitch)
        self.pitch_button.pack(side="left", padx=(px(8), 0))

        devices = RoundPanel(self, row, pad=7, radius=10)
        devices.grid(row=0, column=2, sticky="nsew")
        devices.inner.columnconfigure((0, 1), weight=1, uniform="device")
        devices.inner.rowconfigure(0, weight=1)
        self.device_rows = {}
        for col, (key, title) in enumerate((("in", "INPUT"), ("out", "OUTPUT"))):
            line = tk.Frame(devices.inner, bg=PANEL)
            line.grid(row=0, column=col, sticky="ew", padx=px(4))
            led = tk.Canvas(line, width=px(9), height=px(9), bg=PANEL, highlightthickness=0)
            item = led.create_oval(0, 0, px(8), px(8), fill=LED_OK, outline="")
            led.pack(side="left", padx=(0, px(6)))
            tk.Label(line, text=title, font=self.f_mono, fg=MUTED, bg=PANEL).pack(side="left", padx=(0, px(6)))
            # width=1 にして、長い名前は枠の中で切れるようにする
            name = tk.Label(line, text="", font=self.f_body, fg=TEXT, bg=PANEL, anchor="w", width=1)
            name.pack(side="left", fill="x", expand=True)
            self.device_rows[key] = (led, item, name)

        bottom = tk.Frame(parent, bg=BG)
        bottom.pack(side="bottom", fill="x", pady=(px(8), 0))
        bottom.columnconfigure(3, weight=1)
        self.rec_button = RoundButton(self, bottom, text="●  REC START", on_click=self.toggle_record, width=170, height=76, radius=14, font=self.f_mid)
        self.rec_button.grid(row=0, column=0, sticky="ns", padx=(0, px(8)))
        info = RoundPanel(self, bottom, pad=10, fit_width=True)
        info.grid(row=0, column=1, sticky="ns", padx=(0, px(8)))
        length = tk.Frame(info.inner, bg=PANEL)
        length.pack(side="left", padx=(px(6), px(14)))
        tk.Label(length, text="LENGTH", font=self.f_mono, fg=MUTED, bg=PANEL).pack(anchor="w")
        self.head_label = tk.Label(length, text="00:00:00", font=self.f_mono_big, fg=MUTED, bg=PANEL)
        self.head_label.pack(anchor="w")
        marks = tk.Frame(info.inner, bg=PANEL)
        marks.pack(side="left", padx=(0, px(6)))
        tk.Label(marks, text="MARKS", font=self.f_mono, fg=MUTED, bg=PANEL).pack(anchor="w")
        self.marks_label = tk.Label(marks, text="0", font=self.f_mono_big, fg=TEXT, bg=PANEL)
        self.marks_label.pack(anchor="w")
        transport = RoundPanel(self, bottom, pad=8, fit_width=True)
        transport.grid(row=0, column=2, sticky="ns", padx=(0, px(8)))
        self.transport = {}
        for key, kind, label, command, width, height, is_round in (
                ("prev", "prev", "MARK", lambda: self.key_mark(-1), 60, 56, False),
                ("back", "−2s", "", lambda: self.key_jump(-1), 52, 52, False),
                ("play", "play", "", self.transport_play, 60, 60, True),
                ("forward", "+2s", "", lambda: self.key_jump(1), 52, 52, False),
                ("next", "next", "MARK", lambda: self.key_mark(1), 60, 56, False),
                ("live", "live", "LIVE", self.transport_live, 60, 56, False)):
            button = TransportButton(self, transport.inner, kind, label, command, width, height, is_round)
            button.pack(side="left", padx=px(3))
            self.transport[key] = button
        meters_panel = RoundPanel(self, bottom, pad=12)
        meters_panel.grid(row=0, column=3, sticky="nsew")
        meters_panel.inner.columnconfigure(1, weight=1)
        self.meters = {}
        for i, (key, text) in enumerate((("in", "INPUT"), ("main", "MAIN L"), ("cue", "CUE R"))):
            tk.Label(meters_panel.inner, text=text, font=self.f_mono, fg=MUTED, bg=PANEL, width=7, anchor="w").grid(row=i, column=0, padx=(0, px(8)), pady=px(2))
            m = SegmentMeter(self, meters_panel.inner)
            m.grid(row=i, column=1, sticky="ew")
            self.meters[key] = m

        self.wave = tk.Canvas(parent, bg=WAVE_BG, highlightthickness=0, height=px(200), takefocus=0)
        self.wave.pack(fill="both", expand=True)
        self.wave.bind("<Button-1>", self.on_wave_click)

    # ---- オーディオ起動 ----
    def start_audio(self, first_run=False):
        if not self.cfg["input_device"] or not self.cfg["output_device"]:
            self.engine_error = "SELECT A デバイス IN 設定"
            if first_run:
                self.root.after(200, self.open_settings)
            return
        try:
            self.engine.open(self.cfg)
            self.engine_error = ""
            if self.cfg.get("exclusive") and self.engine.exclusive_failed:
                self.show_notice("EXCLUSIVE NOT AVAILABLE. WINDOWS SOUNDS MAY MIX IN", 8.0)
        except Exception as e:
            log_exception()
            self.engine_error = f"CANNOT OPEN AUDIO: {e}"

    def reconnect(self):
        if self.mode == ONLINE or self.engine.recording:
            self.show_notice("STOP RECORDING FIRST")
            return
        self.leave_search()
        self.engine.close()
        rescan_devices()
        self.start_audio()
        self.root.focus_set()

    def open_settings(self):
        if self.mode == ONLINE or self.engine.recording:
            self.show_notice("STOP RECORDING FIRST")
            return
        if self.settings_open:
            return
        self.leave_search()
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
        self.main_view.pack(fill="both", expand=True, padx=self.px(16), pady=(self.px(6), self.px(9)))
        if cfg is not None:
            rate = int(cfg["samplerate"])
            if rate != self.engine.rate and self.engine.timeline.head == 0 and self.engine.writer is None:
                self.engine.set_rate(rate)
                self.peaks = Peaks(self.engine.timeline)
                self.spp = float(rate * 30 / 900)
                self.apply_cue_settings()
            self.cfg = cfg
            save_config(cfg)
            if cfg["main_channel"] == cfg["cue_channel"]:
                self.show_notice("MAIN AND CUE USE THE SAME CHANNEL", 8.0)
        self.start_audio()
        self.root.focus_force()
        self.draw_band(force=True)

    def apply_cue_settings(self):
        self.engine.cue.speed = self.effective_speed()
        self.engine.cue.keep_pitch = bool(self.v_keep_pitch.get())

    # ---- 上部 ----
    def effective_speed(self):
        return 1.0 if self.shift_held else round(float(self.v_speed.get()), 1)

    def on_speed(self, _value=None):
        if self.mode == ONLINE:
            return
        speed = round(float(self.v_speed.get()), 1)
        self.speed_label.configure(text=f"{speed:.1f}×")
        self.engine.cue.speed = self.effective_speed()
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
            self.show_notice("AUDIO IS NOT RUNNING")
            return
        try:
            self.engine.start_recording(self.cfg["save_dir"])
            if self.marks_file is None:
                self.marks_file = self.engine.writer.path.with_suffix(".marks.txt")
        except OSError as e:
            log_exception()
            self.show_notice(f"CANNOT START RECORDING: {e}", 6.0)
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
        self.current_mark = None
        self.mode = LIVE

    def clamp(self, pos):
        return max(0, min(int(pos), self.engine.timeline.head))

    def play_at(self, pos):
        self.mode = SEARCH
        self.free_view = None
        self.engine.cue.speed = self.effective_speed()
        self.engine.cue.play_from(self.clamp(pos))

    def jump_keep_state(self, pos):
        """マウスの移動。止まっていれば飛ぶだけ、再生中なら飛んだ先から続ける。"""
        cue = self.engine.cue
        was_playing = self.mode == SEARCH and cue.playing()
        self.mode = SEARCH
        self.free_view = None
        self.current_mark = None
        if was_playing:
            cue.play_from(self.clamp(pos))
        else:
            cue.locate(self.clamp(pos))

    # ---- ボタン ----
    def transport_play(self):
        if self.mode == SEARCH:
            self.key_space()
        elif self.mode == LIVE and self.marks:
            self.key_mark(-1)
        self.root.focus_set()

    def transport_live(self):
        if self.mode == SEARCH:
            self.leave_search()
        self.root.focus_set()

    # ---- キー ----
    @staticmethod
    def is_letter(event):
        ks = event.keysym
        if len(ks) == 1 and ks.isascii() and ks.isalpha():
            return True
        if sys.platform == "win32" and (65 <= event.keycode <= 90 or event.keycode == 229):
            return True
        return False

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
        is_enter = ks in ("Return", "KP_Enter")
        if self.mode == ONLINE:
            if is_enter and time.monotonic() - self.online_started >= ENTER_GUARD_SECONDS:
                self.engine.stop_online()
            return
        if ks in ("Shift_L", "Shift_R"):
            self.set_shift(True)
        elif is_enter:
            self.key_enter()
        elif ks == "Escape":
            if self.mode == SEARCH:
                self.leave_search()
        elif ks == "space":
            self.key_space()
        elif ks == "Down":
            self.key_mark(-1)
        elif ks == "Up":
            self.key_mark(1)
        elif ks == "Left":
            self.key_jump(-1)
        elif ks == "Right":
            self.key_jump(1)
        elif self.is_letter(event):
            self.add_mark()

    def on_key_release(self, event):
        self.down.discard(event.keycode)
        if event.keysym in ("Shift_L", "Shift_R"):
            self.set_shift(False)

    def on_focus_out(self, event):
        if event.widget is self.root:
            self.down.clear()
            self.set_shift(False)

    def set_shift(self, held):
        if held == self.shift_held:
            return
        self.shift_held = held
        cue = self.engine.cue
        cue.speed = self.effective_speed()
        # すでに作ってある分を捨てて、聞いている位置からすぐ速度を切り替える
        if self.mode == SEARCH and cue.playing():
            cue.play_from(cue.position())

    def key_enter(self):
        if self.mode != SEARCH:  # 聞かずに出すことはないので、先端では何もしない
            return
        if not self.engine.healthy():
            self.show_notice("AUDIO IS NOT RUNNING")
            return
        pos = self.position()
        limit = self.engine.timeline.head - int(self.engine.rate * HEAD_MARGIN_SECONDS)
        if pos >= limit:
            self.show_notice("NOTHING TO PLAY AT THE LIVE POSITION")
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

    def key_mark(self, direction):
        """↓で前のマーク、↑で次のマーク。マークの少し前から検聴する。"""
        if not self.marks:
            self.show_notice("NO MARKS YET")
            return
        i = self.current_mark
        if i is None or i >= len(self.marks):
            pos = self.position()
            if direction < 0:
                i = bisect.bisect_right(self.marks, pos) - 1
            else:
                i = bisect.bisect_right(self.marks, pos)
        else:
            i += direction
        if not 0 <= i < len(self.marks):
            return
        self.current_mark = i
        offset = float(self.cfg.get("mark_offset", 2.0)) * self.engine.rate
        self.play_at(self.marks[i] - offset)

    def key_jump(self, direction):
        step = int(JUMP_SECONDS * self.engine.rate)
        if self.mode == LIVE and direction > 0:
            return
        if not self.engine.timeline.head:
            return
        self.current_mark = None
        self.play_at(self.position() + direction * step)

    def add_mark(self):
        head = self.engine.timeline.head
        if head == 0:
            self.show_notice("START RECORDING TO MARK")
            return
        if self.marks and self.marks[-1] == head:
            return
        self.marks.append(head)
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

    def grid_start(self, width):
        """画面の左端を、時間で固定した列の境目に揃える。"""
        return np.floor(self.view_start(width) / self.spp) * self.spp

    def on_wave_click(self, event):
        # 波形でもタイムラインでも、叩いた場所に飛ぶ
        if self.mode == ONLINE or not self.engine.timeline.head:
            return
        width = self.wave.winfo_width()
        self.jump_keep_state(self.grid_start(width) + event.x * self.spp)

    def on_wheel(self, event, delta=None):
        if self.mode == ONLINE or self.settings_open:
            return
        mouse_x = self.root.winfo_pointerx() - self.wave.winfo_rootx()
        mouse_y = self.root.winfo_pointery() - self.wave.winfo_rooty()
        if not (0 <= mouse_x < self.wave.winfo_width() and 0 <= mouse_y < self.wave.winfo_height()):
            return
        delta = event.delta if delta is None else delta
        up = delta > 0
        if event.state & 0x0004:
            old = self.spp
            new = old / 1.25 if up else old * 1.25
            new = min(max(new, 2.0), self.engine.rate * 3600 * 6 / max(1, self.wave.winfo_width()))
            width = self.wave.winfo_width()
            if self.mode == SEARCH and not self.engine.cue.playing():
                t = self.view_start(width) + mouse_x * old
                self.free_view = t - mouse_x * new
            self.spp = new
        elif event.state & 0x0001:
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
            self.show_notice("OUT OF MEMORY", 5.0)
        self.peaks.update()
        if self.mode == ONLINE and not e.online:
            self.mode = SEARCH
            e.cue.locate(e.online_pos)
        if self.free_view is not None and self.mode == SEARCH and e.cue.playing():
            self.free_view = None
        if self.settings_open:
            return
        self.draw_wave()
        self.draw_band()
        self.update_labels()
        self.meters["in"].set(e.peak_in)
        self.meters["main"].set(e.peak_main)
        self.meters["cue"].set(e.peak_cue)
        e.peak_in = e.peak_main = e.peak_cue = 0.0
        if self.tick_count % 12 == 0:
            self.update_status()

    def band_content(self):
        e = self.engine
        if self.mode == ONLINE:
            return RED, "#ffffff", "ONLINE PLAY", "チャレンジ　検証中"
        if self.mode == SEARCH:
            if e.cue.playing():
                return YELLOW, YELLOW_TEXT, f"CUE　{self.effective_speed():.1f}倍速", ""
            return YELLOW, YELLOW_TEXT, "STANDBY", ""
        if e.recording:
            return GREEN, "#ffffff", "RECORDING", ""
        return "#5f6b7a", "#ffffff", "STOPPED", ""

    def draw_band(self, force=False):
        state = self.band_content()
        if state == self.band_state and not force:
            return
        self.band_state = state
        c = self.band
        c.delete("all")
        w, h = c.winfo_width(), c.winfo_height()
        if w < 10:
            return
        fill, fg, center, side = state
        round_rect(c, 1, 1, w - 2, h - 2, self.px(14), fill=fill, outline="")
        c.create_text(w / 2, h / 2, text=center, fill=fg, font=self.f_band)
        if side:
            c.create_text(w / 6, h / 2, text=side, fill=fg, font=self.f_band)
            c.create_text(w * 5 / 6, h / 2, text=side, fill=fg, font=self.f_band)

    def draw_wave(self):
        c = self.wave
        w, h = c.winfo_width(), c.winfo_height()
        if w < 10 or h < 40:
            return
        e = self.engine
        rate = e.rate
        c.delete("all")
        spp = self.spp
        # 列の境目を時間で固定し、波形はピクセル単位で送る。これで形が揺れない
        start = self.grid_start(w)
        ruler = self.px(RULER_H)
        wave_h = h - ruler
        mid = ruler + wave_h / 2
        scale = wave_h / 2 * self.v_gain
        c.create_rectangle(0, ruler, w, h, fill=EMPTY_BG, width=0)
        head = e.timeline.head
        x0 = int(round((0 - start) / spp))
        x1 = int(round((head - start) / spp))
        if x1 > 0 and x0 < w:
            c.create_rectangle(max(0, x0), ruler, min(w, x1), h, fill=WAVE_BG, width=0)
        c.create_rectangle(0, 0, w, ruler, fill=RULER_BG, width=0)
        seconds_per_px = spp / rate
        step = next((s for s in (0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 1200, 1800, 3600, 7200)
                     if s / seconds_per_px >= 110), 14400)
        t = max(0.0, np.floor(start / rate / step) * step)
        while True:
            x = int(round((t * rate - start) / spp))
            if x > w:
                break
            if x >= 0:
                c.create_line(x, 0, x, ruler, fill=MUTED)
                c.create_line(x, ruler, x, h, fill=GRID)
                total = int(round(t * 10))
                hh, rem = divmod(total, 36000)
                mm, rem = divmod(rem, 600)
                ss = rem / 10
                text = f"{hh}:{mm:02d}:{ss:02.0f}" if step >= 1 else f"{hh}:{mm:02d}:{ss:04.1f}"
                c.create_text(x + self.px(5), self.px(7), text=text, anchor="nw", fill=TEXT, font=self.f_mono)
            t += step
        c.create_line(0, mid, w, mid, fill=LINE)
        xs, mins, maxs = self.peaks.column_range(start, spp, w)
        if len(xs) >= 2:
            ytop = np.maximum(mid - maxs * scale, ruler)
            ybot = np.minimum(mid - mins * scale, h)
            ybot = np.maximum(ybot, ytop + 1)
            coords = np.empty(len(xs) * 4)
            coords[0::4] = xs
            coords[1::4] = ytop
            coords[2::4] = xs
            coords[3::4] = ybot
            c.create_line(coords.tolist(), fill=WAVE)
        for seam in e.timeline.seams:
            x = int(round((seam - start) / spp))
            if 0 <= x <= w:
                c.create_line(x, ruler, x, h, fill=HEAD, dash=(2, 4))
        left = bisect.bisect_left(self.marks, start)
        right = bisect.bisect_right(self.marks, start + w * spp)
        tri = self.px(6)
        for m in self.marks[left:right]:
            x = int(round((m - start) / spp))
            c.create_line(x, ruler, x, h, fill=MARK)
            c.create_polygon(x - tri, ruler, x + tri, ruler, x, ruler + self.px(9), fill=MARK, width=0)
        if self.mode != LIVE and 0 <= x1 <= w:
            c.create_line(x1, ruler, x1, h, fill=HEAD)
            c.create_text(x1 + self.px(6), h - self.px(8), text="LIVE", anchor="sw", fill=MUTED, font=self.f_mono)
        pos = self.position()
        x = int(round((pos - start) / spp))
        playing = self.mode == SEARCH and e.cue.playing()
        color = {LIVE: TEXT, SEARCH: CUE_LINE, ONLINE: RED}[self.mode]
        if 0 <= x <= w:
            c.create_line(x, 0, x, h, fill=color, width=self.px(2))
            if self.mode == SEARCH and not playing:
                t8 = self.px(8)
                c.create_polygon(x - t8, ruler, x + t8, ruler, x, ruler + self.px(12), fill=color, width=0)
                c.create_text(x + self.px(10), ruler + self.px(6), text="STANDBY", anchor="nw", fill=color, font=self.f_small)

    def update_labels(self):
        e = self.engine
        rate = e.rate
        self.pos_label.configure(text=fmt_clock(self.position(), rate))
        head = e.timeline.head / rate
        self.head_label.configure(text=f"{int(head // 3600):02d}:{int(head % 3600 // 60):02d}:{int(head % 60):02d}")
        self.marks_label.configure(text=str(len(self.marks)))
        if e.recording:
            self.rec_button.set_style(text="■  REC STOP", fill=RED, outline=RED, fg="#ffffff")
        else:
            self.rec_button.set_style(text="●  REC START", fill=RAISED, outline=LINE, fg=TEXT)
        online = self.mode == ONLINE
        state = "disabled" if online else "normal"
        self.rec_button.set_style(enabled=not online)
        self.settings_button.set_style(enabled=not online)
        self.pitch_button.configure(state=state)
        self.speed_scale.configure(state=state)
        playing = online or (self.mode == SEARCH and e.cue.playing())
        for key, button in self.transport.items():
            if key == "play":
                button.set_kind("stop" if playing else "play", enabled=not online)
            else:
                button.set_kind(enabled=not online)
        focused = bool(str(self.root.tk.call("focus"))) if not hasattr(self, "_test_focus") else self._test_focus()
        if focused == self.not_active_shown:
            self.not_active_shown = not focused
            if self.not_active_shown:
                self.not_active.pack(side="left", padx=self.px(12))
            else:
                self.not_active.pack_forget()
        if self.notice and time.monotonic() >= self.notice_until:
            self.notice = ""

    def update_status(self):
        e = self.engine
        ok = e.healthy()
        for key, cfg_key in (("in", "input_device"), ("out", "output_device")):
            led, item, name = self.device_rows[key]
            led.itemconfigure(item, fill=LED_OK if ok else RED)
            name.configure(text=short_device(self.cfg[cfg_key]) or "—")
        if ok == self.reconnect_shown:
            self.reconnect_shown = not ok
            if self.reconnect_shown:
                self.reconnect_button.pack(side="right", padx=(0, self.px(8)), before=self.settings_button)
            else:
                self.reconnect_button.pack_forget()
        items = self.status_items
        items["state"].configure(text="CONNECTED" if ok else "NOT CONNECTED", fg=TEXT if ok else RED_TEXT)
        items["format"].configure(text=f"{e.rate / 1000:g} kHz / 16 bit / MONO")
        items["latency"].configure(text=f"LATENCY {e.latency_ms:.0f} ms" if ok else "LATENCY —")
        items["dropouts"].configure(text=f"DROPOUTS {e.problems}", fg=RED_TEXT if e.problems else TEXT)
        used = len(e.timeline.chunks) * e.timeline.chunk * 2 / 1e9
        items["memory"].configure(text=f"MEMORY {used:.1f} GB")
        try:
            free = shutil.disk_usage(Path(self.cfg["save_dir"]).anchor or ".").free
            items["disk"].configure(text=f"DISK {free / 1e9:.0f} GB", fg=RED_TEXT if free < 2e9 else TEXT)
        except OSError:
            items["disk"].configure(text="DISK —")
        self.status_file.configure(text=e.writer.path.name if e.writer is not None else "")
        # 普段は空。知らせることがあるときだけ上に出す
        if self.notice:
            text, color = self.notice, NOTICE
        elif e.writer is not None and e.writer.error:
            text, color = e.writer.error, RED_TEXT
        elif not ok:
            text, color = (self.engine_error or "AUDIO STOPPED"), RED_TEXT
        else:
            text, color = "", MUTED
        self.status.configure(text=text, fg=color)

    def on_close(self):
        if self.mode == ONLINE:
            self.show_notice("PRESS ENTER TO STOP BEFORE CLOSING")
            return
        if self.engine.recording and not messagebox.askokcancel("REPLAY", "RECORDING. STOP AND CLOSE?"):
            return
        self.engine.shutdown()
        self.root.destroy()


def short_device(label):
    if not label:
        return ""
    name = label.split("] ", 1)[-1]
    return name if len(name) <= 38 else name[:37] + "…"


class SettingsView(tk.Frame):
    """同じウィンドウの中に開く設定画面。"""

    def __init__(self, app):
        super().__init__(app.root, bg=BG)
        self.app = app
        px = app.px
        cfg = app.cfg
        self.v_in = tk.StringVar(value=cfg["input_device"])
        self.v_out = tk.StringVar(value=cfg["output_device"])
        self.v_in_ch = tk.StringVar(value=str(cfg["input_channel"]))
        self.v_main = tk.StringVar(value=str(cfg["main_channel"]))
        self.v_cue = tk.StringVar(value=str(cfg["cue_channel"]))
        self.v_rate = tk.StringVar(value=str(cfg["samplerate"]))
        self.v_latency = tk.StringVar(value=str(cfg["latency"]).upper())
        self.v_exclusive = tk.StringVar(value="ON" if cfg.get("exclusive", True) else "OFF")
        self.v_offset = tk.StringVar(value=f"{float(cfg.get('mark_offset', 2.0)):g}")
        self.v_dir = tk.StringVar(value=cfg["save_dir"])
        self.load_devices()

        header = tk.Frame(self, bg=BG)
        header.pack(fill="x", pady=(0, px(14)))
        tk.Label(header, text="設定", font=app.f_title, fg=TEXT, bg=BG).pack(side="left")
        RoundButton(app, header, text="BACK  Esc", on_click=self.back, width=112).pack(side="right")

        panels = tk.Frame(self, bg=BG)
        panels.pack(fill="both", expand=True)
        panels.columnconfigure((0, 1), weight=1, uniform="panel")
        left_panel = RoundPanel(app, panels, pad=20)
        left_panel.grid(row=0, column=0, sticky="new", padx=(0, px(7)))
        right_panel = RoundPanel(app, panels, pad=20)
        right_panel.grid(row=0, column=1, sticky="new", padx=(px(7), 0))
        left, right = left_panel.inner, right_panel.inner
        tk.Label(left, text="INPUT", font=app.f_mid, fg=TEXT, bg=PANEL).pack(anchor="w")
        self.c_in = self.field(left, "デバイス", self.v_in, [d[1] for d in self.ins])
        self.c_in_ch = self.field(left, "CH", self.v_in_ch, [])
        rate_box = self.field(left, "SAMPLE RATE", self.v_rate, ["44100", "48000", "96000"])
        if app.engine.timeline.head or app.engine.writer is not None:
            rate_box.configure(state="disabled")
        self.field(left, "MARK OFFSET (SEC)", self.v_offset, ["0", "1", "2", "3", "5", "10"])
        tk.Label(left, text="SAVE FOLDER", font=app.f_small, fg=MUTED, bg=PANEL).pack(anchor="w", pady=(px(12), px(4)))
        folder = tk.Frame(left, bg=PANEL)
        folder.pack(fill="x")
        tk.Entry(folder, textvariable=self.v_dir, font=app.f_small, relief="flat", highlightthickness=1,
                 highlightbackground=LINE, highlightcolor=WAVE).pack(side="left", fill="x", expand=True, ipady=px(8))
        RoundButton(app, folder, text="BROWSE", on_click=self.pick_dir, width=92, height=38, radius=12, parent_bg=PANEL).pack(side="left", padx=(px(8), 0))

        tk.Label(right, text="OUTPUT", font=app.f_mid, fg=TEXT, bg=PANEL).pack(anchor="w")
        self.c_out = self.field(right, "デバイス", self.v_out, [d[1] for d in self.outs])
        self.c_main = self.field(right, "MAIN CH", self.v_main, [])
        self.c_cue = self.field(right, "CUE CH", self.v_cue, [])
        self.field(right, "LATENCY", self.v_latency, ["LOW", "HIGH"])
        self.field(right, "EXCLUSIVE (WASAPI)", self.v_exclusive, ["ON", "OFF"])
        self.c_in.bind("<<ComboboxSelected>>", lambda _e: self.update_channels(), add="+")
        self.c_out.bind("<<ComboboxSelected>>", lambda _e: self.update_channels(), add="+")

        footer = tk.Frame(self, bg=BG)
        footer.pack(fill="x", pady=(px(14), 0))
        RoundButton(app, footer, text="APPLY", on_click=self.apply, width=132, height=48, radius=24, fill=WAVE, outline=WAVE, fg="#ffffff", font=app.f_mid).pack(side="right")
        RoundButton(app, footer, text="RESCAN", on_click=self.rescan, width=112, height=48, radius=24).pack(side="right", padx=px(10))
        self.update_channels()

    def load_devices(self):
        self.ins = list_devices("in")
        self.outs = list_devices("out")

    def field(self, parent, label, var, values):
        tk.Label(parent, text=label, font=self.app.f_small, fg=MUTED, bg=PANEL).pack(anchor="w", pady=(self.app.px(12), self.app.px(4)))
        box = ttk.Combobox(parent, textvariable=var, values=values, state="readonly", style="Light.TCombobox", font=self.app.f_small)
        box.pack(fill="x")
        box.bind("<<ComboboxSelected>>", lambda _e: self.focus_set())
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
                   input_channel=int(self.v_in_ch.get()), main_channel=int(self.v_main.get()),
                   cue_channel=int(self.v_cue.get()), samplerate=int(self.v_rate.get()),
                   latency=self.v_latency.get().lower(), exclusive=self.v_exclusive.get() == "ON",
                   mark_offset=float(self.v_offset.get()), save_dir=self.v_dir.get())
        self.app.close_settings(cfg)


def self_test():
    """自動ビルドで、exeが部品込みで動くことを確かめる。"""
    report = APP_DIR / "selftest.txt"
    try:
        apis = [a["name"] for a in sd.query_hostapis()]
        if "ASIO" in apis:
            raise RuntimeError(f"ASIOの部品が含まれています: {apis}")
        if "Windows WASAPI" not in apis:
            raise RuntimeError(f"WASAPIが使えません: {apis}")
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
