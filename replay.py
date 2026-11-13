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

APP_NAME = "チャレンジANN"
VERSION = "0.3"
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

    STEP = 4_000_000  # 1回に作る量の上限。長いファイルを読み込んだ直後でも画面が固まらないように

    def update(self):
        bins = min(self.t.head - self.done, self.STEP) // self.BIN0
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
        self.vu_sum = 0.0  # VU用。入力の絶対値の合計と個数
        self.vu_n = 0
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
            a = np.abs(x)
            peak = float(a.max())
            if peak > self.peak_in:
                self.peak_in = peak
            self.vu_sum += float(a.sum())
            self.vu_n += len(x)
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

    def replace_timeline(self, timeline):
        """録音していないときに、録音データを丸ごと差し替える（新規・読み込み）。"""
        self.recording = False
        self.online = False
        old = self.cue
        cue = CueRenderer(timeline)
        cue.speed, cue.keep_pitch = old.speed, old.keep_pitch
        self.cue = cue
        self.timeline = timeline
        old.quit = True
        old.need.set()
        if self.writer is not None:
            self.writer.close()
            self.writer = None

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
        lambda: w.winmm.timeBeginPeriod(1),
        lambda: w.kernel32.SetPriorityClass(w.kernel32.GetCurrentProcess(), 0x00000080),  # HIGH_PRIORITY_CLASS
        lambda: w.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001),  # スリープ抑止
    ):
        try:
            call()
        except Exception:
            pass


# ------------------------------------------------------ 読み込み・セッション -------
SESSION_VERSION = 1


def open_wav(path, block_frames=1 << 18):
    """WAVを開いて (サンプルレート, フレーム数, 読み出し用のジェネレーター) を返す。
    ジェネレーターは (モノラルのfloat32, 読んだ割合) を順に返す。2ch以上は平均してモノラルにする。"""
    f = open(path, "rb")
    try:
        head = f.read(12)
        if len(head) < 12 or head[:4] != b"RIFF" or head[8:12] != b"WAVE":
            raise ValueError("WAVファイルではありません")
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(12)
        fmt = None
        data_off = data_len = None
        while f.tell() + 8 <= size:
            cid, clen = struct.unpack("<4sI", f.read(8))
            start = f.tell()
            if cid == b"fmt ":
                raw = f.read(clen)
                tag, ch, rate, _bytes_per_sec, _align, bits = struct.unpack("<HHIIHH", raw[:16])
                if tag == 0xFFFE and len(raw) >= 26:  # WAVE_FORMAT_EXTENSIBLE
                    tag = struct.unpack("<H", raw[24:26])[0]
                fmt = (tag, ch, rate, bits)
            elif cid == b"data":
                data_off = start
                data_len = clen if 0 < clen <= size - start else size - start
                if fmt is not None:
                    break
            f.seek(start + clen + (clen & 1))
        if fmt is None or data_off is None:
            raise ValueError("WAVの中身が読めません")
        tag, ch, rate, bits = fmt
        if not ((tag == 1 and bits in (8, 16, 24, 32)) or (tag == 3 and bits in (32, 64))) or ch < 1 or rate < 1000:
            raise ValueError(f"対応していないWAVです（形式 {tag}、{bits}bit、{ch}ch）")
    except Exception:
        f.close()
        raise
    width = bits // 8
    align = width * ch
    frames = data_len // align

    def blocks():
        try:
            f.seek(data_off)
            done = 0
            while done < frames:
                raw = f.read(min(block_frames, frames - done) * align)
                n = len(raw) // align
                if n == 0:
                    break
                b = np.frombuffer(raw, dtype=np.uint8, count=n * align).reshape(n, align)
                if tag == 3:
                    x = b.view("<f4" if width == 4 else "<f8").astype(np.float32)
                elif width == 1:
                    x = (b.astype(np.float32) - 128.0) / 128.0
                elif width == 2:
                    x = b.view("<i2").astype(np.float32) / 32768.0
                elif width == 3:
                    t = b.reshape(n, ch, 3).astype(np.int32)
                    v = t[..., 0] | (t[..., 1] << 8) | (t[..., 2] << 16)
                    x = (np.where(v >= 1 << 23, v - (1 << 24), v)).astype(np.float32) / 8388608.0
                else:
                    x = b.view("<i4").astype(np.float32) / 2147483648.0
                x = x.reshape(n, ch)
                mono = x.mean(axis=1) if ch > 1 else x[:, 0]
                done += n
                yield np.ascontiguousarray(mono, dtype=np.float32), done / max(1, frames)
        finally:
            f.close()

    return rate, frames, blocks()


class Resampler:
    """サンプルレートの変換（窓つきsinc）。少しずつ流し込める。"""

    HALF = 16
    PHASES = 512

    def __init__(self, src_rate, dst_rate):
        self.step = src_rate / dst_rate
        cutoff = min(1.0, dst_rate / src_rate) * 0.94
        h = self.HALF
        self.offsets = np.arange(-h + 1, h + 1)
        frac = np.arange(self.PHASES)[:, None] / self.PHASES
        x = self.offsets[None, :] - frac
        win = np.i0(8.0 * np.sqrt(np.clip(1 - (x / h) ** 2, 0, 1))) / np.i0(8.0)
        table = cutoff * np.sinc(cutoff * x) * win
        self.table = (table / table.sum(axis=1, keepdims=True)).astype(np.float32)
        self.buf = np.zeros(h, dtype=np.float32)
        self.buf_start = -h
        self.t = 0.0

    def process(self, x, final=False):
        self.buf = np.concatenate((self.buf, np.asarray(x, dtype=np.float32)))
        if final:
            self.buf = np.concatenate((self.buf, np.zeros(self.HALF + 2, dtype=np.float32)))
        limit = self.buf_start + len(self.buf) - self.HALF - 1
        if limit < self.t:
            return np.zeros(0, dtype=np.float32)
        n = int(np.floor((limit - self.t) / self.step)) + 1
        out = np.empty(n, dtype=np.float32)
        for s in range(0, n, 32768):
            k = np.arange(s, min(n, s + 32768))
            ts = self.t + k * self.step
            i0 = np.floor(ts).astype(np.int64)
            ph = np.minimum(((ts - i0) * self.PHASES + 0.5).astype(np.int64), self.PHASES - 1)
            idx = (i0 - self.buf_start)[:, None] + self.offsets[None, :]
            out[s:s + len(k)] = np.einsum("ij,ij->i", self.buf[idx], self.table[ph])
        self.t += n * self.step
        keep = int(np.floor(self.t)) - self.HALF + 1 - self.buf_start
        if keep > 0:
            self.buf = self.buf[keep:]
            self.buf_start += keep
        return out


def load_timeline(path, rate, progress=None, cancel=None):
    """WAVを読んで、指定のサンプルレートの Timeline を作る。別スレッドから呼ぶ。"""
    src_rate, _frames, blocks = open_wav(path)
    rs = Resampler(src_rate, rate) if src_rate != rate else None
    tl = Timeline(rate)
    for mono, frac in blocks:
        if cancel is not None and cancel():
            return None
        y = rs.process(mono) if rs is not None else mono
        if len(y):
            tl.write(to_int16(y))
        if progress is not None:
            progress(frac)
    if rs is not None:
        tail = rs.process(np.zeros(0, dtype=np.float32), final=True)
        if len(tail):
            tl.write(to_int16(tail))
    return tl


def write_json(path, data):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def read_session(path):
    """セッションを読んで (音声のパス, [(秒, メモ)], 設定) を返す。"""
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "marks" not in data:
        raise ValueError("チャレンジANN（チャレンジREC）のセッションではありません")
    audio = None
    for cand in (data.get("audio_relative"), data.get("audio")):
        if cand:
            p = Path(cand)
            if not p.is_absolute():
                p = path.parent / p
            if p.exists():
                audio = p
                break
    if audio is None:
        raise ValueError(f"音声ファイルが見つかりません: {data.get('audio') or '（なし）'}")
    marks = []
    for m in data.get("marks", []):
        try:
            marks.append((float(m["time"]), str(m.get("memo", ""))))
        except (KeyError, TypeError, ValueError):
            continue
    return audio, marks, data


def ime_set(widget, enabled):
    """Windowsの日本語入力を、この部品だけオン／オフにする。"""
    if sys.platform != "win32":
        return
    try:
        imm = ctypes.windll.imm32
        user = ctypes.windll.user32
        imm.ImmAssociateContextEx.argtypes = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint)
        user.GetParent.restype = ctypes.c_void_p
        user.GetParent.argtypes = (ctypes.c_void_p,)
        hwnd = widget.winfo_id()
        targets = [hwnd]
        if widget is widget.winfo_toplevel():
            parent = user.GetParent(hwnd)
            if parent:
                targets.append(parent)
        for h in targets:
            imm.ImmAssociateContextEx(h, None, 0x10 if enabled else 0)  # 0x10 = IACE_DEFAULT
    except Exception:
        pass


# -------------------------------------------------------------- 画面 -------
BG = "#15171A"
PANEL = "#1D2024"
PANEL2 = "#2A2D33"
LINE = "#2E3238"
LINE2 = "#3A3F47"
TEXT = "#ECE9E3"
MUTED = "#9AA0A8"
DIM = "#7E848D"
FAINT = "#5C626B"
TEAL = "#4FC1B4"
TEAL_TEXT = "#0B1413"
TEAL_SOFT = "#8FE0D5"
TEAL_DARK = "#16211F"
YELLOW = "#FFD27A"
AMBER = "#E8A33D"
RED = "#D9443F"
RED_SOFT = "#ff8a85"
RED_CURSOR = "#ff4b45"
MARK = "#8EC9FF"
MARK_TEXT = "#B9DCFF"
LANE = "#0F1113"
LANE_EMPTY = "#191B1F"
RULER_BG = "#121418"
GRID = "#1E2125"
SEG_OFF = PANEL2
RASTA_RED = "#d6322f"
RASTA_YELLOW = "#f2c230"
RASTA_GREEN = "#1f8a45"
RULER_H = 22
SPEEDS = (1.0, 1.2, 1.4, 1.5, 1.6, 1.8, 2.0)
PEAK_LED_DB = -1.0


def round_rect(canvas, x0, y0, x1, y1, r, **kw):
    r = max(0.0, min(r, (x1 - x0) / 2, (y1 - y0) / 2))
    pts = (x0 + r, y0, x1 - r, y0, x1, y0, x1, y0 + r, x1, y1 - r, x1, y1,
           x1 - r, y1, x0 + r, y1, x0, y1, x0, y1 - r, x0, y0 + r, x0, y0)
    return canvas.create_polygon(pts, smooth=True, splinesteps=12, **kw)


def mix(c1, c2, t):
    """2つの色の間の色。t=0でc1、t=1でc2。"""
    a = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(c2[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{int(round(x + (y - x) * t)):02x}" for x, y in zip(a, b))


def raised(canvas, x0, y0, x1, y1, r, top, bottom, outline, pressed=False, oval=False):
    """立体のボタンの地を描く。上が明るく下が暗いグラデーションと、下の影。
    押している間は影を消して2ピクセル沈め、明暗を逆にする。文字を下げる量を返す。"""
    sink = 2 if pressed else 0
    if not pressed:
        if oval:
            canvas.create_oval(x0, y0 + 2, x1, y1 + 2, fill="#0B0C0E", outline="")
        else:
            round_rect(canvas, x0, y0 + 2, x1, y1 + 2, r, fill="#0B0C0E", outline="")
    y0 += sink
    y1 += sink
    if pressed:
        top, bottom = mix(bottom, "#000000", 0.25), mix(top, "#000000", 0.2)
    h = int(y1 - y0)
    rr = (x1 - x0) / 2 if oval else min(r, (x1 - x0) / 2, (y1 - y0) / 2)
    for i in range(h):
        if oval:
            d = abs(i + 0.5 - h / 2)
            half = np.sqrt(max(0.0, (h / 2) ** 2 - d * d)) * (x1 - x0) / h
            xa, xb = (x0 + x1) / 2 - half, (x0 + x1) / 2 + half
            t = min(1.0, np.hypot((x0 + x1) / 2 - (x0 + (x1 - x0) * 0.35), i - h * 0.3) / h * 1.4)
            color = mix(top, bottom, t)
        else:
            if i < rr:
                d = rr - i - 0.5
            elif i > h - rr:
                d = i + 0.5 - (h - rr)
            else:
                d = 0.0
            inset = rr - np.sqrt(max(0.0, rr * rr - d * d)) if d else 0.0
            xa, xb = x0 + inset, x1 - inset
            color = mix(top, bottom, i / max(1, h - 1))
        if xb > xa:
            canvas.create_line(xa, y0 + i, xb, y0 + i, fill=color)
    if oval:
        canvas.create_oval(x0, y0, x1, y1, outline=outline)
    else:
        round_rect(canvas, x0, y0, x1, y1, r, fill="", outline=outline)
        if not pressed:
            canvas.create_line(x0 + rr, y0 + 1, x1 - rr, y0 + 1, fill=mix(top, "#ffffff", 0.18))
    return sink


def clip_text(font, text, width):
    """幅に収まらない文字は「…」で切る。"""
    if font.measure(text) <= width:
        return text
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if font.measure(text[:mid] + "…") <= width:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo] + "…"


class RoundPanel(tk.Canvas):
    """角の丸い枠。中身は .inner に入れる。"""

    def __init__(self, app, parent, fill=PANEL, pad=12, radius=10, parent_bg=BG, outline=LINE):
        super().__init__(parent, bg=parent_bg, highlightthickness=0, takefocus=0)
        self.pad = app.px(pad)
        self.radius = app.px(radius)
        self.fill = fill
        self.outline = outline
        self.inner = tk.Frame(self, bg=fill)
        self.window = self.create_window(self.pad, self.pad, anchor="nw", window=self.inner)
        self.bind("<Configure>", self._layout)
        self.inner.bind("<Configure>", self._fit)

    def _fit(self, _event=None):
        need = self.inner.winfo_reqheight() + 2 * self.pad
        if int(self.cget("height")) != need:
            self.configure(height=need)

    def _layout(self, _event=None):
        w, h = self.winfo_width(), self.winfo_height()
        self.delete("bg")
        round_rect(self, 1, 1, w - 2, h - 2, self.radius, fill=self.fill, outline=self.outline, tags="bg")
        self.tag_lower("bg")
        self.itemconfigure(self.window, width=max(1, w - 2 * self.pad), height=max(1, h - 2 * self.pad))


class Btn(tk.Canvas):
    """角の丸いボタン。style で色を選ぶ。使えないときは薄く表示する。"""

    STYLES = {  # 上の色、下の色、ふち、文字
        "normal": ("#3A3F47", "#24272C", "#4A505A", TEXT),
        "teal": ("#24403C", "#15201E", "#3B6A63", "#A9E3DA"),
        "primary": ("#7FDCD0", "#45AFA3", "#8FE6DA", TEAL_TEXT),
        "play": ("#8FE6DA", "#3A9C91", "#4FC1B4", "#0E2A27"),
        "off": ("#2A2D33", "#1F2226", LINE2, DIM),
    }
    DISABLED = (PANEL, PANEL, PANEL2, FAINT)

    def __init__(self, app, parent, text="", on_click=None, width=None, height=30, radius=7, font=None,
                 style="normal", parent_bg=PANEL, kind=None, pad=11, on_disabled=None):
        self.app = app
        self.font = font or app.f_small
        if width is None:
            w = tkfont.Font(font=self.font).measure(text) + 2 * app.px(pad)
        else:
            w = app.px(width)
        super().__init__(parent, width=w, height=app.px(height), bg=parent_bg, highlightthickness=0, takefocus=0, cursor="hand2")
        self.text = text
        self.radius = app.px(radius)
        self.style = style
        self.kind = kind
        self.enabled = True
        self.on_click = on_click
        self.on_disabled = on_disabled
        self.pressed = False
        self.bind("<Configure>", lambda _e: self.redraw())
        self.bind("<ButtonPress-1>", self._press)
        self.bind("<ButtonRelease-1>", self._release)

    def _press(self, _event):
        if self.enabled:
            self.pressed = True
            self.redraw()

    def _release(self, event):
        if self.pressed:
            self.pressed = False
            self.redraw()
        if not (0 <= event.x < self.winfo_width() and 0 <= event.y < self.winfo_height()):
            return
        if self.enabled:
            if self.on_click:
                self.on_click()
        elif self.on_disabled:
            self.on_disabled()

    def set(self, text=None, style=None, kind=None, enabled=None):
        changed = False
        for name, value in (("text", text), ("style", style), ("kind", kind), ("enabled", enabled)):
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
        top, bottom, outline, fg = self.STYLES[self.style] if self.enabled else self.DISABLED
        if not self.enabled and self.kind not in ("play", "stop"):
            round_rect(self, 1, 1, w - 2, h - 4, self.radius, fill=top, outline=outline)
            self.create_text(w / 2, (h - 3) / 2, text=self.text, fill=fg, font=self.font)
            return
        if self.kind in ("play", "stop"):
            d = min(w, h - 3) - 2
            x0 = (w - d) / 2
            if self.enabled:
                dy = raised(self, x0, 1, x0 + d, 1 + d, 0, top, bottom, outline, self.pressed, oval=True)
            else:
                self.create_oval(x0, 1, x0 + d, 1 + d, fill=top, outline=outline)
                dy = 0
            cx, cy, s = w / 2, 1 + d / 2 + dy, d * 0.2
            if self.kind == "play":
                self.create_polygon(cx - s * 0.75, cy - s, cx - s * 0.75, cy + s, cx + s * 1.05, cy, fill=fg, outline="")
            else:
                self.create_rectangle(cx - s * 0.8, cy - s * 0.8, cx + s * 0.8, cy + s * 0.8, fill=fg, outline="")
            return
        dy = raised(self, 1, 1, w - 2, h - 4, self.radius, top, bottom, outline, self.pressed)
        self.create_text(w / 2, (h - 3) / 2 + dy, text=self.text, fill=fg, font=self.font)


class KeyCap(tk.Canvas):
    def __init__(self, app, parent, text, parent_bg=BG):
        font = tkfont.Font(font=app.f_key)
        w, h = font.measure(text) + app.px(12), app.px(18)
        super().__init__(parent, width=w, height=h, bg=parent_bg, highlightthickness=0, takefocus=0)
        r = app.px(4)
        round_rect(self, 1, 1, w - 2, h - 1, r, fill=LINE2, outline="")
        round_rect(self, 1, 1, w - 2, h - 3, r, fill="#22252A", outline=LINE2)
        self.create_text(w / 2, (h - 2) / 2, text=text, fill=TEXT, font=app.f_key)


class EnterBlock(tk.Canvas):
    """ENTER（ONLINE PLAY）。クリックしても Enter キーと同じに動く。"""

    def __init__(self, app, parent):
        super().__init__(parent, width=app.px(108), height=app.px(70), bg=BG, highlightthickness=0, takefocus=0, cursor="hand2")
        self.app = app
        self.state = None
        self.pressed = False
        self.bind("<Configure>", lambda _e: self.redraw(force=True))
        self.bind("<ButtonPress-1>", self._press)
        self.bind("<ButtonRelease-1>", self._release)

    def _press(self, _event):
        if self.state != "memo":
            self.pressed = True
            self.redraw(force=True)

    def _release(self, event):
        if self.pressed:
            self.pressed = False
            self.redraw(force=True)
        if 0 <= event.x < self.winfo_width() and 0 <= event.y < self.winfo_height():
            self.app.enter_click()

    def redraw(self, state=None, force=False):
        state = state or self.state or "normal"
        if state == self.state and not force:
            return
        self.state = state
        self.delete("all")
        app, px = self.app, self.app.px
        w, h = self.winfo_width(), self.winfo_height()
        if w < 10:
            return
        if state == "memo":
            round_rect(self, 1, 1, w - 2, h - 2, px(10), fill=PANEL, outline=TEAL, dash=(4, 3))
            self.create_text(w / 2, h * 0.27, text="メモ入力中", fill=TEAL_SOFT, font=app.f_body_b)
            y = h * 0.5
            t = self.create_text(w / 2, y, text="ONLINE PLAY", fill=DIM, font=app.f_mono)
            x0, _y0, x1, _y1 = self.bbox(t)
            self.create_line(x0, y, x1, y, fill=DIM)
            self.create_text(w / 2, h * 0.74, text="いまは出せません\nEnter で確定", fill=MUTED, font=app.f_tiny, justify="center")
            return
        online = state == "online"
        top, bottom, outline, fg = ("#EC5A55", "#C23733", RED_SOFT, "#ffffff") if online else ("#24403C", "#121A19", TEAL, "#A9E3DA")
        dy = raised(self, 1, 1, w - 2, h - 4, px(10), top, bottom, outline, self.pressed)
        self.create_text(w - px(9), px(9) + dy, text="ENTER", anchor="ne", fill=fg, font=app.f_tiny_mono)
        self.create_text(w / 2, h * 0.41 + dy, text="ONLINE", fill=fg, font=app.f_enter)
        self.create_text(w / 2, h * 0.66 + dy, text="STOP" if online else "PLAY", fill=fg, font=app.f_enter)


class VUMeter(tk.Canvas):
    """アナログのVU。目盛りと針の動きは自作のVUメーター（meter）と同じ。0VU = -20dBFS。"""

    SCALE = ((-20, 0.030), (-10, 0.135), (-7, 0.222), (-5, 0.303), (-3, 0.418), (-2, 0.489),
             (-1, 0.567), (0, 0.654), (1, 0.765), (2, 0.888), (3, 1.000))
    START, END = 133.0, 47.0
    RISE, ZETA = 0.300, 0.81

    def __init__(self, app, parent):
        self.s = app.px(150) / 280
        super().__init__(parent, width=app.px(150), height=int(150 * self.s), bg=PANEL, highlightthickness=0, takefocus=0)
        self.app = app
        self.pts = [(0.0, 0.0)] + [(10 ** (db / 20), f) for db, f in self.SCALE]
        z = self.ZETA
        wd = np.sqrt(1 - z * z)
        x = 0.0
        while x < 50:
            if 1 - np.exp(-z * x) * (np.cos(wd * x) + z / wd * np.sin(wd * x)) >= 0.99:
                break
            x += 0.0005
        self.wn = x / self.RISE
        self.y = 0.0
        self.v = 0.0
        self.needle = None
        self.draw_base()

    def frac(self, v):
        if not v > 0:
            return 0.0
        for (a0, b0), (a1, b1) in zip(self.pts, self.pts[1:]):
            if v <= a1:
                return b0 + (b1 - b0) * (v - a0) / (a1 - a0)
        return 1.0

    def angle(self, db):
        return self.START + self.frac(10 ** (db / 20)) * (self.END - self.START)

    def pt(self, deg, r):
        s = self.s
        a = np.radians(deg)
        return 140 * s + np.cos(a) * r * s, 181 * s - np.sin(a) * r * s

    def draw_base(self):
        s, app = self.s, self.app
        w, h = int(280 * s), int(150 * s)
        self.create_rectangle(0, 0, w, h, fill="#fdfae1", outline="")
        cx, cy, r = 140 * s, 181 * s, 140 * s
        a1 = self.angle(0)
        self.create_arc(cx - r, cy - r, cx + r, cy + r, start=a1, extent=self.START - a1, style="arc", outline="#222", width=max(1, s * 1.5))
        r2 = (140 + 2.5) * s
        a3 = self.angle(3)
        self.create_arc(cx - r2, cy - r2, cx + r2, cy + r2, start=a3, extent=a1 - a3, style="arc", outline="#d00", width=max(2, 5 * s))
        for db, _f in self.SCALE:
            deg = self.angle(db)
            color = "#d00" if db >= 0 else "#222"
            x0, y0 = self.pt(deg, 140)
            x1, y1 = self.pt(deg, 140 + (13 if db >= 0 else 10))
            self.create_line(x0, y0, x1, y1, fill=color, width=max(1, 2.2 * s))
            tx, ty = self.pt(deg, 166)
            self.create_text(min(w - 6, max(6, tx)), ty + 2, text=str(abs(db)), fill=color, font=app.f_vu)
        tx, ty = self.pt(self.angle(-20), 126)
        self.create_text(tx + 4, ty + 3, text="−", fill="#222", font=app.f_vu)
        tx, ty = self.pt(self.angle(3), 126)
        self.create_text(tx - 4, ty + 3, text="+", fill="#d00", font=app.f_vu)
        self.create_text(w / 2, h - app.px(17), text="VU", fill="#222", font=app.f_vu_big)
        self.create_text(w / 2, h - app.px(6), text="0VU = -20dBFS", fill="#8a8678", font=app.f_tiny_mono)
        self.needle = self.create_line(cx, cy, cx, cy, fill="#111", width=max(1, app.px(1.4)))
        self.set_needle(self.START)

    def set_needle(self, deg):
        x0, y0 = self.pt(deg, 0)
        x1, y1 = self.pt(deg, 150)
        self.coords(self.needle, x0, y0, x1, y1)

    def step(self, mean_abs, dt):
        dt = min(max(dt, 0.0), 0.2)
        steps = max(1, int(np.ceil(dt / 0.001)))
        hstep = dt / steps
        wn, z = self.wn, self.ZETA
        for _ in range(steps):
            self.v += (wn * wn * (mean_abs - self.y) - 2 * z * wn * self.v) * hstep
            self.y += self.v * hstep
        if self.y < 0:
            self.y = 0.0
            self.v = 0.0
        db = max(-200.0, min(3.0, 20 * np.log10(max(self.y, 1e-5)) + 23.8))
        self.set_needle(self.angle(db))


class MeterBlock(tk.Canvas):
    """INPUT / MAIN L / CUE R の横のバー。-48〜0dBFS を24コマで。"""

    COUNT = 24
    ROWS = (("in", "INPUT", RASTA_RED, "#ffffff"), ("main", "MAIN L", RASTA_YELLOW, "#241d00"), ("cue", "CUE R", RASTA_GREEN, "#ffffff"))

    def __init__(self, app, parent):
        px = app.px
        self.row_h, self.gap, self.label_w = px(10), px(8), px(52)
        super().__init__(parent, width=px(300), height=3 * px(10) + 3 * px(8) + px(10), bg=PANEL, highlightthickness=0, takefocus=0)
        self.app = app
        self.values = {k: 0.0 for k, *_ in self.ROWS}
        self.lit = {k: -1 for k, *_ in self.ROWS}
        self.items = {}
        self.bind("<Configure>", lambda _e: self.build())

    def build(self):
        self.delete("all")
        app, px = self.app, self.app.px
        w = self.winfo_width()
        x0 = self.label_w + px(8)
        seg_gap = px(2)
        seg = (w - x0 - seg_gap * (self.COUNT - 1)) / self.COUNT
        for r, (key, text, bg, fg) in enumerate(self.ROWS):
            y = r * (self.row_h + self.gap)
            self.create_rectangle(0, y - px(2), self.label_w, y + self.row_h + px(2), fill=bg, outline="")
            self.create_text(self.label_w / 2, y + self.row_h / 2, text=text, fill=fg, font=app.f_tiny_mono_b)
            items = []
            for i in range(self.COUNT):
                x = x0 + i * (seg + seg_gap)
                items.append(self.create_rectangle(x, y, x + seg, y + self.row_h, fill=SEG_OFF, outline=""))
            self.items[key] = items
            self.lit[key] = -1
        y = 3 * (self.row_h + self.gap) - px(2)
        span = w - x0
        for db in (-48, -24, -12, -6, 0):
            x = x0 + span * (db + 48) / 48
            anchor = "nw" if db == -48 else "ne" if db == 0 else "n"
            self.create_text(x, y, text=str(db), anchor=anchor, fill=DIM, font=app.f_tiny_mono)

    def set(self, key, peak):
        db = 20 * np.log10(peak) if peak > 1e-6 else -120.0
        target = min(1.0, max(0.0, (db + 48) / 48))
        v = self.values[key]
        v = target if target > v else max(target, v - 0.025)
        self.values[key] = v
        lit = int(round(v * self.COUNT))
        items = self.items.get(key)
        if lit == self.lit[key] or not items:
            return
        for i, item in enumerate(items):
            zone = TEAL if i < 17 else AMBER if i < 21 else RED
            self.itemconfigure(item, fill=zone if i < lit else SEG_OFF)
        self.lit[key] = lit


class SpeedPills(tk.Canvas):
    """検聴速度。押した速度に切り替わる。"""

    def __init__(self, app, parent, var, command):
        super().__init__(parent, width=app.px(250), height=app.px(24), bg=PANEL, highlightthickness=0, takefocus=0, cursor="hand2")
        self.app = app
        self.var = var
        self.command = command
        self.enabled = True
        self.bind("<Configure>", lambda _e: self.redraw())
        self.bind("<ButtonRelease-1>", self._release)

    def centers(self):
        w = self.winfo_width()
        pill = self.app.px(28)
        return [pill / 2 + i * (w - pill) / (len(SPEEDS) - 1) for i in range(len(SPEEDS))]

    def _release(self, event):
        if not self.enabled:
            return
        cs = self.centers()
        i = min(range(len(cs)), key=lambda k: abs(cs[k] - event.x))
        self.var.set(SPEEDS[i])
        self.redraw()
        self.command()

    def set_enabled(self, enabled):
        if enabled != self.enabled:
            self.enabled = enabled
            self.redraw()

    def redraw(self):
        self.delete("all")
        app, px = self.app, self.app.px
        h = self.winfo_height()
        cs = self.centers()
        self.create_line(cs[0], h / 2, cs[-1], h / 2, fill=LINE2, width=px(2))
        current = round(float(self.var.get()), 1)
        half_w, half_h = px(14), px(10)
        for c, v in zip(cs, SPEEDS):
            on = abs(v - current) < 0.05
            y0, y1 = h / 2 - half_h - 1, h / 2 + half_h - 2
            if not self.enabled:
                round_rect(self, c - half_w, y0, c + half_w, y1, px(5), fill=LINE2 if on else PANEL, outline="")
                self.create_text(c, (y0 + y1) / 2, text=f"{v:.1f}", fill=MUTED if on else FAINT, font=app.f_mono11_b if on else app.f_mono11)
                continue
            if on:
                top, bottom, outline, fg, font = "#7FDCD0", "#45AFA3", "#8FE6DA", TEAL_TEXT, app.f_mono11_b
            else:
                top, bottom, outline, fg, font = "#33373E", "#24272C", "#3E434B", "#B4BAC2", app.f_mono11
            raised(self, c - half_w, y0, c + half_w, y1, px(5), top, bottom, outline)
            self.create_text(c, (y0 + y1) / 2, text=f"{v:.1f}", fill=fg, font=font)


class MarkList(tk.Canvas):
    """右のマーク一覧。古い順に上から並び、いちばん新しいものが下。"""

    def __init__(self, app, parent):
        super().__init__(parent, bg=PANEL, highlightthickness=0, takefocus=0)
        self.app = app
        px = app.px
        self.row_h = px(30)
        self.x_num = px(34)
        self.x_time = px(42)
        self.x_memo = px(112)
        self.top = 0
        self.follow = True
        self.sig = None
        self.f_memo = tkfont.Font(font=app.f_body)
        self.bind("<Configure>", lambda _e: self.redraw(force=True))
        self.bind("<ButtonRelease-1>", self._click)
        self.bind("<Double-Button-1>", self._double)

    def visible(self):
        return max(1, self.winfo_height() // self.row_h)

    def max_top(self):
        return max(0, len(self.app.marks) - self.visible())

    def scroll(self, rows):
        if self.app.editing is not None:
            return
        self.top = min(max(0, self.top + rows), self.max_top())
        self.follow = self.top >= self.max_top()
        self.redraw()

    def ensure_visible(self, i):
        vis = self.visible()
        if i < self.top:
            self.top = i
        elif i >= self.top + vis:
            self.top = i - vis + 1
        self.top = min(max(0, self.top), self.max_top())
        self.follow = self.top >= self.max_top()

    def y0(self):
        n = len(self.app.marks)
        vis = self.visible()
        return self.winfo_height() - min(n, vis) * self.row_h

    def row_at(self, y):
        k = int((y - self.y0()) // self.row_h)
        i = self.top + k
        if y < self.y0() or not 0 <= i < len(self.app.marks) or k >= self.visible():
            return None
        return i

    def memo_box(self, i):
        y = self.y0() + (i - self.top) * self.row_h
        px = self.app.px
        return self.x_memo, y + px(4), self.winfo_width() - px(10), y + self.row_h - px(4)

    def _click(self, event):
        i = self.row_at(event.y)
        if i is not None and event.x < self.x_memo:
            self.app.jump_to_mark(i)

    def _double(self, event):
        i = self.row_at(event.y)
        if i is not None and event.x >= self.x_memo:
            self.app.begin_memo(i)

    def redraw(self, force=False):
        app = self.app
        w, h = self.winfo_width(), self.winfo_height()
        if w < 20 or h < 10:
            return
        if app.editing is None:
            if self.follow:
                self.top = self.max_top()
            self.top = min(self.top, self.max_top())
        cur = app.highlight_mark()
        sig = (app.list_version, self.top, cur, app.mode == ONLINE, app.editing, w, h)
        if sig == self.sig and not force:
            return
        self.sig = sig
        self.delete("rows")
        px = app.px
        marks = app.marks
        rate = app.engine.rate
        if not marks:
            self.create_text(w / 2, h / 2, text="マークはまだありません\nA–Z キーで打てます", fill=DIM, font=app.f_small, justify="center", tags="rows")
            return
        y0 = self.y0()
        for k in range(min(self.visible(), len(marks) - self.top)):
            i = self.top + k
            pos = marks[i]
            y = y0 + k * self.row_h
            is_cur = i == cur
            editing = app.editing == pos
            if editing:
                self.create_rectangle(0, y, w, y + self.row_h, fill=PANEL, outline="", tags="rows")
            elif is_cur:
                self.create_rectangle(0, y, w, y + self.row_h, fill="#1F2D2D", outline="", tags="rows")
            if is_cur:
                self.create_rectangle(0, y, px(3), y + self.row_h, fill=RED_CURSOR if app.mode == ONLINE else TEAL, outline="", tags="rows")
            self.create_line(0, y + self.row_h - 1, w, y + self.row_h - 1, fill="#24272C", tags="rows")
            cy = y + self.row_h / 2
            self.create_text(self.x_num, cy, text=str(i + 1), anchor="e", fill=TEAL_SOFT if is_cur else DIM, font=app.f_mono, tags="rows")
            secs = pos / rate
            t = f"{int(secs // 3600)}:{int(secs % 3600 // 60):02d}:{int(secs % 60):02d}"
            item = self.create_text(self.x_time, cy, text=t, anchor="w", fill=TEXT if is_cur else "#B4BAC2", font=app.f_mono11, tags="rows")
            bx0, _by0, bx1, by1 = self.bbox(item)
            self.create_line(bx0, by1, bx1, by1, fill=LINE2, tags="rows")
            if editing:
                continue
            mx0, my0, mx1, my1 = self.memo_box(i)
            memo = app.memos.get(pos, "")
            if memo:
                round_rect(self, mx0, my0, mx1, my1, px(5), fill="#191B1F", outline=LINE, tags="rows")
                self.create_text(mx0 + px(7), cy, text=clip_text(self.f_memo, memo, mx1 - mx0 - px(14)), anchor="w", fill=TEXT, font=app.f_body, tags="rows")
            else:
                self.create_text(mx0 + px(7), cy, text="メモ…", anchor="w", fill=FAINT, font=app.f_body, tags="rows")


class Frost:
    """メモ入力中に、波形以外へかぶせる白っぽいすりガラス。Windowsだけ。"""

    KEY = "#ff00fe"

    def __init__(self, app):
        self.app = app
        self.win = None
        self.canvas = None
        self.noise = None
        self.last = None
        self.ok = sys.platform == "win32"

    def _build(self):
        root = self.app.root
        win = tk.Toplevel(root)
        win.withdraw()
        win.overrideredirect(True)
        win.attributes("-alpha", 0.42)
        win.attributes("-transparentcolor", self.KEY)
        win.transient(root)
        c = tk.Canvas(win, bg="#C9D0D6", highlightthickness=0, cursor="arrow")
        c.pack(fill="both", expand=True)
        c.bind("<ButtonRelease-1>", lambda _e: self.app.commit_memo())
        rng = np.random.default_rng(7)
        size = 96
        palette = np.array(["#C9D0D6", "#E3E7EB", "#F4F6F8", "#B1B8BF"])
        pick = rng.choice(4, size=(size, size), p=(0.70, 0.14, 0.10, 0.06))
        img = tk.PhotoImage(width=size, height=size)
        img.put(" ".join("{" + " ".join(palette[row]) + "}" for row in pick))
        self.win, self.canvas, self.noise = win, c, img

    def show(self, holes):
        if not self.ok:
            return
        try:
            if self.win is None:
                self._build()
            root = self.app.root
            x, y, w, h = root.winfo_rootx(), root.winfo_rooty(), root.winfo_width(), root.winfo_height()
            key = (x, y, w, h, tuple(holes))
            if key != self.last:
                self.last = key
                self.win.geometry(f"{w}x{h}+{x}+{y}")
                c = self.canvas
                c.delete("all")
                size = 96
                for ty in range(0, h, size):
                    for tx in range(0, w, size):
                        c.create_image(tx, ty, image=self.noise, anchor="nw")
                for hx0, hy0, hx1, hy1 in holes:
                    c.create_rectangle(hx0 - x, hy0 - y, hx1 - x, hy1 - y, fill=self.KEY, outline="")
            if self.win.state() != "normal":
                self.win.deiconify()
                self.win.lift()
        except tk.TclError:
            log_exception()
            self.ok = False

    def hide(self):
        self.last = None
        if self.win is not None:
            try:
                self.win.withdraw()
            except tk.TclError:
                pass


class App:
    def __init__(self, root, dnd=False):
        self.root = root
        self.scale = max(1.0, root.winfo_fpixels("1i") / 96)
        self.cfg = load_config()
        self.engine = Engine(int(self.cfg["samplerate"]))
        self.peaks = Peaks(self.engine.timeline)
        self.mode = LIVE
        self.marks = []
        self.memos = {}
        self.current_mark = None
        self.list_version = 0
        self.editing = None
        self.edit_entry = None
        self.edit_item = None
        self.source = None  # None / "rec" / "file"
        self.file_path = None
        self.session_path = None
        self.dirty = False
        self.autosave_at = 0.0
        self.loading = None
        self.spp = float(self.engine.rate * 30 / 900)
        self.v_gain = 1.0
        self.free_view = None
        self.online_started = 0.0
        self.online_start_pos = 0
        self.down = set()
        self.shift_held = False
        self.notice = ""
        self.notice_color = YELLOW
        self.notice_until = 0.0
        self.engine_error = ""
        self.tick_count = 0
        self.last_tick = time.monotonic()
        self.settings_open = False
        self.settings_view = None
        self.band_state = None
        self.peak_led_until = 0.0
        self.peak_led_on = None
        self.spin = 0

        families = set(tkfont.families())
        family = next((f for f in ("Yu Gothic UI", "Meiryo UI", "Noto Sans CJK JP") if f in families), "TkDefaultFont")
        mono = next((f for f in ("Consolas", "DejaVu Sans Mono") if f in families), family)
        self.f_title = (family, -self.px(18), "bold")
        self.f_h1 = (family, -self.px(20), "bold")
        self.f_h2 = (family, -self.px(14), "bold")
        self.f_band = (family, -self.px(20), "bold")
        self.f_band_side = (family, -self.px(15), "bold")
        self.f_pos = (mono, -self.px(22))
        self.f_body = (family, -self.px(12))
        self.f_body_b = (family, -self.px(13), "bold")
        self.f_small = (family, -self.px(11))
        self.f_small_b = (family, -self.px(11), "bold")
        self.f_tiny = (family, -self.px(10))
        self.f_mono = (mono, -self.px(10))
        self.f_mono11 = (mono, -self.px(11))
        self.f_mono11_b = (mono, -self.px(11), "bold")
        self.f_tiny_mono = (mono, -self.px(8))
        self.f_tiny_mono_b = (mono, -self.px(9), "bold")
        self.f_key = (mono, -self.px(10))
        self.f_enter = (family, -self.px(16))
        self.f_btn = (family, -self.px(13), "bold")
        self.f_rec = (family, -self.px(15), "bold")
        self.f_rec_sub = (family, -self.px(9), "bold")
        self.f_vu = ("Arial", -max(7, self.px(8)), "bold")
        self.f_vu_big = ("Arial", -self.px(11), "bold")

        root.title(f"{APP_NAME} {VERSION}")
        root.configure(bg=BG)
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        root.geometry(f"{min(self.px(1320), sw - 40)}x{min(self.px(860), sh - 90)}")
        root.minsize(min(self.px(1240), sw - 20), min(self.px(700), sh - 90))
        if sys.platform == "win32":
            try:
                root.state("zoomed")  # 全画面での使用が基本
            except tk.TclError:
                pass
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        icon = Path(getattr(sys, "_MEIPASS", APP_DIR)) / "icon.ico"
        if sys.platform == "win32" and icon.exists():
            try:
                root.iconbitmap(default=str(icon))
            except tk.TclError:
                pass
        self.build_style()
        self.frost = Frost(self)
        self.build_statusbar()
        self.main_view = tk.Frame(root, bg=BG)
        self.build(self.main_view)
        self.main_view.pack(fill="both", expand=True, padx=self.px(18))
        root.bind("<KeyPress>", self.on_key)
        root.bind("<KeyRelease>", self.on_key_release)
        root.bind("<FocusOut>", self.on_focus_out)
        root.bind("<MouseWheel>", self.on_wheel)
        root.bind("<Button-4>", lambda e: self.on_wheel(e, 120))
        root.bind("<Button-5>", lambda e: self.on_wheel(e, -120))
        if dnd:
            try:
                root.drop_target_register("DND_Files")
                root.dnd_bind("<<Drop>>", self.on_drop)
            except Exception:
                log_exception()
        root.update_idletasks()
        ime_set(root, False)  # マークのキーを日本語入力に取られないようにする（メモ欄だけオンにする）
        self.start_audio(first_run=True)
        root.focus_force()
        self.tick()

    def px(self, n):
        return int(round(n * getattr(self, "scale", 1.0)))

    def build_style(self):
        style = ttk.Style(self.root)
        style.theme_use("clam")
        for name, border in (("Dark.TCombobox", LINE2), ("Changed.TCombobox", YELLOW)):
            style.configure(name, fieldbackground=BG, background=PANEL2, foreground=TEXT, arrowcolor=MUTED,
                            bordercolor=border, lightcolor=border, darkcolor=border, padding=self.px(6))
            style.map(name, fieldbackground=[("readonly", BG), ("disabled", PANEL)], foreground=[("readonly", TEXT), ("disabled", FAINT)],
                      selectbackground=[("readonly", BG)], selectforeground=[("readonly", TEXT)])
        self.root.option_add("*TCombobox*Listbox.background", PANEL2)
        self.root.option_add("*TCombobox*Listbox.foreground", TEXT)
        self.root.option_add("*TCombobox*Listbox.selectBackground", TEAL)
        self.root.option_add("*TCombobox*Listbox.selectForeground", TEAL_TEXT)

    # ---- 組み立て ----
    def build_statusbar(self):
        px = self.px
        bar = tk.Frame(self.root, bg=BG)
        bar.pack(side="bottom", fill="x")
        tk.Frame(bar, bg=LINE, height=1).pack(side="top", fill="x")
        row = tk.Frame(bar, bg=BG)
        row.pack(fill="x", padx=px(18))
        self.status_items = {}
        for key in ("state", "latency", "dropouts", "memory", "disk", "format"):
            label = tk.Label(row, text="", font=self.f_mono11, fg=MUTED, bg=BG)
            label.pack(side="left", padx=(0, px(16)), pady=px(3))
            self.status_items[key] = label
        self.reconnect_button = Btn(self, row, text="再接続", on_click=self.reconnect, height=18, radius=5, parent_bg=BG, style="teal", pad=8)
        self.reconnect_shown = False
        self.status_file = tk.Label(row, text="", font=self.f_mono11, fg=MUTED, bg=BG)
        self.status_file.pack(side="right")

    def build(self, parent):
        px = self.px
        self.build_header(parent)
        tk.Frame(parent, bg=LINE, height=1).pack(fill="x")

        self.band = tk.Canvas(parent, height=px(40), bg=BG, highlightthickness=0, takefocus=0)
        self.band.pack(fill="x", pady=px(10))
        self.band.bind("<Configure>", lambda _e: self.draw_band(force=True))

        bottom = tk.Frame(parent, bg=BG, height=px(262))
        bottom.pack(side="bottom", fill="x", pady=(px(10), px(12)))
        bottom.grid_propagate(False)
        bottom.columnconfigure(0, weight=1)
        bottom.columnconfigure(1, minsize=px(330))
        bottom.rowconfigure(0, weight=1)
        self.build_bottom(bottom)

        wave_card = RoundPanel(self, parent, pad=10)
        wave_card.pack(fill="both", expand=True)
        head = tk.Frame(wave_card.inner, bg=PANEL)
        head.pack(fill="x")
        self.wave_title = tk.Label(head, text="録音（録りっぱなし）", font=self.f_h2, fg=TEXT, bg=PANEL)
        self.wave_title.pack(side="left", padx=(px(4), px(14)))
        self.pos_label = tk.Label(head, text="00:00:00.00", font=self.f_pos, fg=YELLOW, bg=PANEL)
        self.pos_label.pack(side="left", padx=(0, px(12)))
        self.pos_note = tk.Label(head, text="", font=self.f_small, fg=MUTED, bg=PANEL)
        self.pos_note.pack(side="left", padx=(0, px(14)))
        self.not_active = tk.Label(head, text=" NOT ACTIVE  クリックで操作できます ", font=self.f_small_b, fg="#241703", bg=YELLOW)
        self.not_active_shown = False
        self.length_label = tk.Label(head, text="", font=self.f_mono11, fg=MUTED, bg=PANEL)
        self.length_label.pack(side="right", padx=(px(14), px(4)))
        self.notice_label = tk.Label(head, text="", font=self.f_body_b, fg=YELLOW, bg=PANEL, anchor="e")
        self.notice_label.pack(side="right", fill="x", expand=True)
        self.wave = tk.Canvas(wave_card.inner, bg=LANE, highlightthickness=0, height=px(180), takefocus=0)
        self.wave.pack(fill="both", expand=True, pady=(px(6), 0))
        self.wave.bind("<Button-1>", self.on_wave_click)

    def build_header(self, parent):
        px = self.px
        header = tk.Frame(parent, bg=BG)
        header.pack(fill="x", pady=(px(10), px(8)))
        header.columnconfigure(2, weight=1)

        brand = tk.Frame(header, bg=BG)
        brand.grid(row=0, column=0, sticky="nw", padx=(0, px(12)))
        header.columnconfigure(0, minsize=px(190))
        tk.Label(brand, text=APP_NAME, font=self.f_title, fg=TEXT, bg=BG).pack(anchor="w")
        self.brand_lines = {}
        for key, title in (("session", "セッション"), ("in", "入力"), ("out", "出力")):
            line = tk.Frame(brand, bg=BG)
            line.pack(anchor="w")
            tk.Label(line, text=title, font=self.f_small, fg=MUTED, bg=BG).pack(side="left", padx=(0, px(5)))
            value = tk.Label(line, text="", font=self.f_small, fg=TEXT, bg=BG)
            value.pack(side="left")
            self.brand_lines[key] = value

        files = tk.Frame(header, bg=BG)
        files.grid(row=0, column=1, sticky="w", padx=(0, px(12)))
        self.file_buttons = {}
        busy = lambda: self.show_notice("録音中はできません")
        for i, (key, text, cmd, style, guard) in enumerate((
                ("load", "音声読込", self.action_load_wav, "teal", busy),
                ("save", "保存", self.action_save, "normal", None),
                ("open", "開く", self.action_open, "normal", busy),
                ("new", "新規", self.action_new, "normal", busy),
                ("settings", "設定", self.open_settings, "normal", lambda: self.show_notice("録音を止めてから開いてください")),
                ("folder", "フォルダ", self.open_folder, "normal", None))):
            b = Btn(self, files, text=text, on_click=cmd, width=72, height=25, radius=7, style=style, parent_bg=BG, on_disabled=guard)
            b.grid(row=i // 2, column=i % 2, padx=(0, px(4)), pady=(0, px(4)))
            self.file_buttons[key] = b

        guide = tk.Frame(header, bg=BG)
        guide.grid(row=0, column=2, sticky="nsew", padx=(0, px(12)))
        tk.Frame(guide, bg=LINE2, width=px(2)).pack(side="left", fill="y", padx=(0, px(10)))
        tk.Label(guide, text="操作", font=self.f_small_b, fg=AMBER, bg=BG).pack(side="left", anchor="n", pady=(px(4), 0), padx=(0, px(10)))
        keys = tk.Frame(guide, bg=BG)
        keys.pack(side="left", fill="both", expand=True)
        items = (("A–Z", "マーク"), ("↓ ↑", "マーク移動"), ("← →", "2秒"), ("SPACE", "再生/止める"),
                 ("SHIFT", "押す間 等倍"), ("⇧A–Z", "カーソルにマーク"), (", .", "縮小/拡大"), ("ESC", "先端に戻る"))
        self.key_cells = []
        for key, text in items:
            cell = tk.Frame(keys, bg=BG)
            KeyCap(self, cell, key).pack(side="left", padx=(0, px(5)))
            tk.Label(cell, text=text, font=self.f_body, fg=MUTED, bg=BG).pack(side="left")
            self.key_cells.append(cell)
        self.key_cols = 0
        self.layout_keys(cols=4)
        keys.bind("<Configure>", lambda e: self.layout_keys(width=e.width))

        self.enter_block = EnterBlock(self, header)
        self.enter_block.grid(row=0, column=3, sticky="ns", padx=(0, px(12)))

        vu_box = tk.Frame(header, bg=PANEL, highlightthickness=1, highlightbackground=LINE)
        vu_box.grid(row=0, column=4, sticky="ns", padx=(0, px(12)))
        self.vu = VUMeter(self, vu_box)
        self.vu.pack(side="left", padx=(px(8), px(6)), pady=px(4))
        led_col = tk.Frame(vu_box, bg=PANEL)
        led_col.pack(side="left", padx=(0, px(8)))
        tk.Label(led_col, text="INPUT", font=self.f_tiny_mono, fg=MUTED, bg=PANEL).pack()
        self.peak_led = tk.Canvas(led_col, width=px(13), height=px(13), bg=PANEL, highlightthickness=0)
        self.peak_led_item = self.peak_led.create_oval(1, 1, px(12), px(12), fill="#3A2020", outline="#5A2A2C")
        self.peak_led.pack(pady=px(3))
        tk.Label(led_col, text="PEAK", font=self.f_tiny_mono, fg=MUTED, bg=PANEL).pack()

        self.rec_button = tk.Canvas(header, width=px(96), height=px(70), bg=BG, highlightthickness=0, takefocus=0, cursor="hand2")
        self.rec_button.grid(row=0, column=5, sticky="ns")
        self.rec_state = None
        self.rec_pressed = False
        self.rec_button.bind("<Configure>", lambda _e: self.draw_rec_button(force=True))
        self.rec_button.bind("<ButtonPress-1>", lambda _e: self.press_rec(True))
        self.rec_button.bind("<ButtonRelease-1>", lambda e: self.press_rec(False, e))

    def fit_meters(self, width, right):
        """メーターの長さは300くらい。狭い画面では、ボタンが切れないように縮める。"""
        px = self.px
        want = min(px(300), max(px(200), width - right.winfo_reqwidth() - px(30)))
        if abs(int(self.meter_block.cget("width")) - want) > 2:
            self.meter_block.configure(width=want)

    def layout_keys(self, cols=None, width=None):
        """操作説明は、入る幅に合わせて列の数を変える（4列〜2列）。"""
        if cols is None:
            if width < 20:
                return
            pad = self.px(10)
            cols = 2
            for n in (4, 3):
                need = sum(max(c.winfo_reqwidth() for c in self.key_cells[k::n]) + pad for k in range(n))
                if need <= width:
                    cols = n
                    break
        if cols == self.key_cols:
            return
        self.key_cols = cols
        for i, cell in enumerate(self.key_cells):
            cell.grid(row=i // cols, column=i % cols, sticky="w", padx=(0, self.px(10)), pady=self.px(3))

    def build_bottom(self, bottom):
        px = self.px
        left = tk.Frame(bottom, bg=BG)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, px(10)))
        ctrl = RoundPanel(self, left, pad=12)
        ctrl.pack(fill="x")
        self.meter_block = MeterBlock(self, ctrl.inner)
        self.meter_block.pack(side="left", anchor="center")
        tk.Frame(ctrl.inner, bg=LINE, width=1).pack(side="left", fill="y", padx=px(14))
        right = tk.Frame(ctrl.inner, bg=PANEL)
        right.pack(side="left", fill="x", expand=True)
        row = tk.Frame(right, bg=PANEL)
        row.pack(anchor="w")
        self.transport = {}
        for key, text, command, kind, width in (
                ("prev", "◀◀ MARK", lambda: self.key_mark(-1), None, 78),
                ("back", "−2s", lambda: self.key_jump(-1), None, 54),
                ("play", "", self.transport_play, "play", 40),
                ("forward", "+2s", lambda: self.key_jump(1), None, 54),
                ("next", "MARK ▶▶", lambda: self.key_mark(1), None, 78),
                ("live", "LIVE ▶▶|", self.transport_live, None, 78)):
            b = Btn(self, row, text=text, on_click=command, width=width, height=43, radius=7, font=self.f_btn,
                    style="play" if kind else "normal", kind=kind)
            b.pack(side="left", padx=(0, px(6)))
            self.transport[key] = b
        ctrl.inner.bind("<Configure>", lambda e: self.fit_meters(e.width, right), add="+")
        speed = tk.Frame(right, bg=PANEL)
        speed.pack(anchor="w", pady=(px(10), 0))
        tk.Label(speed, text="検聴速度", font=self.f_small_b, fg=AMBER, bg=PANEL).pack(side="left", padx=(0, px(10)))
        self.v_speed = tk.DoubleVar(value=1.5)
        self.speed_pills = SpeedPills(self, speed, self.v_speed, self.on_speed)
        self.speed_pills.pack(side="left", padx=(0, px(10)))
        self.v_keep_pitch = tk.BooleanVar(value=True)
        self.pitch_button = Btn(self, speed, text="KEEP PITCH", on_click=self.on_pitch, height=25, radius=7, font=self.f_small_b, style="teal")
        self.pitch_button.pack(side="left")

        info = RoundPanel(self, left, pad=12)
        info.pack(fill="both", expand=True, pady=(px(10), 0))
        info.inner.columnconfigure(1, weight=1)
        self.info_values = {}
        for r, (key, title) in enumerate((("session", "セッション"), ("rec", "録音ファイル"), ("file", "読込WAV"))):
            tk.Label(info.inner, text=title, font=self.f_body, fg=MUTED, bg=PANEL).grid(row=r, column=0, sticky="w", padx=(px(2), px(18)), pady=px(2))
            value = tk.Label(info.inner, text="", font=self.f_body, fg=TEXT, bg=PANEL, anchor="w")
            value.grid(row=r, column=1, sticky="w", pady=px(2))
            self.info_values[key] = value

        marks = RoundPanel(self, bottom, pad=4)
        marks.grid(row=0, column=1, sticky="nsew")
        head = tk.Frame(marks.inner, bg=PANEL)
        head.pack(fill="x", padx=px(8), pady=(px(5), px(5)))
        tk.Label(head, text="マーク", font=self.f_h2, fg=AMBER, bg=PANEL).pack(side="left")
        self.mark_count = tk.Label(head, text="0", font=self.f_mono11, fg=MUTED, bg=PANEL)
        self.mark_count.pack(side="left", padx=px(8))
        tk.Label(head, text="時刻クリックで移動　ダブルクリックでメモ", font=self.f_tiny, fg=DIM, bg=PANEL).pack(side="right", padx=(px(8), 0))
        tk.Frame(marks.inner, bg=LINE, height=1).pack(fill="x")
        self.mark_list = MarkList(self, marks.inner)
        self.mark_list.pack(fill="both", expand=True)

    # ---- オーディオ起動 ----
    def start_audio(self, first_run=False):
        if not self.cfg["input_device"] or not self.cfg["output_device"]:
            self.engine_error = "設定で入出力のデバイスを選んでください"
            if first_run:
                self.root.after(200, self.open_settings)
            return
        try:
            self.engine.open(self.cfg)
            self.engine_error = ""
            if self.cfg.get("exclusive") and self.engine.exclusive_failed:
                self.show_notice("排他モードで開けませんでした。Windowsの音が混ざることがあります", 8.0)
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
        self.root.focus_set()

    def open_settings(self):
        if self.mode == ONLINE or self.engine.recording:
            self.show_notice("録音を止めてから開いてください")
            return
        if self.settings_open or self.loading:
            return
        self.commit_memo()
        self.leave_search()
        self.engine.close()
        rescan_devices()
        self.settings_open = True
        self.main_view.pack_forget()
        self.settings_view = SettingsView(self)
        self.settings_view.pack(fill="both", expand=True, padx=self.px(18))
        self.root.focus_set()

    def close_settings(self, cfg=None):
        self.settings_open = False
        self.settings_view.destroy()
        self.settings_view = None
        self.main_view.pack(fill="both", expand=True, padx=self.px(18))
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
                self.show_notice("本線と検聴が同じチャンネルです", 8.0)
        self.start_audio()
        self.root.focus_force()
        self.draw_band(force=True)

    def open_folder(self):
        d = Path(self.cfg["save_dir"])
        try:
            d.mkdir(parents=True, exist_ok=True)
            if sys.platform == "win32":
                os.startfile(str(d))
        except OSError as e:
            self.show_notice(f"フォルダを開けません: {e}", 5.0, RED_SOFT)
        self.root.focus_set()

    def apply_cue_settings(self):
        self.engine.cue.speed = self.effective_speed()
        self.engine.cue.keep_pitch = bool(self.v_keep_pitch.get())

    # ---- 上部 ----
    def effective_speed(self):
        return 1.0 if self.shift_held else round(float(self.v_speed.get()), 1)

    def on_speed(self, _value=None):
        if self.mode == ONLINE:
            return
        self.engine.cue.speed = self.effective_speed()
        self.root.focus_set()

    def on_pitch(self):
        if self.mode == ONLINE:
            return
        self.v_keep_pitch.set(not self.v_keep_pitch.get())
        self.engine.cue.keep_pitch = bool(self.v_keep_pitch.get())
        self.root.focus_set()

    def show_notice(self, text, seconds=3.0, color=YELLOW):
        self.notice = text
        self.notice_color = color
        self.notice_until = time.monotonic() + seconds
        self.update_status()

    # ---- 録音 ----
    def toggle_record(self):
        if self.mode == ONLINE or self.loading or self.editing is not None:
            return
        if self.engine.recording:
            self.engine.stop_recording()
            return
        if not self.engine.healthy():
            self.show_notice("オーディオが動いていません", 3.0, RED_SOFT)
            return
        if self.source == "file":
            if not messagebox.askokcancel(APP_NAME, "読み込んだ音声を閉じて、新しく録音を始めますか？", parent=self.root):
                self.root.focus_force()
                return
            if not self.reset_session(ask=True):
                self.root.focus_force()
                return
        try:
            self.engine.start_recording(self.cfg["save_dir"])
            self.source = "rec"
        except OSError as e:
            log_exception()
            self.show_notice(f"録音を始められません: {e}", 6.0, RED_SOFT)
        self.root.focus_force()

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

    def highlight_mark(self):
        """一覧で目立たせるマーク。選んでいるマーク、なければ今の位置の直前のマーク。"""
        if not self.marks:
            return None
        if self.current_mark is not None and self.current_mark < len(self.marks):
            return self.current_mark
        i = bisect.bisect_right(self.marks, self.position()) - 1
        return i if i >= 0 else None

    def jump_to_mark(self, i):
        """一覧の時刻をクリック。マークの少し前（↓↑と同じ秒数）へ飛ぶ。"""
        if self.mode == ONLINE or self.editing is not None or not self.engine.timeline.head or not 0 <= i < len(self.marks):
            return
        offset = float(self.cfg.get("mark_offset", 2.0)) * self.engine.rate
        self.jump_keep_state(self.marks[i] - offset)
        self.current_mark = i
        self.root.focus_set()

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
        if self.editing is not None:  # メモ入力中のキーはメモ欄だけが受ける
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
        if event.state & 0x0004:  # Ctrl
            low = ks.lower()
            if low == "s":
                self.action_save(save_as=bool(event.state & 0x0001))
            elif low == "o":
                self.action_open()
            return
        if self.loading:
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
        elif ks in ("period", "greater"):
            self.zoom_keys(True)
        elif ks in ("comma", "less"):
            self.zoom_keys(False)
        elif self.is_letter(event):
            if self.shift_held or event.state & 0x0001:
                self.add_mark_at_cursor()
            else:
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
            self.show_notice("オーディオが動いていません", 3.0, RED_SOFT)
            return
        pos = self.position()
        limit = self.engine.timeline.head - int(self.engine.rate * HEAD_MARGIN_SECONDS)
        if pos >= limit:
            self.show_notice("先端では出せません")
            return
        self.free_view = None
        self.engine.go_online(pos)
        self.online_start_pos = pos
        self.mode = ONLINE
        self.online_started = time.monotonic()

    def enter_click(self):
        """ENTERの枠をクリック。キーと同じ決まりで、出す／止める。"""
        if self.settings_open or self.editing is not None or self.loading:
            return
        if self.mode == ONLINE:
            if time.monotonic() - self.online_started >= ENTER_GUARD_SECONDS:
                self.engine.stop_online()
        else:
            self.key_enter()
        self.root.focus_set()

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
        if self.mode == ONLINE or self.loading:
            return
        if not self.marks:
            self.show_notice("マークがまだありません")
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
        self.mark_list.ensure_visible(i)

    def key_jump(self, direction):
        if self.mode == ONLINE or self.loading:
            return
        step = int(JUMP_SECONDS * self.engine.rate)
        if self.mode == LIVE and direction > 0:
            return
        if not self.engine.timeline.head:
            return
        self.current_mark = None
        self.play_at(self.position() + direction * step)

    def add_mark(self):
        if self.source == "file":
            self.add_mark_at_cursor()
            return
        self.insert_mark(self.engine.timeline.head)

    def add_mark_at_cursor(self):
        """Shift＋文字。探している最中に、今聞いている位置へマークを打つ。"""
        self.insert_mark(self.position())

    def insert_mark(self, pos):
        if self.engine.timeline.head == 0:
            self.show_notice("録音を始めるとマークを打てます")
            return
        pos = int(pos)
        i = bisect.bisect_left(self.marks, pos)
        if i < len(self.marks) and self.marks[i] == pos:
            return
        self.marks.insert(i, pos)
        if self.current_mark is not None and self.current_mark >= i:
            self.current_mark += 1
        self.changed()

    def changed(self):
        """マークかメモが変わった。一覧を描き直し、録音ファイルの横へ自動で保存する。"""
        self.list_version += 1
        self.dirty = True
        self.autosave_at = time.monotonic() + 1.0

    def zoom_keys(self, zoom_in):
        """キーでの横の拡大・縮小。カーソルは左から1/3のまま。"""
        new = self.spp / 1.25 if zoom_in else self.spp * 1.25
        self.spp = min(max(new, 2.0), self.engine.rate * 3600 * 6 / max(1, self.wave.winfo_width()))
        self.free_view = None

    # ---- メモ ----
    def begin_memo(self, i):
        if self.mode == ONLINE:
            self.show_notice("ONLINE PLAY 中はメモを書けません")
            return
        if self.loading or not 0 <= i < len(self.marks):
            return
        self.commit_memo()
        ml = self.mark_list
        ml.ensure_visible(i)
        pos = self.marks[i]
        self.editing = pos
        ml.redraw(force=True)
        x0, y0, x1, y1 = ml.memo_box(i)
        entry = tk.Entry(ml, font=self.f_body, bg="#22252A", fg=TEXT, insertbackground=TEXT, relief="flat",
                         highlightthickness=1, highlightbackground=TEAL, highlightcolor=TEAL)
        entry.insert(0, self.memos.get(pos, ""))
        self.edit_item = ml.create_window(x0, y0, anchor="nw", window=entry, width=x1 - x0, height=y1 - y0)
        self.edit_entry = entry
        for seq in ("<Return>", "<KP_Enter>"):
            entry.bind(seq, lambda _e: (self.commit_memo(), "break")[1])
        entry.bind("<Escape>", lambda _e: (self.cancel_memo(), "break")[1])
        self.down.clear()
        self.set_shift(False)
        entry.focus_force()
        entry.icursor("end")
        entry.update_idletasks()
        ime_set(self.root, True)  # Windowsではキー入力がウィンドウ全体に届くので、入力中だけ全体をオンにする
        ime_set(entry, True)
        self.enter_block.redraw("memo")
        self.update_frost()
        self.root.after(30, lambda: self.edit_entry is entry and entry.focus_force())

    def update_frost(self):
        if self.editing is None or self.edit_entry is None:
            return
        holes = []
        ml = self.mark_list
        i = bisect.bisect_left(self.marks, self.editing)
        try:
            x, y = ml.winfo_rootx(), ml.winfo_rooty() + ml.y0() + (i - ml.top) * ml.row_h
            holes.append((x, y, x + ml.winfo_width(), y + ml.row_h))
        except tk.TclError:
            pass
        for w in (self.wave, self.enter_block):
            try:
                x, y = w.winfo_rootx(), w.winfo_rooty()
                holes.append((x - 2, y - 2, x + w.winfo_width() + 2, y + w.winfo_height() + 2))
            except tk.TclError:
                pass
        self.frost.show(holes)

    def commit_memo(self):
        if self.editing is None or self.edit_entry is None:
            return
        text = self.edit_entry.get().strip()
        pos = self.editing
        if text != self.memos.get(pos, ""):
            if text:
                self.memos[pos] = text
            else:
                self.memos.pop(pos, None)
            self.changed()
        self.end_memo()

    def cancel_memo(self):
        self.end_memo()

    def end_memo(self):
        self.frost.hide()
        ime_set(self.root, False)
        if self.edit_entry is not None:
            try:
                self.mark_list.delete(self.edit_item)
                self.edit_entry.destroy()
            except tk.TclError:
                pass
        self.edit_entry = None
        self.edit_item = None
        self.editing = None
        self.list_version += 1
        self.down.clear()
        self.enter_block.redraw("online" if self.mode == ONLINE else "normal")
        self.root.focus_force()

    # ---- 読み込み・セッション ----
    def audio_path(self):
        if self.engine.writer is not None:
            return self.engine.writer.path
        return self.file_path

    def busy_reason(self, what):
        if self.engine.recording:
            return f"録音中は{what}できません"
        if self.mode == ONLINE:
            return f"ONLINE PLAY 中は{what}できません"
        if self.loading:
            return "読み込み中です"
        return ""

    def confirm_discard(self):
        if self.dirty and (self.marks or self.memos):
            ok = messagebox.askokcancel(APP_NAME, "保存していないマークやメモがあります。閉じてもいいですか？", parent=self.root)
            self.root.focus_force()
            return ok
        return True

    def reset_session(self, ask=True):
        if ask and not self.confirm_discard():
            return False
        self.leave_search()
        self.engine.replace_timeline(Timeline(self.engine.rate))
        self.peaks = Peaks(self.engine.timeline)
        self.marks = []
        self.memos = {}
        self.source = None
        self.file_path = None
        self.session_path = None
        self.dirty = False
        self.autosave_at = 0.0
        self.list_version += 1
        self.mark_list.follow = True
        return True

    def action_new(self):
        self.commit_memo()
        reason = self.busy_reason("新規に")
        if reason:
            self.show_notice(reason)
            return
        if self.reset_session(ask=True):
            self.show_notice("新しいセッションにしました")
        self.root.focus_force()

    def action_load_wav(self):
        self.commit_memo()
        reason = self.busy_reason("読み込み")
        if reason:
            self.show_notice(reason)
            return
        path = filedialog.askopenfilename(parent=self.root, title="WAVを読み込む", filetypes=[("WAV", "*.wav *.WAV"), ("すべて", "*.*")])
        self.root.focus_force()
        if path:
            self.start_load(Path(path))

    def action_open(self):
        self.commit_memo()
        reason = self.busy_reason("開くことは")
        if reason:
            self.show_notice(reason)
            return
        path = filedialog.askopenfilename(parent=self.root, title="セッションを開く", initialdir=self.cfg["save_dir"],
                                          filetypes=[("セッション", "*.json"), ("すべて", "*.*")])
        self.root.focus_force()
        if path:
            self.open_session(Path(path))

    def open_session(self, path):
        try:
            audio, marks, data = read_session(path)
        except Exception as e:
            self.show_notice(f"開けません: {e}", 6.0, RED_SOFT)
            return
        self.start_load(audio, marks=marks, session=path, data=data)

    def start_load(self, path, marks=None, session=None, data=None):
        reason = self.busy_reason("読み込み")
        if reason:
            self.show_notice(reason)
            return
        if not self.confirm_discard():
            return
        job = {"path": Path(path), "marks": marks or [], "session": session, "data": data or {}, "rate": self.engine.rate,
               "progress": 0.0, "done": False, "error": None, "timeline": None, "cancel": False}

        def run():
            try:
                job["timeline"] = load_timeline(job["path"], job["rate"],
                                                progress=lambda f: job.__setitem__("progress", f),
                                                cancel=lambda: job["cancel"])
            except MemoryError:
                job["error"] = "メモリが足りません"
            except Exception as e:
                log_exception()
                job["error"] = str(e)
            finally:
                job["done"] = True

        self.leave_search()
        self.loading = job
        threading.Thread(target=run, daemon=True).start()

    def finish_load(self, job):
        self.loading = None
        if job["error"] or job["timeline"] is None:
            self.show_notice(f"読み込めません: {job['error'] or '中止しました'}", 6.0, RED_SOFT)
            return
        tl = job["timeline"]
        self.leave_search()
        self.engine.replace_timeline(tl)
        self.peaks = Peaks(tl)
        rate = tl.rate
        self.marks = []
        self.memos = {}
        for secs, memo in job["marks"]:
            pos = min(max(0, int(round(secs * rate))), tl.head)
            i = bisect.bisect_left(self.marks, pos)
            if i < len(self.marks) and self.marks[i] == pos:
                continue
            self.marks.insert(i, pos)
            if memo:
                self.memos[pos] = memo
        data = job["data"]
        if "cue_speed" in data:
            try:
                speed = float(data["cue_speed"])
                self.v_speed.set(min(SPEEDS, key=lambda s: abs(s - speed)))
            except (TypeError, ValueError):
                pass
        if "keep_pitch" in data:
            self.v_keep_pitch.set(bool(data["keep_pitch"]))
        self.apply_cue_settings()
        self.speed_pills.redraw()
        self.source = "file"
        self.file_path = job["path"]
        self.session_path = job["session"]
        self.dirty = False
        self.autosave_at = 0.0
        self.list_version += 1
        self.mark_list.follow = True
        self.show_notice(f"読み込みました: {job['path'].name}")

    def session_data(self, base_dir):
        audio = self.audio_path()
        rate = self.engine.rate
        rel = ""
        if audio is not None:
            try:
                rel = os.path.relpath(audio, base_dir)
            except ValueError:  # 別のドライブ
                rel = ""
        return {
            "app": "ChallengeREC", "version": SESSION_VERSION,
            "saved_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "audio": str(audio) if audio is not None else "", "audio_relative": rel,
            "samplerate": rate,
            "cue_speed": round(float(self.v_speed.get()), 1), "keep_pitch": bool(self.v_keep_pitch.get()),
            "marks": [{"time": round(m / rate, 4), "memo": self.memos.get(m, "")} for m in self.marks],
        }

    def action_save(self, save_as=False):
        self.commit_memo()
        if self.mode == ONLINE or self.loading:
            return
        audio = self.audio_path()
        if audio is None:
            self.show_notice("保存するものがまだありません")
            return
        path = None if save_as else self.session_path
        if path is None:
            name = filedialog.asksaveasfilename(parent=self.root, title="セッションを保存", defaultextension=".json",
                                                initialdir=str(Path(audio).parent), initialfile=Path(audio).stem + ".json",
                                                filetypes=[("セッション", "*.json")])
            self.root.focus_force()
            if not name:
                return
            path = Path(name)
        try:
            write_json(path, self.session_data(Path(path).parent))
        except OSError as e:
            log_exception()
            self.show_notice(f"保存できません: {e}", 6.0, RED_SOFT)
            return
        self.session_path = Path(path)
        self.dirty = False
        self.show_notice(f"保存しました: {Path(path).name}")

    def autosave(self):
        """録音中のマークとメモを、録音ファイルの横（同じ名前の .json）へ書いておく。落ちたときの控え。"""
        self.autosave_at = 0.0
        w = self.engine.writer
        if w is None:
            return
        try:
            path = w.path.with_suffix(".json")
            write_json(path, self.session_data(path.parent))
        except OSError:
            log_exception()

    def on_drop(self, event):
        try:
            files = self.root.tk.splitlist(event.data)
        except tk.TclError:
            files = []
        if files:
            path = Path(files[0])
            self.commit_memo()
            reason = self.busy_reason("読み込み")
            if reason:
                self.show_notice(reason)
            elif path.suffix.lower() == ".json":
                self.open_session(path)
            else:
                self.start_load(path)
        return getattr(event, "action", "copy")

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
        if self.mode == ONLINE or not self.engine.timeline.head or self.loading:
            return
        width = self.wave.winfo_width()
        self.jump_keep_state(self.grid_start(width) + event.x * self.spp)

    def pointer_in(self, widget):
        x = self.root.winfo_pointerx() - widget.winfo_rootx()
        y = self.root.winfo_pointery() - widget.winfo_rooty()
        return 0 <= x < widget.winfo_width() and 0 <= y < widget.winfo_height(), x

    def on_wheel(self, event, delta=None):
        if self.settings_open:
            return
        delta = event.delta if delta is None else delta
        up = delta > 0
        inside_list, _ = self.pointer_in(self.mark_list)
        if inside_list:
            self.mark_list.scroll(-3 if up else 3)
            return
        if self.mode == ONLINE:
            return
        inside, mouse_x = self.pointer_in(self.wave)
        if not inside:
            return
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
        now = time.monotonic()
        dt = now - self.last_tick
        self.last_tick = now
        e = self.engine
        try:
            e.timeline.reserve()
        except MemoryError:
            log_exception()
            self.show_notice("メモリが足りません", 5.0, RED_SOFT)
        self.peaks.update()
        if self.loading and self.loading["done"]:
            self.finish_load(self.loading)
        if self.mode == ONLINE and not e.online:
            # Enterで止めたら、出した頭に戻してSTANDBY（フェーダーを上げ忘れたときに、そのまま出し直せる）。
            # 録音の先端まで出し切ったときは、先端で止まったまま
            self.mode = SEARCH
            self.current_mark = None
            e.cue.locate(e.online_pos if e.online_ended else self.online_start_pos)
        if self.free_view is not None and self.mode == SEARCH and e.cue.playing():
            self.free_view = None
        if self.autosave_at and now >= self.autosave_at:
            self.autosave()
        vu_sum, vu_n = e.vu_sum, e.vu_n
        e.vu_sum, e.vu_n = 0.0, 0
        peak_in = e.peak_in
        if self.settings_open:
            e.peak_in = e.peak_main = e.peak_cue = 0.0
            return
        self.vu.step(vu_sum / vu_n if vu_n else 0.0, dt)
        if peak_in >= 10 ** (PEAK_LED_DB / 20):
            self.peak_led_until = now + 1.5
        led = now < self.peak_led_until
        if led != self.peak_led_on:
            self.peak_led_on = led
            self.peak_led.itemconfigure(self.peak_led_item, fill=RED if led else "#3A2020", outline=RED_SOFT if led else "#5A2A2C")
        self.spin += 1
        self.draw_wave()
        self.draw_band()
        self.update_labels()
        self.mark_list.redraw()
        if self.editing is not None:
            self.update_frost()
        self.meter_block.set("in", peak_in)
        self.meter_block.set("main", e.peak_main)
        self.meter_block.set("cue", e.peak_cue)
        e.peak_in = e.peak_main = e.peak_cue = 0.0
        if self.tick_count % 12 == 0:
            self.update_status()

    def band_content(self):
        e = self.engine
        if self.mode == ONLINE:
            return RED, RED_SOFT, "#ffffff", "ONLINE PLAY", "チャレンジ　検証中"
        if self.mode == SEARCH:
            if e.cue.playing():
                return TEAL, TEAL, TEAL_TEXT, f"CUE　{self.effective_speed():.1f}倍速", ""
            return TEAL, TEAL, TEAL_TEXT, "STANDBY", ""
        if e.recording:
            return "#192526", "#2F645F", TEAL_SOFT, "RECORDING", ""
        return PANEL2, LINE2, "#B4BAC2", "STOPPED", ""

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
        fill, outline, fg, center, side = state
        round_rect(c, 1, 1, w - 2, h - 2, self.px(8), fill=fill, outline=outline)
        c.create_text(w / 2, h / 2, text=center, fill=fg, font=self.f_band)
        if side:
            c.create_text(w / 6, h / 2, text=side, fill=fg, font=self.f_band_side)
            c.create_text(w * 5 / 6, h / 2, text=side, fill=fg, font=self.f_band_side)

    def press_rec(self, down, event=None):
        self.rec_pressed = down
        self.draw_rec_button()
        if not down and event is not None and 0 <= event.x < self.rec_button.winfo_width() and 0 <= event.y < self.rec_button.winfo_height():
            self.toggle_record()

    def draw_rec_button(self, force=False):
        rec = self.engine.recording
        enabled = self.mode != ONLINE and not self.loading and self.editing is None
        state = (rec, enabled, self.rec_pressed)
        if state == self.rec_state and not force:
            return
        self.rec_state = state
        c = self.rec_button
        c.delete("all")
        w, h = c.winfo_width(), c.winfo_height()
        if w < 10:
            return
        if rec:
            top, bottom, outline, fg, word, sub = "#EC5A55", "#C23733", "#F07A75", "#ffffff", "● 録音中", "REC STOP"
        else:
            top, bottom, outline, fg, word, sub = "#2E3238", "#1B1D21", RED, RED_SOFT, "● 録音開始", "REC START"
        if not enabled:
            round_rect(c, 1, 1, w - 2, h - 4, self.px(10), fill="#8a3431" if rec else PANEL, outline="#7a3330" if rec else LINE2)
            dy = 0
            fg = "#f0c9c7" if rec else FAINT
        else:
            dy = raised(c, 1, 1, w - 2, h - 4, self.px(10), top, bottom, outline, self.rec_pressed)
        c.create_text(w / 2, (h - 3) / 2 - self.px(8) + dy, text=word, fill=fg, font=self.f_rec)
        c.create_text(w / 2, (h - 3) / 2 + self.px(11) + dy, text=sub, fill=fg, font=self.f_rec_sub)

    def draw_wave(self):
        c = self.wave
        w, h = c.winfo_width(), c.winfo_height()
        if w < 10 or h < 40:
            return
        e = self.engine
        rate = e.rate
        px = self.px
        c.delete("all")
        spp = self.spp
        # 列の境目を時間で固定し、波形はピクセル単位で送る。これで形が揺れない
        start = self.grid_start(w)
        ruler = px(RULER_H)
        wave_h = h - ruler
        mid = ruler + wave_h / 2
        scale = wave_h / 2 * self.v_gain * 0.92
        c.create_rectangle(0, ruler, w, h, fill=LANE_EMPTY, width=0)
        head = e.timeline.head
        x0 = int(round((0 - start) / spp))
        x1 = int(round((head - start) / spp))
        if x1 > 0 and x0 < w:
            c.create_rectangle(max(0, x0), ruler, min(w, x1), h, fill=LANE, width=0)
        c.create_rectangle(0, 0, w, ruler, fill=RULER_BG, width=0)
        c.create_line(0, ruler, w, ruler, fill=LINE)
        seconds_per_px = spp / rate
        step = next((s for s in (0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 1200, 1800, 3600, 7200)
                     if s / seconds_per_px >= 110), 14400)
        t = max(0.0, np.floor(start / rate / step) * step)
        while True:
            x = int(round((t * rate - start) / spp))
            if x > w:
                break
            if x >= 0:
                c.create_line(x, 0, x, h, fill=GRID)
                total = int(round(t * 10))
                hh, rem = divmod(total, 36000)
                mm, rem = divmod(rem, 600)
                ss = rem / 10
                text = f"{hh}:{mm:02d}:{ss:02.0f}" if step >= 1 else f"{hh}:{mm:02d}:{ss:04.1f}"
                c.create_text(x + px(4), px(4), text=text, anchor="nw", fill=DIM, font=self.f_mono)
            t += step
        c.create_line(0, mid, w, mid, fill=GRID)
        xs, mins, maxs = self.peaks.column_range(start, spp, w)
        if len(xs) >= 2:
            ytop = np.maximum(mid - maxs * scale, ruler + 1)
            ybot = np.minimum(mid - mins * scale, h)
            ybot = np.maximum(ybot, ytop + 1)
            coords = np.empty(len(xs) * 4)
            coords[0::4] = xs
            coords[1::4] = ytop
            coords[2::4] = xs
            coords[3::4] = ybot
            c.create_line(coords.tolist(), fill=TEAL)
        for seam in e.timeline.seams:
            x = int(round((seam - start) / spp))
            if 0 <= x <= w:
                c.create_line(x, ruler, x, h, fill=DIM, dash=(2, 4))
        left = bisect.bisect_left(self.marks, start)
        right = bisect.bisect_right(self.marks, start + w * spp)
        for i in range(left, right):
            m = self.marks[i]
            x = int(round((m - start) / spp))
            c.create_line(x, ruler, x, h, fill=MARK, width=max(1, px(1.5)))
            memo = self.memos.get(m, "")
            c.create_text(x + px(4), ruler + px(3), text=f"{i + 1} {memo}" if memo else str(i + 1), anchor="nw", fill=MARK_TEXT, font=self.f_tiny)
        if self.mode != LIVE and 0 <= x1 <= w:
            c.create_line(x1, ruler, x1, h, fill="#8A8D92")
            label = "LIVE" if e.recording else "END" if self.source == "file" else "REC END"
            c.create_text(x1 + px(5), h - px(6), text=label, anchor="sw", fill=MUTED, font=self.f_mono)
        pos = self.position()
        x = int(round((pos - start) / spp))
        playing = self.mode == SEARCH and e.cue.playing()
        color = {LIVE: TEXT, SEARCH: YELLOW, ONLINE: RED_CURSOR}[self.mode]
        if 0 <= x <= w:
            c.create_line(x, 0, x, h, fill=color, width=px(2))
            if self.mode == SEARCH and not playing:
                t8 = px(7)
                c.create_polygon(x - t8, ruler, x + t8, ruler, x, ruler + px(10), fill=color, width=0)
                c.create_text(x + px(8), ruler + px(3), text="STANDBY", anchor="nw", fill=color, font=self.f_small_b)
        if self.loading:
            self.draw_spinner(c, w, h)

    def draw_spinner(self, c, w, h):
        px = self.px
        job = self.loading
        text = f"読込中　{job['path'].name}　{int(job['progress'] * 100)}%"
        font = tkfont.Font(font=self.f_small)
        tw = font.measure(text)
        bw, bh = tw + px(44), px(30)
        x1, y1 = w - px(14), h - px(12)
        x0, y0 = x1 - bw, y1 - bh
        round_rect(c, x0, y0, x1, y1, bh / 2, fill="#15171A", outline=LINE)
        r = px(8)
        cx, cy = x0 + px(18), (y0 + y1) / 2
        c.create_oval(cx - r, cy - r, cx + r, cy + r, outline=LINE, width=px(3))
        c.create_arc(cx - r, cy - r, cx + r, cy + r, start=-(self.spin * 30) % 360, extent=100, style="arc", outline=TEAL, width=px(3))
        c.create_text(cx + r + px(8), cy, text=text, anchor="w", fill=MUTED, font=self.f_small)

    def update_labels(self):
        e = self.engine
        rate = e.rate
        self.pos_label.configure(text=fmt_clock(self.position(), rate))
        playing = self.mode == SEARCH and e.cue.playing()
        if self.mode == ONLINE:
            note = "本線に出しています"
        elif self.mode == SEARCH:
            note = "検聴中" if playing else "スタンバイ位置"
        elif e.recording:
            note = "録音の先端"
        elif self.source == "file":
            note = "ファイルの終わり"
        elif e.timeline.head:
            note = "止めた位置"
        else:
            note = "録音していません"
        self.pos_note.configure(text=note)
        title = f"読込WAV　{self.file_path.name}" if self.source == "file" and self.file_path else "録音（録りっぱなし）"
        if self.wave_title.cget("text") != title:
            self.wave_title.configure(text=title)
        head = e.timeline.head / rate
        self.length_label.configure(text=f"LENGTH {int(head // 3600):02d}:{int(head % 3600 // 60):02d}:{int(head % 60):02d}　／　MARKS {len(self.marks)}")
        self.mark_count.configure(text=str(len(self.marks)))
        online = self.mode == ONLINE
        self.draw_rec_button()
        state = "memo" if self.editing is not None else "online" if online else "normal"
        self.enter_block.redraw(state)
        blocked = e.recording or online or bool(self.loading)
        fb = self.file_buttons
        fb["load"].set(enabled=not blocked)
        fb["open"].set(enabled=not blocked)
        fb["new"].set(enabled=not blocked)
        fb["save"].set(enabled=not online and not self.loading and self.audio_path() is not None)
        fb["settings"].set(enabled=not blocked)
        self.speed_pills.set_enabled(not online)
        self.pitch_button.set(style="teal" if self.v_keep_pitch.get() else "off", enabled=not online)
        for key, button in self.transport.items():
            if key == "play":
                button.set(kind="stop" if (online or playing) else "play", enabled=not online)
            else:
                button.set(enabled=not online)
        focused = bool(str(self.root.tk.call("focus"))) if not hasattr(self, "_test_focus") else self._test_focus()
        if focused == self.not_active_shown:
            self.not_active_shown = not focused
            if self.not_active_shown:
                self.not_active.pack(side="left", padx=(0, self.px(10)), after=self.pos_note)
            else:
                self.not_active.pack_forget()
        if self.notice and time.monotonic() >= self.notice_until:
            self.notice = ""
            self.update_status()

    def update_status(self):
        e = self.engine
        ok = e.healthy()
        self.brand_lines["in"].configure(text=short_device(self.cfg["input_device"], 26) or "—")
        self.brand_lines["out"].configure(text=short_device(self.cfg["output_device"], 26) or "—")
        session = self.session_path.stem if self.session_path else "未保存"
        self.brand_lines["session"].configure(text=short_device(session, 24) + (" *" if self.dirty and self.session_path else ""))
        self.info_values["session"].configure(text=self.session_path.name if self.session_path else "未保存（保存で名前をつけます）",
                                              fg=TEXT if self.session_path else DIM)
        self.info_values["rec"].configure(text=e.writer.path.name if e.writer is not None else "—", fg=TEXT if e.writer is not None else DIM)
        if self.source == "file" and self.file_path:
            self.info_values["file"].configure(text=self.file_path.name, fg=TEXT)
        else:
            self.info_values["file"].configure(text="なし（音声読込 か ドラッグ＆ドロップ）", fg=DIM)
        if ok == self.reconnect_shown:
            self.reconnect_shown = not ok
            if self.reconnect_shown:
                self.reconnect_button.pack(side="left", padx=(0, self.px(16)), after=self.status_items["state"])
            else:
                self.reconnect_button.pack_forget()
        items = self.status_items
        items["state"].configure(text="CONNECTED" if ok else "NOT CONNECTED", fg=TEAL_SOFT if ok else RED_SOFT)
        items["format"].configure(text=f"{e.rate / 1000:g}kHz / 16bit / MONO")
        items["latency"].configure(text=f"LATENCY {e.latency_ms:.0f} ms" if ok else "LATENCY —")
        items["dropouts"].configure(text=f"DROPOUTS {e.problems}", fg=RED_SOFT if e.problems else MUTED)
        used = len(e.timeline.chunks) * e.timeline.chunk * 2 / 1e9
        items["memory"].configure(text=f"MEMORY {used:.1f} GB")
        try:
            free = shutil.disk_usage(Path(self.cfg["save_dir"]).anchor or ".").free
            items["disk"].configure(text=f"DISK {free / 1e9:.0f} GB", fg=RED_SOFT if free < 2e9 else MUTED)
        except OSError:
            items["disk"].configure(text="DISK —")
        files = []
        if self.session_path:
            files.append(self.session_path.name)
        audio = self.audio_path()
        if audio is not None:
            files.append(Path(audio).name)
        self.status_file.configure(text="　／　".join(files))
        # 普段は空。知らせることがあるときだけ出す
        if self.notice:
            text, color = self.notice, self.notice_color
        elif e.writer is not None and e.writer.error:
            text, color = e.writer.error, RED_SOFT
        elif not ok:
            text, color = (self.engine_error or "オーディオが止まっています"), RED_SOFT
        else:
            text, color = "", YELLOW
        self.notice_label.configure(text=text, fg=color)

    def on_close(self):
        if self.mode == ONLINE:
            self.show_notice("Enter で止めてから閉じてください")
            return
        self.commit_memo()
        if self.engine.recording:
            if not messagebox.askokcancel(APP_NAME, "録音中です。止めて閉じますか？", parent=self.root):
                return
        elif not self.confirm_discard():
            return
        if self.engine.writer is not None:
            self.autosave()
        if self.loading:
            self.loading["cancel"] = True
        self.engine.shutdown()
        self.root.destroy()


def short_device(label, limit=38):
    if not label:
        return ""
    name = label.split("] ", 1)[-1]
    return name if len(name) <= limit else name[:limit - 1] + "…"


class Seg(tk.Canvas):
    """少ない選択肢を横に並べて選ぶ。"""

    def __init__(self, app, parent, options, var, enabled=True):
        super().__init__(parent, height=app.px(33), bg=PANEL, highlightthickness=0, takefocus=0, cursor="hand2")
        self.app = app
        self.options = list(options)
        self.var = var
        self.enabled = enabled
        self.bind("<Configure>", lambda _e: self.redraw())
        self.bind("<ButtonRelease-1>", self._release)
        var.trace_add("write", lambda *_a: self.redraw())

    def _cells(self):
        w = self.winfo_width()
        gap = self.app.px(4)
        cw = (w - gap * (len(self.options) - 1)) / len(self.options)
        return [(i * (cw + gap), i * (cw + gap) + cw) for i in range(len(self.options))]

    def _release(self, event):
        if not self.enabled:
            return
        for (x0, x1), opt in zip(self._cells(), self.options):
            if x0 <= event.x <= x1:
                self.var.set(opt)
                break

    def redraw(self):
        self.delete("all")
        app = self.app
        h = self.winfo_height()
        current = self.var.get()
        for (x0, x1), opt in zip(self._cells(), self.options):
            on = opt == current
            if not self.enabled:
                round_rect(self, x0 + 1, 1, x1 - 1, h - 4, app.px(7), fill=LINE2 if on else BG, outline=LINE2)
                self.create_text((x0 + x1) / 2, (h - 3) / 2, text=opt, fill=MUTED if on else FAINT, font=app.f_mono11_b if on else app.f_mono11)
                continue
            if on:
                top, bottom, outline, fg, font = "#7FDCD0", "#45AFA3", "#8FE6DA", TEAL_TEXT, app.f_mono11_b
            else:
                top, bottom, outline, fg, font = "#33373E", "#24272C", "#3E434B", "#B4BAC2", app.f_mono11
            raised(self, x0 + 1, 1, x1 - 1, h - 4, app.px(7), top, bottom, outline)
            self.create_text((x0 + x1) / 2, (h - 3) / 2, text=opt, fill=fg, font=font)


class SettingsView(tk.Frame):
    """同じウィンドウの中に開く設定画面。まだ反映していない変更は黄色で示す。"""

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
        self.watch = []  # (変数, 最初の値, 点, コンボボックス)
        self.load_devices()

        # 画面いっぱいに広げず、中央に一定の幅で置く
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, minsize=min(px(880), app.root.winfo_screenwidth() - px(40)))
        self.columnconfigure(2, weight=1)
        content = tk.Frame(self, bg=BG)
        content.grid(row=0, column=1, sticky="new", pady=(px(24), 0))

        header = tk.Frame(content, bg=BG)
        header.pack(fill="x", pady=(0, px(14)))
        tk.Label(header, text="設定", font=app.f_h1, fg=TEXT, bg=BG).pack(side="left")
        tk.Label(header, text=APP_NAME, font=app.f_small, fg=MUTED, bg=BG).pack(side="left", padx=px(12))
        Btn(app, header, text="BACK   ESC", on_click=self.back, height=30, radius=8, parent_bg=BG).pack(side="right")
        legend = tk.Frame(header, bg=BG)
        legend.pack(side="right", padx=px(16))
        tk.Label(legend, text="●", font=app.f_small, fg=YELLOW, bg=BG).pack(side="left")
        tk.Label(legend, text="まだ反映していない変更", font=app.f_small, fg=MUTED, bg=BG).pack(side="left", padx=(px(4), 0))

        panels = tk.Frame(content, bg=BG)
        panels.pack(fill="x")
        panels.columnconfigure((0, 1), weight=1, uniform="panel")
        left_panel = RoundPanel(app, panels, pad=16)
        left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, px(7)))
        right_panel = RoundPanel(app, panels, pad=16)
        right_panel.grid(row=0, column=1, sticky="nsew", padx=(px(7), 0))
        left, right = left_panel.inner, right_panel.inner
        self.title(left, "INPUT", "録音する入力")
        self.c_in = self.combo(left, "デバイス", self.v_in, [d[1] for d in self.ins])
        self.c_in_ch = self.combo(left, "CH", self.v_in_ch, [])
        rate_locked = bool(app.engine.timeline.head or app.engine.writer is not None)
        self.seg(left, "SAMPLE RATE", self.v_rate, ["44100", "48000", "96000"], "録音や読み込みがあるときは変えられません" if rate_locked else "", enabled=not rate_locked)
        self.seg(left, "MARK OFFSET（秒）", self.v_offset, ["0", "1", "2", "3", "5", "10"], "マークの何秒前から検聴するか")
        dot = self.field_head(left, "保存フォルダ", "")
        folder = tk.Frame(left, bg=PANEL)
        folder.pack(fill="x")
        tk.Entry(folder, textvariable=self.v_dir, font=app.f_mono11, relief="flat", highlightthickness=1, bg=BG, fg=TEXT, insertbackground=TEXT,
                 highlightbackground=LINE2, highlightcolor=TEAL).pack(side="left", fill="x", expand=True, ipady=px(6))
        Btn(app, folder, text="参照", on_click=self.pick_dir, height=30, radius=7).pack(side="left", padx=(px(6), 0))
        self.track(self.v_dir, dot)

        self.title(right, "OUTPUT", "L 本線 ／ R 検聴")
        self.c_out = self.combo(right, "デバイス", self.v_out, [d[1] for d in self.outs])
        self.c_main = self.combo(right, "MAIN CH（本線）", self.v_main, [])
        self.c_cue = self.combo(right, "CUE CH（検聴）", self.v_cue, [])
        self.seg(right, "LATENCY", self.v_latency, ["LOW", "HIGH"], "")
        self.seg(right, "EXCLUSIVE（WASAPI 排他）", self.v_exclusive, ["ON", "OFF"], "開けないときは共有で動きます")
        self.c_in.bind("<<ComboboxSelected>>", lambda _e: self.update_channels(), add="+")
        self.c_out.bind("<<ComboboxSelected>>", lambda _e: self.update_channels(), add="+")

        footer = RoundPanel(app, content, pad=10)
        footer.pack(fill="x", pady=(px(14), 0))
        tk.Label(footer.inner, text="状態", font=app.f_small, fg=MUTED, bg=PANEL).pack(side="left", padx=(px(4), px(10)))
        tk.Label(footer.inner, text="設定を開いている間は、音を止めています", font=app.f_body, fg=TEXT, bg=PANEL).pack(side="left")
        Btn(app, footer.inner, text="反映する", on_click=self.apply, height=36, radius=8, font=app.f_body_b, style="primary", pad=24).pack(side="right")
        Btn(app, footer.inner, text="デバイス再検索", on_click=self.rescan, height=36, radius=8, font=app.f_body_b, pad=16).pack(side="right", padx=px(8))
        self.update_channels()
        self.refresh_changes()

    def title(self, parent, text, sub):
        px = self.app.px
        row = tk.Frame(parent, bg=PANEL)
        row.pack(fill="x")
        tk.Label(row, text=text, font=self.app.f_h2, fg=TEXT, bg=PANEL).pack(side="left")
        tk.Label(row, text=sub, font=self.app.f_small, fg=MUTED, bg=PANEL).pack(side="left", padx=px(10))
        tk.Frame(parent, bg=LINE, height=1).pack(fill="x", pady=(px(8), 0))

    def field_head(self, parent, label, note):
        px = self.app.px
        row = tk.Frame(parent, bg=PANEL)
        row.pack(fill="x", pady=(px(12), px(5)))
        dot = tk.Label(row, text="●", font=self.app.f_tiny, fg=YELLOW, bg=PANEL)
        tk.Label(row, text=label, font=self.app.f_small, fg=MUTED, bg=PANEL).pack(side="left")
        if note:
            tk.Label(row, text=note, font=self.app.f_tiny, fg=DIM, bg=PANEL).pack(side="right")
        dot.row = row
        return dot

    def combo(self, parent, label, var, values):
        dot = self.field_head(parent, label, "")
        box = ttk.Combobox(parent, textvariable=var, values=values, state="readonly", style="Dark.TCombobox", font=self.app.f_body)
        box.pack(fill="x")
        box.bind("<<ComboboxSelected>>", lambda _e: self.focus_set(), add="+")
        self.track(var, dot, box)
        return box

    def seg(self, parent, label, var, options, note, enabled=True):
        dot = self.field_head(parent, label, note)
        Seg(self.app, parent, options, var, enabled=enabled).pack(fill="x")
        self.track(var, dot)

    def track(self, var, dot, box=None):
        self.watch.append((var, var.get(), dot, box))
        var.trace_add("write", lambda *_a: self.refresh_changes())

    def refresh_changes(self):
        for var, first, dot, box in self.watch:
            changed = var.get() != first
            if changed and not dot.winfo_ismapped():
                dot.pack(side="left", padx=(0, self.app.px(4)), before=dot.row.winfo_children()[1])
            elif not changed and dot.winfo_ismapped():
                dot.pack_forget()
            if box is not None:
                box.configure(style="Changed.TCombobox" if changed else "Dark.TCombobox")

    def load_devices(self):
        self.ins = list_devices("in")
        self.outs = list_devices("out")

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


def make_root():
    """ドラッグ＆ドロップが使えるウィンドウを作る。部品が無ければ普通のウィンドウ。"""
    try:
        from tkinterdnd2 import TkinterDnD
        return TkinterDnD.Tk(), True
    except Exception:
        log_exception()
        return tk.Tk(), False


def check_files():
    """WAVの読み込み（形式とサンプルレートの変換）と、セッションの保存・読み込みを確かめる。"""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        src_rate, secs = 44100, 2.0
        n = int(src_rate * secs)
        t = np.arange(n) / src_rate
        left = 0.5 * np.sin(2 * np.pi * 1000 * t)
        right = 0.5 * np.sin(2 * np.pi * 1000 * t + 0.3)
        v = np.round(np.stack([left, right], axis=1) * 8388607).astype(np.int32).reshape(-1)
        raw = np.stack([v & 0xFF, (v >> 8) & 0xFF, (v >> 16) & 0xFF], axis=1).astype(np.uint8).tobytes()
        path = d / "test24.wav"
        with open(path, "wb") as f:
            f.write(struct.pack("<4sI4s4sIHHIIHH4sI", b"RIFF", 36 + len(raw), b"WAVE", b"fmt ", 16,
                                1, 2, src_rate, src_rate * 6, 6, 24, b"data", len(raw)))
            f.write(raw)
        tl = load_timeline(path, 48000)
        if abs(tl.head - int(48000 * secs)) > 2:
            raise RuntimeError(f"読み込んだ長さが違います: {tl.head}")
        y = tl.read(4800, 48000)
        spec = np.abs(np.fft.rfft(y * np.hanning(len(y))))
        peak_hz = np.argmax(spec) * 48000 / len(y)
        level = np.sqrt(np.mean(y ** 2)) * np.sqrt(2)
        if abs(peak_hz - 1000) > 2 or not 0.45 < level < 0.52:
            raise RuntimeError(f"読み込んだ音が違います: {peak_hz:.1f}Hz, {level:.3f}")
        session = d / "s.json"
        write_json(session, {"app": "ChallengeREC", "version": SESSION_VERSION, "audio": str(path), "audio_relative": "test24.wav",
                             "marks": [{"time": 0.5, "memo": "噛み"}]})
        audio, marks, _data = read_session(session)
        if audio != path or marks != [(0.5, "噛み")]:
            raise RuntimeError("セッションを読めません")


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
        check_files()
        root, dnd = make_root()
        root.withdraw()
        root.update()
        root.destroy()
        if not dnd:
            raise RuntimeError("ドラッグ＆ドロップの部品（tkdnd）が読み込めません")
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
    root, dnd = make_root()
    root.report_callback_exception = log_exception
    App(root, dnd)
    root.mainloop()


if __name__ == "__main__":
    main()
