import sys
import cv2
import socket
import threading
import time
import logging
import struct
import queue
import collections
import numpy as np
from reedsolo import RSCodec
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QLabel, QHBoxLayout, QGroupBox, QGridLayout)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor
import pyaudio
from opuslib import Decoder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

class GlobalFlags: 
    loss = 0
    lag = 0
    fps = 30
    t=0
class SyncPlayer:
    def __init__(self, video_frame_interval=1/30, audio_frame_interval=0.02):
        self.video_queue = collections.deque(maxlen=200)  # 视频帧队列
        self.audio_queue = collections.deque(maxlen=200)  # 音频帧队列
        self.video_frame_interval = video_frame_interval
        self.audio_frame_interval = audio_frame_interval
        self.base_time = None
        self.lock = threading.Lock()
        self.start_real_time = None
        self.last_video_ts = 0
        self.last_audio_ts = 0
        self.sync_status = ""
        self.sync_color = "#7F7F7F"
        self.last_frame_time = time.time()
        self.video_offset = 0  # 视频时间偏移
        self.audio_offset = 0  # 音频时间偏移
        self.initialized = False  # 是否已完成初始化
        self.last_frame_id = 0  # 跟踪最后处理的帧ID
        self.last_frame_timestamp = 0  # 跟踪最后处理的帧时间戳
        self.first_video_offset=0
    def add_video(self, timestamp, frame, frame_id):
        with self.lock:
            # 修正时间戳，考虑时间偏移
            adjusted_ts = timestamp + self.video_offset
            
            # 检查帧ID是否连续
            if frame_id < self.last_frame_id:
                logging.warning(f"收到过时视频帧 (ID: {frame_id} < {self.last_frame_id})")
                return
                
            # 更新最后帧ID
            self.last_frame_id = frame_id
            
            # 过滤过时的帧（超过1秒）
            if self.initialized and self.get_current_media_time() - adjusted_ts > 20:
                logging.warning(f"过滤过时视频帧 (ID: {frame_id}, 时间差: {time.time() - adjusted_ts:.3f}s)")
                return
                
            # 添加到队列
            self.video_queue.append((adjusted_ts, frame, frame_id))
            
    def add_audio(self, timestamp, pcm_data, sequence_id):
        with self.lock:
            # 修正时间戳，考虑时间偏移
            adjusted_ts = timestamp + self.audio_offset
            
            # 过滤过时的音频帧（超过1秒）
            if self.initialized and time.time() - adjusted_ts > 0.4:
                return
                
            # 添加到队列
            self.audio_queue.append((adjusted_ts, pcm_data, sequence_id))
            
    def _init_base_time(self):
        """初始化基准时间（改进的时间基准计算）"""
        if self.base_time is not None:
            return
            
        if not self.video_queue or not self.audio_queue:
            # 如果队列都为空，使用当前时间作为基准
            self.base_time = time.time()
            self.start_real_time = time.time()
            self.initialized = True
            return
            
        # 获取队列中的最小时间戳
        video_min = min(ts for ts, _, _ in self.video_queue) if self.video_queue else float('inf')
        audio_min = min(ts for ts, _, _ in self.audio_queue) if self.audio_queue else float('inf')
        
        if video_min == float('inf') and audio_min == float('inf'):
            # 如果没有有效时间戳，使用当前时间
            self.base_time = time.time()
            self.start_real_time = time.time()
            self.initialized = True
            return
            
        # 计算时间偏移量
        current_time = time.time()
        self.video_offset = current_time - video_min if video_min != float('inf') else 0
        self.audio_offset = current_time - audio_min if audio_min != float('inf') else 0
        
        # 重新计算所有帧的时间戳
        with self.lock:
            # 更新视频队列时间戳
            new_video_queue = []
            for ts, frame, frame_id in self.video_queue:
                adjusted_ts = ts + self.video_offset
                new_video_queue.append((adjusted_ts, frame, frame_id))
            self.video_queue = collections.deque(
                sorted(new_video_queue, key=lambda x: x[0]), 
                maxlen=100
            )
            
            # 更新音频队列时间戳
            new_audio_queue = []
            for ts, pcm_data, sequence_id in self.audio_queue:
                adjusted_ts = ts + self.audio_offset
                new_audio_queue.append((adjusted_ts, pcm_data, sequence_id))
            self.audio_queue = collections.deque(
                sorted(new_audio_queue, key=lambda x: x[0]), 
                maxlen=200
            )
        
            # 设置基准时间
            video_min = min(ts for ts, _, _ in self.video_queue) if self.video_queue else float('inf')
            audio_min = min(ts for ts, _, _ in self.audio_queue) if self.audio_queue else float('inf')
            self.base_time = min(video_min, audio_min)
            self.start_real_time = time.time()
            self.initialized = True
            logging.info(f"初始化基准时间: {self.base_time:.3f}, 视频偏移: {self.video_offset:.3f}s, 音频偏移: {self.audio_offset:.3f}s")
            self.first_video_offset=self.video_offset
    def get_current_media_time(self):
        """获取当前媒体时间（容错处理）"""
        if self.base_time is None:
            self._init_base_time()
            
        if self.base_time is None:
            return 0
            
        elapsed = time.time() - self.start_real_time
        #print(self.base_time + elapsed)
        if GlobalFlags.loss>2:
            return self.base_time + elapsed
        return GlobalFlags.t+self.base_time-self.audio_offset+self.first_video_offset
    def get_next_video(self):
        """获取下一个应播放的视频帧（改进的帧选择算法）"""
        self._init_base_time()
        if not self.video_queue:
            return None
            
        current_media_time = self.get_current_media_time()
        
        # 查找最接近但不超过当前时间的帧
        best_frame = None
        best_ts = 0
        best_id = 0
        best_index = -1
        
        # 首先尝试找到时间戳最接近当前媒体时间的帧
        # 但优先选择时间戳小于或等于当前媒体时间的帧
        for i, (ts, frame, frame_id) in enumerate(self.video_queue):
            # 确保帧ID连续
            if frame_id >= self.last_frame_timestamp:
                if ts <= current_media_time:
                    if best_frame is None or ts > best_ts:
                        best_frame = frame
                        best_ts = ts
                        best_id = frame_id
                        best_index = i
                else:
                    # 一旦遇到时间戳大于当前媒体时间的帧，停止搜索
                    break
        temp_frame=best_frame
        temp_ts=best_ts
        temp_id=best_id
        temp_index=best_index
        # 最接近的下一帧
        if best_frame is None or self.video_queue:
            # 找到时间戳大于当前媒体时间的最小帧
            for i, (ts, frame, frame_id) in enumerate(self.video_queue):
                if ts >= current_media_time and frame_id >= self.last_frame_timestamp:
                    best_frame = frame
                    best_ts = ts
                    best_id = frame_id
                    best_index = i
                    break
        if abs(best_ts-current_media_time)>abs(temp_ts-current_media_time):
            best_frame=temp_frame
            best_id=temp_id
            best_index=temp_index
            best_ts=temp_ts
        # 记录最后使用的视频时间戳
        if best_frame is not None:
            self.last_video_ts = best_ts
            self.last_frame_timestamp = best_id
            # 从队列中移除该帧
            # if best_index >= 0:
            #     del self.video_queue[best_index]
        
        # 清理过期帧（超过当前时间0.5秒）
        while self.video_queue and self.video_queue[0][0] < current_media_time - 0.2:
            #logging.warning(f"清理过期视频帧 (时间差: {current_media_time - self.video_queue[0][0]:.3f}s)")
            self.video_queue.popleft()
                
        # 避免频繁刷新导致卡顿
        current_time = time.time()
        if current_time - self.last_frame_time < 0.02:  # 最小刷新间隔20ms
            return None
        # print(GlobalFlags.t)
        if abs(best_ts - current_media_time) > 0.3:  # 300ms阈值
            new_offset = current_media_time - best_ts
            self.video_offset = self.video_offset*0.5+new_offset * 0.5  # 平滑调整
        print(best_ts-current_media_time)
        self.last_frame_time = current_time
        return best_frame
        
    def get_next_audio(self):
        """获取下一个应播放的音频帧（改进的帧选择算法）"""
        self._init_base_time()
        if not self.audio_queue:
            return None
            
        current_media_time = self.get_current_media_time()
        
        # 查找最接近但不超过当前时间的帧
        best_audio = None
        best_ts = 0
        best_index = -1
        
        # 首先尝试找到时间戳最接近当前媒体时间的帧
        # 但优先选择时间戳小于或等于当前媒体时间的帧
        for i, (ts, pcm_data, sequence_id) in enumerate(self.audio_queue):
            if ts <= current_media_time:
                if best_audio is None or ts > best_ts:
                    best_audio = pcm_data
                    best_ts = ts
                    best_index = i
            else:
                # 一旦遇到时间戳大于当前媒体时间的帧，停止搜索
                break
        temp_ts=best_ts
        temp_audio=best_audio
        temp_index=best_index
        # 如果没找到小于当前时间的帧，则选择最接近的下一帧
        if best_audio is None or self.audio_queue:
            # 找到时间戳大于当前媒体时间的最小帧
            for i, (ts, pcm_data, sequence_id) in enumerate(self.audio_queue):
                if ts >= current_media_time:
                    best_audio = pcm_data
                    best_ts = ts
                    best_index = i
                    break
        if abs(best_ts-current_media_time)>abs(temp_ts-current_media_time):
            best_index=temp_index
            best_audio=temp_audio
            best_ts=temp_ts
        # 记录最后使用的音频时间戳
        if best_audio is not None:
            self.last_audio_ts = best_ts
            # 从队列中移除该帧
            # if best_index >= 0:
            #     del self.audio_queue[best_index]
        
        # 清理过期帧（超过当前时间1.0秒）
        while self.audio_queue and self.audio_queue[0][0] < current_media_time - 0.4:
            self.audio_queue.popleft()
                
        return best_audio
        
    def update_sync_status(self):
        """更新同步状态（更精确的同步检测）"""
        if self.last_video_ts == 0 or self.last_audio_ts == 0:
            self.sync_status = "未同步"
            self.sync_color = "#7F7F7F"
            return
            
        av_diff = (self.last_video_ts - self.last_audio_ts) * 1000  # 转换为毫秒
        
        if abs(av_diff) < 40:  # 40ms以内视为完美同步
            self.sync_status = f"同步良好 (±{abs(av_diff):.0f}ms)"
            self.sync_color = "#4CAF50"  # 绿色
        elif abs(av_diff) < 80:  # 80ms以内视为可接受
            self.sync_status = f"同步可接受 (±{abs(av_diff):.0f}ms)"
            self.sync_color = "#FFC107"  # 黄色
        else:
            self.sync_status = f"同步问题: {av_diff:.0f}ms"
            self.sync_color = "#F44336"  # 红色

class VideoClient:
    def __init__(self, server_ip='127.0.0.1', port=9999):
        self.server_ip = server_ip
        self.port = port
        self.rs = RSCodec(1)
        self.last_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.buffer = {}
        self.expected_frames = {}
        self.frame_timestamps = collections.deque(maxlen=200)
        self.smoothed_fps = 0.0
        self.alpha = 0.2
        self.latency = -1
        self.loss_rate = 0.0
        self.frame_loss_stats = {}
        self.last_loss_report = 0
        self.loss_report_interval = 2
        self.running = True
        self.frame_handler = None
        self.last_data_time = time.time()
        self.last_frame_id = 0  # 跟踪最后接收的帧ID
        
        # 视频缓冲
        self.frame_buffer = collections.deque(maxlen=100)
        self.buffer_lock = threading.Lock()
        self.target_buffer_size = 50
        self.min_buffer_size = 50
        self.max_buffer_size = 100
        self.buffer_underrun_count = 0
        
        self._init_socket()
        threading.Thread(target=self._send_heartbeat, daemon=True).start()

    def _init_socket(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024)
        self.sock.settimeout(0.1)
        self.sock.sendto(b'REGISTER', (self.server_ip, self.port))

    def start_receiving(self):
        while self.running:
            try:
                data, _ = self.sock.recvfrom(1024 * 1024)
                self.last_data_time = time.time()
                
                if len(data) < 24:
                    continue
                    
                try:
                    frame_id, total_chunks, chunk_idx, server_time, media_timestamp = struct.unpack("!IHHdd", data[:24])
                except struct.error as e:
                    logging.error(f"解包错误: {e}")
                    continue
                    
                chunk_data = data[24:]
                current_time = time.time()
                self.latency = (current_time - server_time) * 1000
                GlobalFlags.lag = self.latency

                # 检查帧ID是否连续
                if frame_id < self.last_frame_id:
                    logging.warning(f"收到过时视频帧 (ID: {frame_id} < {self.last_frame_id})")
                    continue
                    
                self.last_frame_id = frame_id

                if frame_id not in self.buffer:
                    self.buffer[frame_id] = {}
                    self.expected_frames[frame_id] = total_chunks
                    self.frame_loss_stats[frame_id] = {
                        'total': total_chunks,
                        'received': 0,
                        'start_time': time.time()
                    }
                
                if chunk_idx not in self.buffer[frame_id]:
                    self.buffer[frame_id][chunk_idx] = (chunk_data, media_timestamp)
                    self.frame_loss_stats[frame_id]['received'] += 1

                # 检查帧完整性
                if len(self.buffer[frame_id]) == total_chunks:
                    self._assemble_frame(frame_id, total_chunks)
                    del self.buffer[frame_id]
                    del self.expected_frames[frame_id]
                    if frame_id in self.frame_loss_stats:
                        del self.frame_loss_stats[frame_id]
                        
                    # 帧率计算
                    current_time = time.time()
                    self.frame_timestamps.append(current_time)
                    if len(self.frame_timestamps) >= 2:
                        time_span = self.frame_timestamps[-1] - self.frame_timestamps[0]
                        actual_fps = (len(self.frame_timestamps) - 1) / time_span
                        if self.smoothed_fps == 0.0:
                            self.smoothed_fps = actual_fps
                        else:
                            self.smoothed_fps = self.alpha * actual_fps + (1 - self.alpha) * self.smoothed_fps
                        GlobalFlags.fps = self.smoothed_fps
                # 定期报告丢包率
                if current_time - self.last_loss_report >= self.loss_report_interval:
                    self._report_loss_rate()
                    self.last_loss_report = current_time

            except socket.timeout:
                # if time.time() - self.last_data_time > 5.0:
                #     logging.warning("视频连接超时，尝试重连")
                #     self._reconnect()
                #     self.last_data_time = time.time()
                    
                # if time.time() - self.last_loss_report >= self.loss_report_interval:
                #     self._report_loss_rate()
                #     self.last_loss_report = time.time()
                pass
            except (ConnectionResetError, OSError) as e:
                self._reconnect()
            except Exception as e:
                logging.error(f"视频接收错误: {e}")

    def _report_loss_rate(self):
        if not self.frame_loss_stats:
            self.loss_rate = 0.0
            return
        total_lost = 0
        total_chunks = 0
        current_time = time.time()
        
        # 清理超时帧
        for frame_id in list(self.frame_loss_stats.keys()):
            stats = self.frame_loss_stats[frame_id]
            if current_time - stats['start_time'] > 5:
                del self.frame_loss_stats[frame_id]
                total_lost += max(0, stats['total'] - stats['received'])
                total_chunks += stats['total']

        # 统计有效帧
        for frame_id, stats in self.frame_loss_stats.items():
            total_chunks += stats['total']
            received = min(stats['received'], stats['total'])
            #total_lost += max(0, stats['total'] - received)

        # 计算丢包率
        if total_chunks > 0:
            self.loss_rate = total_lost / total_chunks*2
        else:
            self.loss_rate = 0.0
        GlobalFlags.loss = self.loss_rate
        # 发送报告
        report_msg = f"LOSS_REPORT:{int(time.time())}:{self.loss_rate}"
        try:
            self.sock.sendto(report_msg.encode(), (self.server_ip, self.port))
        except:
            pass
            
        # 根据网络状况自适应调整缓冲大小
        if GlobalFlags.loss > 10 or GlobalFlags.lag > 200:
            # 网络差时增加缓冲大小
            new_size = min(self.max_buffer_size, self.target_buffer_size + 1)
            if new_size != self.target_buffer_size:
                with self.buffer_lock:
                    # 创建新尺寸的缓冲队列
                    new_buffer = collections.deque(maxlen=new_size)
                    # 复制现有帧（保留最近帧）
                    for frame in self.frame_buffer:
                        new_buffer.append(frame)
                    self.frame_buffer = new_buffer
                    self.target_buffer_size = new_size
        elif GlobalFlags.loss < 2 and GlobalFlags.lag < 50 and self.target_buffer_size > self.min_buffer_size:
            # 网络好时减少缓冲大小
            new_size = max(self.min_buffer_size, self.target_buffer_size - 1)
            if new_size != self.target_buffer_size:
                with self.buffer_lock:
                    new_buffer = collections.deque(maxlen=new_size)
                    # 只保留最新的帧
                    while len(self.frame_buffer) > new_size:
                        self.frame_buffer.popleft()
                    for frame in self.frame_buffer:
                        new_buffer.append(frame)
                    self.frame_buffer = new_buffer
                    self.target_buffer_size = new_size

    def _assemble_frame(self, frame_id, total_chunks):
        try:
            chunks = [self.buffer[frame_id][i][0] for i in range(total_chunks)]
            fec_data = b"".join(chunks)

            # 获取媒体时间戳
            media_timestamp = self.buffer[frame_id][0][1]
            
            # 动态调整FEC
            if GlobalFlags.loss > 10:
                self.rs = RSCodec(4)
            elif GlobalFlags.loss > 6:
                self.rs = RSCodec(2)
            elif GlobalFlags.loss > 4:
                self.rs = RSCodec(2)
            else:
                self.rs = RSCodec(1)
            frame_data = self.rs.decode(fec_data)[0]
            
            # 使用imdecode的优化参数
            frame = cv2.imdecode(
                np.frombuffer(frame_data, np.uint8), 
                cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
            )
            
            if frame is not None:
                with self.buffer_lock:
                    if len(self.frame_buffer) == self.frame_buffer.maxlen:
                        self.frame_buffer.popleft()
                    self.frame_buffer.append((media_timestamp, frame))
                    self.last_frame = frame
                
                if self.frame_handler and callable(self.frame_handler):
                    self.frame_handler(media_timestamp, frame, frame_id)

        except Exception as e:
            if frame_id in self.frame_loss_stats:
                del self.frame_loss_stats[frame_id]
            logging.error(f"帧重组错误: {e}")
        finally:
            # 释放资源
            if 'frame_data' in locals():
                del frame_data
            if 'fec_data' in locals():
                del fec_data

    def get_frame(self):
        """从缓冲队列获取一帧视频，如果缓冲为空则返回最后一帧"""
        with self.buffer_lock:
            if self.frame_buffer:
                self.buffer_underrun_count = max(0, self.buffer_underrun_count - 1)
                return self.frame_buffer.popleft()
            else:
                self.buffer_underrun_count += 1
                return (0, self.last_frame)

    def _send_heartbeat(self):
        while self.running:
            try:
                heartbeat_msg = f"HEARTBEAT:{time.time()}"
                self.sock.sendto(heartbeat_msg.encode(), (self.server_ip, self.port))
            except:
                self._reconnect()
            time.sleep(1)

    def _reconnect(self):
        self.sock.close()
        time.sleep(1)
        self._init_socket()
        
    def close(self):
        self.running = False
        self.sock.close()

class AudioClient:
    def __init__(self, server_ip='127.0.0.1', port=9998):
        self.server_ip = server_ip
        self.port = port
        self.buffer = queue.Queue(maxsize=600)
        self.stop_event = threading.Event()
        self.initial_buffer_size = 20
        self.sequence_id = 0
        self.last_sequence_id = 0
        self.received_count = 0
        self.dropped_count = 0
        self.latency = -1
        self.running = True
        self.start_time = time.time()
        self.current_audio_ts = 0
        self.frame_handler = None
        self.last_data_time = time.time()
        
        self._init_socket()
        self.decoder = Decoder(24000, 2)
        self.player_thread = threading.Thread(
            target=self._audio_player, 
            args=(self.buffer, self.stop_event, self.initial_buffer_size),
            daemon=True
        )
        self.player_thread.start()

    def _init_socket(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024)
        self.sock.settimeout(0.1)
        self.sock.sendto(b'REGISTER', (self.server_ip, self.port))

    def start_receiving(self):
        threading.Thread(target=self._send_heartbeat, daemon=True).start()
        
        while self.running:
            try:
                data, _ = self.sock.recvfrom(1024 * 4)
                self.last_data_time = time.time()
                
                if len(data) < 20:
                    continue
                    
                try:
                    server_time, media_timestamp, sequence_id = struct.unpack('!ddI', data[:20])
                except struct.error as e:
                    logging.error(f"音频解包错误: {e}")
                    continue
                    
                opus_data = data[20:]
                current_time = time.time()
                self.latency = (current_time - server_time) * 1000
                self.current_audio_ts = media_timestamp

                # 检测序列号不连续
                if sequence_id < self.last_sequence_id:
                    logging.warning(f"收到过时音频帧 (序列: {sequence_id} < {self.last_sequence_id})")
                    self.last_sequence_id = sequence_id
                elif sequence_id > self.last_sequence_id + 1:
                    gap = sequence_id - self.last_sequence_id - 1
                    self.dropped_count += gap
                    logging.warning(f"音频帧丢失: {gap}帧 (序列: {self.last_sequence_id} -> {sequence_id})")
                    
                self.last_sequence_id = sequence_id
                self.received_count += 1
                
                # 解码音频
                try:
                    pcm_data = self.decoder.decode(opus_data, 480)
                except Exception as e:
                    logging.error(f"音频解码错误: {e}")
                    continue
                
                # # 检测过时帧
                # current_time = time.time()
                # if current_time - media_timestamp > 1.0:
                #     self.dropped_count += 1
                #     continue
                
                # 存入缓冲
                if self.buffer.full():
                    try:
                        self.buffer.get_nowait()
                    except queue.Empty:
                        pass
                
                self.buffer.put((media_timestamp, sequence_id, pcm_data))
                
                if self.frame_handler and callable(self.frame_handler):
                    self.frame_handler(media_timestamp, pcm_data, sequence_id)

            except socket.timeout:
                # if time.time() - self.last_data_time > 5.0:
                #     logging.warning("音频连接超时，尝试重连")
                #     self._reconnect()
                #     self.last_data_time = time.time()
                pass
            except (ConnectionResetError, OSError) as e:
                self._reconnect()
            except Exception as e:
                logging.error(f"音频接收错误: {e}")

    def _audio_player(self, buffer, stop_event, initial_buffer_size):
        pa = pyaudio.PyAudio()
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=2,
            rate=24000,
            output=True,
            frames_per_buffer=480,
            start=False
        )
        
        # 初始缓冲
        wait_start = time.time()
        while buffer.qsize() < initial_buffer_size and not stop_event.is_set():
            if time.time() - wait_start > 2.0:
                break
            time.sleep(0.01)
        
        if not stop_event.is_set():
            stream.start_stream()
        
        # 播放控制
        next_play_time = time.perf_counter() + 0.02
        underrun_count = 0
        
        while not stop_event.is_set():
            try:
                # 获取音频帧
                frame_data = buffer.get(timeout=0.1)  # 增大超时时间
                media_ts, seq_id, pcm_data = frame_data
                
                # 播放音频
                stream.write(pcm_data)
                underrun_count = 0
                
                # 更新播放时间
                next_play_time += 0.02
                
                # 计算等待时间
                current_time = time.perf_counter()
                wait_time = next_play_time - current_time
                GlobalFlags.t+=0.02

                if wait_time > 0.001:
                    time.sleep(wait_time)

            except queue.Empty:
                # 缓冲区空时输出静音
                silence = b'\x00' * (480 * 2 * 2)
                stream.write(silence)
                next_play_time = time.perf_counter()
                underrun_count += 1
                # if underrun_count > 200:  # 增加容忍度
                #     next_play_time = time.perf_counter()
                #     logging.warning("音频缓冲不足，重置播放时间")
                    
            except Exception as e:
                logging.error(f"音频播放错误: {e}")
                break
        
        # 清理资源
        stream.stop_stream()
        stream.close()
        pa.terminate()

    def _send_heartbeat(self):
        while self.running:
            try:
                heartbeat_msg = f"HEARTBEAT:{time.time()}"
                self.sock.sendto(heartbeat_msg.encode(), (self.server_ip, self.port))
            except:
                self._reconnect()
            time.sleep(1)

    def _reconnect(self):
        self.sock.close()
        time.sleep(1)
        self._init_socket()
        
    def close(self):
        self.running = False
        self.stop_event.set()
        self.sock.close()

class ClientUI(QMainWindow):
    def __init__(self, server_ip='127.0.0.1', video_port=9999, audio_port=9998):
        super().__init__()
        self.sync_player = SyncPlayer()
        self.video_client = VideoClient(server_ip, video_port)
        self.audio_client = AudioClient(server_ip, audio_port)
        
        # 设置回调函数
        self.video_client.frame_handler = lambda ts, frame, fid: self.sync_player.add_video(ts, frame, fid)
        self.audio_client.frame_handler = lambda ts, pcm, seq: self.sync_player.add_audio(ts, pcm, seq)
        
        self.init_ui()
        self.start_video_stream()
        self.start_audio_stream()
        
        # 深色主题样式
        self.setStyleSheet("""
            QMainWindow { 
                background-color: #1E1E1E; 
                color: #DCDCDC; 
                font-family: Segoe UI, Arial;
            }
            QGroupBox {
                background-color: #252526;
                border: 1px solid #3C3C3C;
                color: #9CDCFE;
                font-weight: bold;
                border-radius: 5px;
                margin-top: 1ex;
                padding: 10px;
            }
            QLabel { 
                font-size: 14px; 
                padding: 4px;
            }
            QLabel#statusLabel {
                font-size: 14px;
                font-weight: bold;
                padding: 8px;
                border-radius: 5px;
                min-width: 150px;
                text-align: center;
            }
            #videoDisplay {
                background-color: black;
                border: 1px solid #3C3C3C;
                border-radius: 4px;
            }
            .warning {
                background-color: #FFC107;
                color: black;
                font-weight: bold;
            }
        """)
        
        # 设置窗口大小
        self.setMinimumSize(1000, 700)

    def init_ui(self):
        self.setWindowTitle("传输客户端")
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(15, 15, 15, 15)

        # 视频显示区域
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(800, 450)
        self.video_label.setObjectName("videoDisplay")
        main_layout.addWidget(self.video_label, 4)

        # 网络状态面板
        status_group = QGroupBox("网络与同步状态")
        status_layout = QGridLayout()
        status_layout.setSpacing(10)
        
        # 视频状态
        self.video_latency_label = QLabel("视频延时: -- ms")
        self.video_latency_label.setObjectName("statusLabel")
        self.video_loss_label = QLabel("视频丢包率: --%")
        self.video_loss_label.setObjectName("statusLabel")
        self.video_fps_label = QLabel("视频帧率: -- fps")
        self.video_fps_label.setObjectName("statusLabel")
        self.video_buffer_label = QLabel("视频缓冲: --/--")
        self.video_buffer_label.setObjectName("statusLabel")
        self.video_frame_id_label = QLabel("视频帧ID: --")
        self.video_frame_id_label.setObjectName("statusLabel")
        
        # 音频状态
        self.audio_latency_label = QLabel("音频延时: -- ms")
        self.audio_latency_label.setObjectName("statusLabel")
        self.audio_loss_label = QLabel("音频丢包率: --%")
        self.audio_loss_label.setObjectName("statusLabel")
        self.audio_fps_label = QLabel("音频帧率: -- fps")
        self.audio_fps_label.setObjectName("statusLabel")
        self.audio_seq_id_label = QLabel("音频序列号: --")
        self.audio_seq_id_label.setObjectName("statusLabel")
        
        # 同步状态
        self.sync_status_label = QLabel("同步状态: --")
        self.sync_status_label.setObjectName("statusLabel")
        
        # 警告标签
        self.warning_label = QLabel("")
        self.warning_label.setVisible(False)
        self.warning_label.setStyleSheet("background-color: #FFC107; color: black; padding: 8px;")
        
        # 添加到布局
        status_layout.addWidget(QLabel("<b>视频状态</b>"), 0, 0, 1, 3)#最后一个数字是有几个标签
        status_layout.addWidget(self.video_latency_label, 1, 0)
        status_layout.addWidget(self.video_loss_label, 1, 1)
        status_layout.addWidget(self.video_fps_label, 1, 2)
        #status_layout.addWidget(self.video_buffer_label, 1, 3)
        status_layout.addWidget(self.video_frame_id_label, 2, 0)
        
        # status_layout.addWidget(QLabel("<b>音频状态</b>"), 3, 0, 1, 4)
        # status_layout.addWidget(self.audio_latency_label, 4, 0)
        # status_layout.addWidget(self.audio_loss_label, 4, 1)
        # status_layout.addWidget(self.audio_fps_label, 4, 2)
        # status_layout.addWidget(self.audio_seq_id_label, 5, 0)
        
        # status_layout.addWidget(QLabel("<b>同步状态</b>"), 6, 0, 1, 4)
        # status_layout.addWidget(self.sync_status_label, 7, 0, 1, 4)
        
        # status_layout.addWidget(self.warning_label, 8, 0, 1, 4)
        
        status_group.setLayout(status_layout)
        main_layout.addWidget(status_group, 1)
        
        # 添加布局分隔
        #main_layout.addStretch(1)

    def start_video_stream(self):
        """启动视频接收线程"""
        self.receive_thread = threading.Thread(target=self.video_client.start_receiving)
        self.receive_thread.daemon = True
        self.receive_thread.start()
        
    def start_audio_stream(self):
        """启动音频接收线程"""
        self.audio_receive_thread = threading.Thread(target=self.audio_client.start_receiving)
        self.audio_receive_thread.daemon = True
        self.audio_receive_thread.start()
        
        # 启动UI更新定时器
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_ui)
        self.timer.start(30)  # 约33fps

    def update_ui(self):
        """更新UI显示"""
        # 获取视频帧
        video_frame = self.sync_player.get_next_video()
        if video_frame is not None:
            # 转换颜色空间
            frame_rgb = cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame_rgb.shape
            bytes_per_line = ch * w
            
            # 创建QImage并显示
            qimg = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)
            
            # 保持宽高比缩放
            self.video_label.setPixmap(pixmap.scaled(
                self.video_label.width(), 
                self.video_label.height(),
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            ))
        
        # 更新网络状态
        self.update_network_status()
        
        # 更新同步状态
        # self.sync_player.update_sync_status()
        # self.sync_status_label.setText(f"同步状态: {self.sync_player.sync_status}")
        # self.sync_status_label.setStyleSheet(f"background-color: {self.sync_player.sync_color}; color: white;")
        
        # 检查网络问题并显示警告
        # self.check_network_issues()
        
    def update_network_status(self):
        """更新网络状态显示"""
        # 视频状态
        self.video_latency_label.setText(f"视频延时: {self.video_client.latency:.1f} ms")
        self.video_loss_label.setText(f"视频丢包率: {self.video_client.loss_rate*100:.1f}%")
        self.video_fps_label.setText(f"视频帧率: {self.video_client.smoothed_fps:.1f} fps")
        
        # 获取当前缓冲状态
        with self.video_client.buffer_lock:
            buffer_size = len(self.video_client.frame_buffer)
            target_size = self.video_client.target_buffer_size
            self.video_buffer_label.setText(f"视频缓冲: {buffer_size}/{target_size}帧")
        
        # 显示视频帧ID
        self.video_frame_id_label.setText(f"视频帧ID: {self.sync_player.last_frame_timestamp}")
        
        # 音频状态
        current_time = time.time()
        audio_fps = self.audio_client.received_count / max(1, current_time - self.audio_client.start_time)
        self.audio_fps_label.setText(f"音频帧率: {audio_fps:.1f} fps")
        self.audio_latency_label.setText(f"音频延时: {self.audio_client.latency:.1f} ms")
        
        # 显示音频序列号
        self.audio_seq_id_label.setText(f"音频序列号: {self.audio_client.last_sequence_id}")
        
        # 计算音频丢包率
        if self.audio_client.last_sequence_id > 0:
            expected_count = self.audio_client.last_sequence_id
            received_count = self.audio_client.received_count
            if expected_count > 0:
                audio_loss = 1 - (received_count / expected_count)
                self.audio_loss_label.setText(f"音频丢包率: {audio_loss*100:.1f}%")
        
        # 根据缓冲状态更新标签颜色
        self.update_label_colors()
        
    def check_network_issues(self):
        """检查网络问题并显示警告"""
        warning_text = ""
        
        # 视频缓冲不足
        with self.video_client.buffer_lock:
            buffer_size = len(self.video_client.frame_buffer)
            target_size = self.video_client.target_buffer_size
            
        if buffer_size < target_size * 0.3:
            warning_text += "视频缓冲不足! "
            
        # 音频丢包率过高
        audio_loss = 0.0
        if self.audio_client.last_sequence_id > 0:
            expected_count = self.audio_client.last_sequence_id
            received_count = self.audio_client.received_count
            if expected_count > 0:
                audio_loss = 1 - (received_count / expected_count)
                if audio_loss > 0.15:  # 丢包率超过15%
                    warning_text += "音频丢包率高! "
        
        # 视频丢包率过高
        if self.video_client.loss_rate > 0.15:
            warning_text += "视频丢包率高! "
            
        # 高延迟
        if self.video_client.latency > 300 or self.audio_client.latency > 300:
            warning_text += "网络延迟高! "
            
        # 同步问题
        if "问题" in self.sync_player.sync_status:
            warning_text += "音视频不同步! "
            
        # 帧ID不连续
        if self.sync_player.last_frame_id > 0 and self.sync_player.last_frame_timestamp < self.sync_player.last_frame_id - 10:
            warning_text += "视频帧跳跃! "
            
        # 显示警告
        if warning_text:
            self.warning_label.setText("警告: " + warning_text)
            self.warning_label.setVisible(True)
            self.warning_label.setStyleSheet("background-color: #FFC107; color: black; padding: 8px;")
        else:
            self.warning_label.setVisible(False)
        
    def update_label_colors(self):
        """根据网络质量更新标签颜色"""
        # 视频延时标签颜色
        latency = self.video_client.latency
        if latency < 0:
            color = "#7F7F7F"  # 灰色 - 无效值
        elif latency < 50:
            color = "#4CAF50"  # 绿色 - 优秀
        elif latency < 100:
            color = "#FFC107"   # 黄色 - 可接受
        else:
            color = "#F44336"   # 红色 - 差
        self.video_latency_label.setStyleSheet(f"background-color: {color}; color: white;")
        
        # 视频丢包率标签颜色
        loss = self.video_client.loss_rate * 100
        if loss < 0:
            color = "#7F7F7F"
        elif loss < 5:
            color = "#4CAF50"
        elif loss < 15:
            color = "#FFC107"
        else:
            color = "#F44336"
        self.video_loss_label.setStyleSheet(f"background-color: {color}; color: white;")
        
        # 视频帧率标签颜色
        fps = self.video_client.smoothed_fps
        if fps > 25:
            color = "#4CAF50"
        elif fps > 15:
            color = "#FFC107"
        else:
            color = "#F44336"
        self.video_fps_label.setStyleSheet(f"background-color: {color}; color: white;")
        
        # 视频缓冲标签颜色
        with self.video_client.buffer_lock:
            buffer_size = len(self.video_client.frame_buffer)
            target_size = self.video_client.target_buffer_size
            
        if buffer_size < target_size * 0.3:
            color = "#F44336"  # 红色 - 不足
        elif buffer_size < target_size * 0.7:
            color = "#FFC107"  # 黄色 - 警告
        else:
            color = "#4CAF50"  # 绿色 - 良好
        self.video_buffer_label.setStyleSheet(f"background-color: {color}; color: white;")
        
        # 音频延时标签颜色
        audio_latency = self.audio_client.latency
        if audio_latency < 0:
            color = "#7F7F7F"
        elif audio_latency < 100:
            color = "#4CAF50"
        elif audio_latency < 200:
            color = "#FFC107"
        else:
            color极 = "#F44336"
        self.audio_latency_label.setStyleSheet(f"background-color: {color}; color: white;")
        
        # 音频丢包率标签颜色
        audio_loss = 0.0
        if self.audio_client.last_sequence_id > 0:
            expected_count = self.audio_client.last_sequence_id
            received_count = self.audio_client.received_count
            if expected_count > 0:
                audio_loss = 1 - (received_count / expected_count)
                if audio_loss < 0:
                    color = "#7F7F7F"
                elif audio_loss < 0.05:
                    color = "#4CAF50"
                elif audio_loss < 0.15:
                    color = "#FFC107"
                else:
                    color = "#F44336"
                self.audio_loss_label.setStyleSheet(f"background-color: {color}; color: white;")

    def closeEvent(self, event):
        """窗口关闭时清理资源"""
        self.video_client.close()
        self.audio_client.close()
        self.timer.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # 设置应用样式
    app.setStyle("Fusion")
    palette = app.palette()
    palette.setColor(palette.Window, QColor(30, 30, 30))
    palette.setColor(palette.WindowText, QColor(220, 220, 220))
    palette.setColor(palette.Base, QColor(25, 25, 25))
    palette.setColor(palette.AlternateBase, QColor(35, 35, 35))
    palette.setColor(palette.Text, QColor(220, 220, 220))
    palette.setColor(palette.Button, QColor(50, 50, 50))
    palette.setColor(palette.ButtonText, QColor(220, 220, 220))
    palette.setColor(palette.Highlight, QColor(0, 87, 184))
    palette.setColor(palette.HighlightedText, Qt.white)
    app.setPalette(palette)
    
    window = ClientUI(server_ip="127.0.0.1")
    window.show()
    sys.exit(app.exec_())