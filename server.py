import sys 
import os
import time
import cv2
import socket
import threading
import logging
import struct
import numpy as np
from reedsolo import RSCodec
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QPushButton, QFileDialog, QGridLayout, QGroupBox, QMessageBox)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QObject,QEvent
from PyQt5.QtGui import QPainter, QColor, QPen
from PyQt5.QtChart import QChart, QChartView, QLineSeries, QValueAxis
import subprocess
import ffmpeg
from opuslib import Encoder, Decoder

# 配置日志和性能监控
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
PERF_MONITOR = {"sent_frames": 0, "start_time": time.time()}
PORT = 9999
AUDIO_PORT = 9998

class GlobalFlags:
    pause_flag = False
    loss = 0
    lag = 0

class VideoServer(QObject):
    update_stats = pyqtSignal(float, float, float)
    
    def __init__(self, host='0.0.0.0', port=PORT, video_path='demo.mp4'):
        super().__init__()
        self.host = host
        self.port = port
        self.video_path = video_path
        self.clients = {}
        self.heartbeat_interval = 5
        self.rs = RSCodec(1)
        self.frame_id = 0
        self.latency_data = {}
        self.loss_data = {}
        self.monitor_interval = 2
        self.transmission_paused = False
        self.current_throughput = 0.0
        self.current_latency = 0.0
        self.current_loss_rate = 0.0
        self.reload_flag = False
        self.lock = threading.Lock()
        
        # 音画同步新增属性
        self.video_time_base = 0  # 视频时间基准
        self.video_frame_interval = 0.0333  # 30fps视频帧间隔 (1/30)
        self.first_video_timestamp = None  # 首帧时间戳
        
        # 网络优化
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1024 * 1024)
        self.sock.setblocking(False)
        self.sock.bind((self.host, self.port))
        
        # 获取视频基本信息
        # self.cap = cv2.VideoCapture(self.video_path)
        # self.base_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        # self.cap.release()

    def start(self):
        threading.Thread(target=self._heartbeat_checker, daemon=True).start()
        threading.Thread(target=self._handle_messages, daemon=True).start()
        threading.Thread(target=self._stream_video, daemon=True).start()
        threading.Thread(target=self._monitor_network, daemon=True).start()
        logging.info(f"视频服务器启动于 {self.host}:{self.port}")

    def _stream_video(self):
        cap = None  # 初始化为 None
        base_fps = 30  # 默认帧率
        chunk_size = 1400  # 原始分片大小
        cap = cv2.VideoCapture(self.video_path)
        base_fps = cap.get(cv2.CAP_PROP_FPS)
        self.video_frame_interval = 1.0 / base_fps  # 计算实际帧间隔
        
        # 初始化时间基准
        if self.first_video_timestamp is None:
            self.first_video_timestamp = time.time()
            self.video_time_base = self.first_video_timestamp
            
        while True:
            # 视频重载检测
            with self.lock:
                if self.reload_flag:
                    if cap and cap.isOpened():
                        cap.release()
                    cap = cv2.VideoCapture(self.video_path)
                    base_fps = cap.get(cv2.CAP_PROP_FPS) or 30
                    self.video_frame_interval = 1.0 / base_fps
                    self.reload_flag = False
                    logging.info(f"重新加载视频，帧率: {base_fps}fps")

            if cap is None or not cap.isOpened():
                cap = cv2.VideoCapture(self.video_path)
                if not cap.isOpened():
                    logging.error("无法打开视频文件")
                    time.sleep(1)
                    continue
                    
            while GlobalFlags.pause_flag:
                time.sleep(0.1)
                continue
                
            time_start = time.time()
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            # 动态分辨率调整
            if GlobalFlags.lag > 600:
                frame = cv2.resize(frame, (640, 360))
            elif GlobalFlags.lag > 50:
                frame = cv2.resize(frame, (1280, 720))
            else:
                frame = cv2.resize(frame, (1280, 720))
            chunk_size = 1400
            
            # 计算媒体时间戳
            media_timestamp = self.video_time_base
            self.video_time_base += self.video_frame_interval
            
            # 动态压缩质量
            if GlobalFlags.lag > 350 or GlobalFlags.loss > 5:
                quality = 50
            elif GlobalFlags.lag > 200:
                quality = 55
            else:
                quality = 60
                
            _, buffer = cv2.imencode('.jpg', frame, [
                cv2.IMWRITE_JPEG_QUALITY, quality, 
                cv2.IMWRITE_JPEG_OPTIMIZE, 1
            ])
            frame_data = buffer.tobytes()
            
            # 动态FEC
            if GlobalFlags.loss > 10:
                self.rs = RSCodec(4)
            elif GlobalFlags.loss > 6:
                self.rs = RSCodec(2)
            else:
                self.rs = RSCodec(1)
            fec_data = self.rs.encode(frame_data)

            # 分片发送
            num_chunks = (len(fec_data) + chunk_size - 1) // chunk_size
            current_time = time.time()
            for idx in range(num_chunks):
                start = idx * chunk_size
                end = min(start + chunk_size, len(fec_data))
                
                # 包头增加媒体时间戳
                header = struct.pack("!IHHdd", 
                                    self.frame_id, num_chunks, idx, 
                                    current_time, media_timestamp)
                
                for addr in list(self.clients.keys()):
                    try:
                        self.sock.sendto(header + fec_data[start:end], addr)
                        PERF_MONITOR["sent_frames"] += 1
                    except BlockingIOError:
                        pass

            # 性能监控
            if time.time() - PERF_MONITOR["start_time"] > 5:
                self.current_throughput = PERF_MONITOR["sent_frames"] / (time.time() - PERF_MONITOR["start_time"])
                PERF_MONITOR.update({"sent_frames": 0, "start_time": time.time()})
                # self.update_stats.emit(
                #     self.current_throughput,
                #     self.current_latency,
                #     self.current_loss_rate * 100
                # )

            self.frame_id = (self.frame_id + 1) % 2**32
            # 帧率控制
            elapsed = time.time() - time_start
            sleep_time = max(0, self.video_frame_interval - elapsed)
            time.sleep(sleep_time/80000)

    def _handle_messages(self):
        while True:
            try:
                data, addr = self.sock.recvfrom(1024)
                if data.startswith(b'HEARTBEAT'):
                    parts = data.split(b':')
                    if len(parts) == 2:
                        client_time = float(parts[1])
                        server_time = time.time()
                        latency = (server_time - client_time) * 1000
                        
                        if addr in self.latency_data:
                            self.latency_data[addr].append(latency)
                            if len(self.latency_data[addr]) > 10:
                                self.latency_data[addr].pop(0)
                        else:
                            self.latency_data[addr] = [latency]
                    
                    self.clients[addr] = (time.time(), self.frame_id)
                elif data == b'REGISTER':
                    self.clients[addr] = (time.time(), self.frame_id)
                    logging.info(f"新视频客户端注册: {addr}")
                elif data.startswith(b'LOSS_REPORT'):
                    parts = data.split(b':')
                    if len(parts) == 3:
                        try:
                            frame_id = int(parts[1])
                            loss_rate = float(parts[2])
                            self.current_loss_rate=loss_rate 
                            self.loss_data[addr] = loss_rate
                        except ValueError:
                            pass
            except BlockingIOError:
                time.sleep(0.01)

    def _heartbeat_checker(self):
        while True:
            time.sleep(self.heartbeat_interval)
            current_time = time.time()
            
            for addr, (last_beat, _) in list(self.clients.items()):
                if current_time - last_beat > self.heartbeat_interval * 3:
                    logging.warning(f"客户端 {addr} 心跳超时，已移除")
                    if addr in self.latency_data:
                        del self.latency_data[addr]
                    if addr in self.loss_data:
                        del self.loss_data[addr]
                    del self.clients[addr]

    def _monitor_network(self):
        while True:
            time.sleep(self.monitor_interval)
            
            if not self.clients:
                continue
                
            avg_latency = {}
            for addr, latencies in self.latency_data.items():
                if latencies:
                    avg_latency[addr] = sum(latencies) / len(latencies)
            
            status = "视频网络状况:\n"
            for addr in self.clients:
                latency = avg_latency.get(addr, -1)
                loss = self.loss_data.get(addr, -1)
                if loss <= 0.0: 
                    print(loss)
                    loss = 0
                status += f"{addr}: 延时={latency:.1f}ms, 丢包率={loss:.1%}\n"
                GlobalFlags.lag = latency
                GlobalFlags.loss = loss
            logging.info(status)

            total_latency, total_loss, count = 0, 0, 0
            for addr in self.clients:
                latency = avg_latency.get(addr, 0)
                loss = self.loss_data.get(addr, 0)
                total_latency += latency
                total_loss += loss
                count += 1
            
            if count > 0:
                self.current_latency = total_latency / count
                self.current_loss_rate = total_loss / count

    def close(self):
        self.running = False
        
        if self.sock:
            self.sock.close()
            self.sock = None
        
        with self.lock:
            self.clients.clear()

class AudioServer(QObject):
    update_stats = pyqtSignal(float, float, float)
    
    def __init__(self, host='0.0.0.0', port=AUDIO_PORT, video_path='demo.mp4'):
        super().__init__()
        self.host = host
        self.port = port
        self.video_path = video_path
        self.clients = {}
        self.heartbeat_interval = 5
        self.latency_data = {}
        self.lock = threading.Lock()
        self.encoder = None
        self.ffmpeg = None
        self.sample_rate = 24000
        self.channels = 2
        self.frame_size = 480  # 20ms帧大小
        self.frame_duration = 0.02
        self.current_throughput = 0.0
        self.current_latency = 0.0
        self.current_loss_rate = 0.0
        self.sequence_id = 0
        self.sent_frames = 0
        self.start_time = time.time()
        self.reload_flag = False
        
        # 音画同步新增属性
        self.audio_time_base = 0  # 音频时间基准
        self.audio_frame_interval = 0.02  # 20ms音频帧间隔
        self.first_audio_timestamp = None  # 首帧时间戳
        
        # 创建Socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1024 * 1024)
        self.sock.setblocking(False)
        self.sock.bind((self.host, self.port))
        
        # 初始化音频编码器
        self._init_encoder()
        self._init_ffmpeg()

    def _init_encoder(self):
        self.encoder = Encoder(self.sample_rate, self.channels, 'audio')
        self.encoder.bitrate = 64000  # 64kbps

    def _init_ffmpeg(self):
        if self.ffmpeg and self.ffmpeg.poll() is None:
            self.ffmpeg.terminate()
            
        self.ffmpeg = subprocess.Popen(
            [
                'ffmpeg',
                '-i', self.video_path,
                '-vn',
                '-ac', str(self.channels),
                '-ar', str(self.sample_rate),
                '-f', 's16le',
                '-c:a', 'pcm_s16le',
                '-loglevel', 'quiet',
                '-'
            ],
            stdout=subprocess.PIPE,
            bufsize=self.frame_size * self.channels * 2 * 10,

            startupinfo=subprocess.STARTUPINFO(
                dwFlags=subprocess.STARTF_USESHOWWINDOW,
                wShowWindow=subprocess.SW_HIDE
            ) if os.name == 'nt' else None
        )

    def start(self):
        # 初始化时间基准
        if self.first_audio_timestamp is None:
            self.first_audio_timestamp = time.time()
            self.audio_time_base = self.first_audio_timestamp
            
        threading.Thread(target=self._stream_audio, daemon=True).start()
        threading.Thread(target=self._handle_messages, daemon=True).start()
        threading.Thread(target=self._heartbeat_checker, daemon=True).start()
        logging.info(f"音频服务器启动于 {self.host}:{self.port}")

    def _stream_audio(self):
        next_frame_time = time.perf_counter()
        MAX_SKEW = 0.005  # 允许的时间偏差5ms
        
        while True:
            with self.lock:
                if self.reload_flag:
                    self._init_ffmpeg()
                    self.reload_flag = False
                    logging.info("音频流重新加载")
            
            if GlobalFlags.pause_flag:
                time.sleep(0.1)
                continue
                
            try:
                # 读取PCM数据
                pcm_data = self.ffmpeg.stdout.read(self.frame_size * self.channels * 2)
                
                # 检测流结束
                if not pcm_data:
                    self.ffmpeg.terminate()
                    self._init_ffmpeg()
                    continue
                if GlobalFlags.loss<3:
                    self.encoder.bitrate=64000
                else:
                    self.encoder.bitrate=32000
                # Opus编码
                opus_data = self.encoder.encode(pcm_data, self.frame_size)
                
                # 计算媒体时间戳
                media_timestamp = self.audio_time_base
                self.audio_time_base += self.audio_frame_interval
                
                # 添加时间戳头
                timestamp = time.time()
                header = struct.pack('!ddI', timestamp, media_timestamp, self.sequence_id)
                packet = header + opus_data
                
                # 发送给所有客户端
                for addr in list(self.clients.keys()):
                    try:
                        self.sock.sendto(packet, addr)
                        self.sent_frames += 1
                    except BlockingIOError:
                        pass
                
                # 更新性能统计
                elapsed = time.time() - self.start_time
                if elapsed > 5:
                    self.current_throughput = self.sent_frames / elapsed
                    self.sent_frames = 0
                    self.start_time = time.time()
                    self.update_stats.emit(
                        self.current_throughput,
                        self.current_latency,
                        self.current_loss_rate * 100
                    )
                
                self.sequence_id += 1
                
                # 精确时间控制
                next_frame_time += self.frame_duration
                current_time = time.perf_counter()
                wait_time = next_frame_time - current_time
                
                if wait_time > MAX_SKEW:
                    time.sleep((wait_time - MAX_SKEW))
                elif wait_time < -MAX_SKEW:
                    next_frame_time = current_time

            except Exception as e:
                logging.error(f"音频流错误: {str(e)}")
                time.sleep(0.1)

    def _handle_messages(self):
        while True:
            try:
                data, addr = self.sock.recvfrom(1024)
                if data.startswith(b'HEARTBEAT'):
                    parts = data.split(b':')
                    if len(parts) == 2:
                        client_time = float(parts[1])
                        server_time = time.time()
                        latency = (server_time - client_time) * 1000
                        
                        if addr in self.latency_data:
                            self.latency_data[addr].append(latency)
                            if len(self.latency_data[addr]) > 10:
                                self.latency_data[addr].pop(0)
                        else:
                            self.latency_data[addr] = [latency]
                    
                    self.clients[addr] = time.time()
                elif data == b'REGISTER':
                    self.clients[addr] = time.time()
                    logging.info(f"新音频客户端注册: {addr}")
            except BlockingIOError:
                time.sleep(0.01)

    def _heartbeat_checker(self):
        while True:
            time.sleep(self.heartbeat_interval)
            current_time = time.time()
            
            for addr, last_beat in list(self.clients.items()):
                if current_time - last_beat > self.heartbeat_interval * 3:
                    logging.warning(f"音频客户端 {addr} 心跳超时，已移除")
                    if addr in self.latency_data:
                        del self.latency_data[addr]
                    del self.clients[addr]
                    
            # 计算平均延迟
            if self.latency_data:
                total_latency = 0
                count = 0
                for latencies in self.latency_data.values():
                    if latencies:
                        total_latency += sum(latencies) / len(latencies)
                        count += 1
                if count > 0:
                    self.current_latency = total_latency / count


    def close(self):
        """释放音频服务器所有资源"""
        self.running = False
        
        if self.ffmpeg:
            
            self.ffmpeg.terminate()
            try:
                # 等待进程退出，超时则强制杀死
                self.ffmpeg.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                self.ffmpeg.kill()
            self.ffmpeg = None
        
        if self.encoder:
            self.encoder = None
        
        if self.sock:
            self.sock.close()
            self.sock = None
        
class ServerUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.video_server = VideoServer()
        self.audio_server = AudioServer()
        self.init_ui()
        self.setup_charts()
        self.current_length = 60

        # 连接信号
        #self.video_server.update_stats.connect(self.update_video_stats)
        #self.audio_server.update_stats.connect(self.update_audio_stats)

        # 启动服务器线程
        self.video_thread = QThread()
        self.video_server.moveToThread(self.video_thread)
        self.video_thread.started.connect(self.video_server.start)
        self.video_thread.start()
        
        self.audio_thread = QThread()
        self.audio_server.moveToThread(self.audio_thread)
        self.audio_thread.started.connect(self.audio_server.start)
        self.audio_thread.start()

    def init_ui(self):
        self.setWindowTitle("传输服务端")
        self.setGeometry(100, 100, 1200, 800)
        self.setStyleSheet("""
            QMainWindow { background-color: #1E1E1E; }
            QGroupBox {
                background-color: #252526;
                border: 1px solid #3C3C3C;
                color: #9CDCFE;
                font-weight: bold;
                border-radius: 5px;
                margin-top: 1ex;
            }
            QPushButton {
                background-color: #0057B8;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #0078D7; }
            QPushButton:pressed { background-color: #004B8D; }
            QChart { background-color: #2D2D30; }
        """)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QGridLayout(central_widget)

        # 顶部控制区域
        control_group = QGroupBox("传输控制")
        control_layout = QHBoxLayout()
        self.btn_select = QPushButton("选择视频")
        self.btn_pause = QPushButton("暂停传输")
        self.btn_select.clicked.connect(self.select_video)
        self.btn_pause.clicked.connect(self.toggle_pause)
        control_layout.addWidget(self.btn_select)
        control_layout.addWidget(self.btn_pause)
        control_layout.addStretch()
        control_group.setLayout(control_layout)
        layout.addWidget(control_group, 0, 0, 1, 2)

        # 实时数据显示面板
        stats_group = QGroupBox("实时数据")
        stats_layout = QGridLayout()
        
        # 视频统计
        self.video_labels = [
            QLabel("视频吞吐量: -- fps"),
            QLabel("视频延时: -- ms"),
            QLabel("视频丢包率: -- %")
        ]
        
        # 音频统计
        self.audio_labels = [
            QLabel("音频吞吐量: -- fps"),
            QLabel("音频延时: -- ms"),
            QLabel("音频丢包率: -- %")
        ]
        
        # 设置标签样式
        label_style = """
            font-size: 14px; 
            font-weight: bold; 
            padding: 5px;
            border-radius: 5px;
            min-width: 150px;
            color: #FFFFFF;
        """
        
        for i, label in enumerate(self.video_labels):
            label.setStyleSheet(label_style + "background-color: rgba(0, 0, 0, 150); color: #FFFFFF;")
            stats_layout.addWidget(label, 0, i)
            
        for i, label in enumerate(self.audio_labels):
            label.setStyleSheet(label_style + "background-color: #7F3A3A; color: white;")
            #stats_layout.addWidget(label, 1, i)
            
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group, 1, 0, 1, 2)

        # 性能图表区域
        self.chart_views = [QChartView() for _ in range(3)]
        chart_titles = ["吞吐量 (fps)", "网络延时 (ms)", "丢包率 (%)"]
        
        for i, view in enumerate(self.chart_views):
            view.setMinimumSize(400, 300)
            view.setRenderHint(QPainter.Antialiasing)
            layout.addWidget(view, 2, i % 2, 1, 1)

    def setup_charts(self):
        self.series = {
            'video_throughput': QLineSeries(),
            'latency': QLineSeries(),
            'loss': QLineSeries()
        }
        
        colors = {
            'video_throughput': QColor(65, 105, 225),  # 蓝色
            'latency': QColor(220, 20, 60),             # 红色
            'loss': QColor(255, 165, 0)                # 橙色
        }
        
        # 设置图表样式
        for name, series in self.series.items():
            pen = series.pen()
            pen.setWidth(3)
            pen.setColor(colors[name])
            series.setPen(pen)
            
            # 创建图表
            chart = QChart()
            chart.addSeries(series)
            
            # 设置坐标轴
            axis_x = QValueAxis()
            axis_y = QValueAxis()
            
            # 根据数据类型设置范围
            if 'throughput' in name:
                axis_y.setRange(0, 100)
                chart.setTitle("吞吐量 (fps)")
            elif 'latency' in name:
                axis_y.setRange(0, 500)
                chart.setTitle("网络延时 (ms)")
            else:
                axis_y.setRange(0, 10)
                chart.setTitle("丢包率 (%)")
                
            axis_x.setRange(0, 60)
            axis_x.setGridLineVisible(True)
            axis_y.setGridLineVisible(True)
            axis_x.setGridLineColor(QColor(100, 100, 100, 150))
            axis_x.setMinorGridLineColor(QColor(80, 80, 80, 100))
            axis_y.setMinorGridLineVisible(True)
            axis_y.setGridLineColor(QColor(100, 100, 100, 150))
            axis_y.setMinorGridLineColor(QColor(80, 80, 80, 100))
            
            chart.addAxis(axis_x, Qt.AlignBottom)
            chart.addAxis(axis_y, Qt.AlignLeft)
            series.attachAxis(axis_x)
            series.attachAxis(axis_y)
            
            # 存储图表
            if 'throughput' in name:
                idx = 0
            elif 'latency' in name:
                idx = 1
            else:
                idx = 2
            self.chart_views[list(self.series.keys()).index(name)].setChart(chart)

    def select_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择视频文件", "", "视频文件 (*.mp4 *.avi)"
        )
        if path:
            with self.video_server.lock:
                self.video_server.video_path = path
                self.video_server.reload_flag = True
                
            with self.audio_server.lock:
                self.audio_server.video_path = path
                self.audio_server.reload_flag = True
                
            logging.info(f"已选择新视频: {path}")

    def toggle_pause(self):
        GlobalFlags.pause_flag = not GlobalFlags.pause_flag
        self.btn_pause.setText("继续传输" if GlobalFlags.pause_flag else "暂停传输")

    def update_video_stats(self):
        throughput=self.video_server.current_throughput
        latency=self.video_server.current_latency
        loss=GlobalFlags.loss*100
        self.video_labels[0].setText(f"视频吞吐量: {throughput:.1f} fps")
        self.video_labels[1].setText(f"视频延时: {latency:.1f} ms")
        self.video_labels[2].setText(f"视频丢包率: {loss:.1f} %")
        
        # 更新图表
        self._update_series('video_throughput', throughput)
        self._update_series('latency', latency)
        self._update_series('loss', loss)

    def update_audio_stats(self, throughput, latency, loss):
        self.audio_labels[0].setText(f"音频吞吐量: {throughput:.1f} fps")
        self.audio_labels[2].setText(f"音频丢包率: {loss:.1f} %")

        self._update_series('audio_throughput', throughput)
    
    def _update_series(self, series_name, value):
        series = self.series[series_name]
        
        # 添加新数据点
        #x = series.count()
        #series.append(x, value)
        
        # 保持数据长度
        if series.count() > self.current_length:
            series.removePoints(0, 1)
            self.current_length+=1

        series.append(series.count(), value)    
        # 更新X轴范围
        # chart_view = None
        # if series_name == 'video_throughput' or series_name == 'audio_throughput':
        #     chart_view = self.chart_views[0]
        # elif series_name == 'latency':
        #     chart_view = self.chart_views[1]
        # elif series_name == 'loss':
        #     chart_view = self.chart_views[2]
            
        if series.count() > 60:
            chart_view = self.chart_views[list(self.series.keys()).index(series_name)]
            axis_x = chart_view.chart().axes(Qt.Horizontal)[0]
            axis_x.setRange(series.count() - 60, series.count())

    def closeEvent(self, event: QEvent):
        """窗口关闭时释放所有资源"""
        self.video_server.close()
        self.audio_server.close()
        
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ServerUI()
    window.show()
    timer = QTimer()
    timer.timeout.connect(window.update_video_stats)
    timer.start(1000)
    sys.exit(app.exec_())
