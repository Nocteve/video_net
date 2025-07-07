import cv2
import socket
import threading
import time
import logging
import numpy as np
import struct
from reedsolo import RSCodec

# 配置日志和性能监控
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
PERF_MONITOR = {"sent_frames": 0, "start_time": time.time()}

class VideoServer:
    def __init__(self, host='0.0.0.0', port=9999, video_path='demo.mp4'):
        self.host = host
        self.port = port
        self.video_path = video_path
        self.clients = {}  # {addr: (last_heartbeat, last_frame_id)}
        self.heartbeat_interval = 5
        self.rs = RSCodec(1)  # 基础冗余20字节 这里不是简单正比关系，记得改，客户端也要改
        self.frame_id = 0
        self.weak_network_flag = False  # 弱网检测标志

        # 网络优化：1MB缓冲区 + 非阻塞Socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1024 * 1024)
        self.sock.setblocking(False)
        self.sock.bind((host, port))

    def start(self):
        threading.Thread(target=self._heartbeat_checker, daemon=True).start()
        threading.Thread(target=self._handle_messages, daemon=True).start()
        threading.Thread(target=self._stream_video, daemon=True).start()
        logging.info(f"服务端启动于 {self.host}:{self.port}")

    def _stream_video(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            logging.error("无法打开视频文件")
            return

        # 动态帧率控制（根据客户端数量调整）
        base_fps = cap.get(cv2.CAP_PROP_FPS)
        chunk_size = 1400  # MTU友好分片

        while True:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            # 动态降分辨率（客户端>3时降为720P）
            if len(self.clients) > 3:
                frame = cv2.resize(frame, (1280, 720))
            else:
                frame = cv2.resize(frame, (1920, 1080))
            frame = cv2.resize(frame, (640, 360))
            frame = cv2.resize(frame, (1280, 720))
            # 高效压缩（70%质量 + 快速编码）
            _, buffer = cv2.imencode('.jpg', frame, [
                cv2.IMWRITE_JPEG_QUALITY, 70, 
                cv2.IMWRITE_JPEG_OPTIMIZE, 1
            ])
            frame_data = buffer.tobytes()
            
            # 弱网时增加FEC冗余（心跳超时触发）
            fec_data = self.rs.encode(frame_data) if not self.weak_network_flag else RSCodec(20).encode(frame_data)

            # 分片发送协议（头部：帧ID4B + 总分片数2B + 当前分片2B）
            num_chunks = (len(fec_data) + chunk_size - 1) // chunk_size
            header_base = struct.pack("!IHH", self.frame_id, num_chunks, 0)
            
            for idx in range(num_chunks):
                start = idx * chunk_size
                end = min(start + chunk_size, len(fec_data))
                chunk = fec_data[start:end]
                header = struct.pack("!IHH", self.frame_id, num_chunks, idx)
                
                # 非阻塞发送（避免卡顿）
                for addr in list(self.clients.keys()):
                    try:
                        self.sock.sendto(header + chunk, addr)
                        PERF_MONITOR["sent_frames"] += 1
                    except BlockingIOError:
                        pass  # 缓冲区满时跳过

            # 性能监控日志（每5秒输出吞吐量）
            if time.time() - PERF_MONITOR["start_time"] > 5:
                fps = PERF_MONITOR["sent_frames"] / (time.time() - PERF_MONITOR["start_time"])
                logging.info(f"传输吞吐: {fps:.1f}fps | 客户端数: {len(self.clients)}")
                PERF_MONITOR["sent_frames"] = 0
                PERF_MONITOR["start_time"] = time.time()

            self.frame_id = (self.frame_id + 1) % 2**32
            # print(base_fps)
            # time.sleep(1 / base_fps)  # 精准帧率控制

    def _handle_messages(self):
        while True:
            try:
                data, addr = self.sock.recvfrom(1024*1024*10)
                if data == b'HEARTBEAT':
                    self.clients[addr] = (time.time(), self.frame_id)
                elif data == b'REGISTER':
                    self.clients[addr] = (time.time(), self.frame_id)
                    logging.info(f"新客户端注册: {addr}")
            except BlockingIOError:
                time.sleep(0.01)  # 无数据时释放CPU

    def _heartbeat_checker(self):
        while True:
            time.sleep(self.heartbeat_interval)
            current_time = time.time()
            weak_count = 0
            
            for addr, (last_beat, _) in list(self.clients.items()):
                if current_time - last_beat > self.heartbeat_interval * 2:
                    weak_count += 1
                if current_time - last_beat > self.heartbeat_interval * 3:
                    logging.warning(f"客户端 {addr} 心跳超时，已移除")
                    del self.clients[addr]
            
            # 超过30%客户端弱网时激活高冗余模式
            self.weak_network_flag = weak_count > 0.3 * len(self.clients)

if __name__ == "__main__":
    server = VideoServer(video_path="demo.mp4")
    server.start()
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        server.sock.close()