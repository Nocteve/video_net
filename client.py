import cv2
import socket
import threading
import time
import logging
import numpy as np
import struct
from reedsolo import RSCodec

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
PERF_MONITOR = {"recv_fps": 0, "last_frame_time": time.time()}

class VideoClient:
    def __init__(self, server_ip='127.0.0.1', port=9999):
        self.server_ip = server_ip
        self.port = port
        self.rs = RSCodec(1)
        self.last_frame = np.zeros((720, 1280, 3), dtype=np.uint8)  # 防黑屏缓存
        self.buffer = {}  # {frame_id: {chunk_index: data}}
        self.expected_frames = {}  # {frame_id: total_chunks}
        self.reconnect_count = 0

        self._init_socket()
        threading.Thread(target=self._send_heartbeat, daemon=True).start()

    def _init_socket(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024)
        self.sock.sendto(b'REGISTER', (self.server_ip, self.port))

    def start(self):
        cv2.namedWindow("Video Stream", cv2.WINDOW_NORMAL)
        last_render_time = time.time()
        
        while True:
            try:
                data, _ = self.sock.recvfrom(1024 * 1024)
                PERF_MONITOR["recv_fps"] += 1
                
                # 解析协议头 (帧ID4B + 总分片数2B + 分片索引2B)
                if len(data) < 8: continue
                header = data[:8]
                frame_id, total_chunks, chunk_idx = struct.unpack("!IHH", header)
                chunk_data = data[8:]
                
                # 初始化帧缓冲区
                if frame_id not in self.buffer:
                    self.buffer[frame_id] = {}
                    self.expected_frames[frame_id] = total_chunks
                
                # 存储分片（跳过重复分片）
                if chunk_idx not in self.buffer[frame_id]:
                    self.buffer[frame_id][chunk_idx] = chunk_data
                
                # 帧完整性检查
                if len(self.buffer[frame_id]) == total_chunks:
                    self._assemble_frame(frame_id, total_chunks)
                    del self.buffer[frame_id]
                    del self.expected_frames[frame_id]

                    # PERF_MONITOR["recv_fps"] += 1

                    # 动态帧率控制（防止渲染阻塞）
                    current_time = time.time()
                    if current_time - last_render_time < 0.01:  # >100fps时跳帧
                        continue
                    last_render_time = current_time
                    
            except (ConnectionResetError, socket.timeout):
                self._reconnect()
            except BlockingIOError:
                time.sleep(0.01)  # 无数据时释放CPU
                
            # 显示性能面板
            self._show_perf_panel()

    def _assemble_frame(self, frame_id, total_chunks):
        try:
            # 排序重组数据
            chunks = [self.buffer[frame_id][i] for i in range(total_chunks)]
            fec_data = b"".join(chunks)
            
            # FEC解码
            frame_data = self.rs.decode(fec_data)[0]
            frame = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)
            
            if frame is not None:
                self.last_frame = frame
                cv2.imshow("Video Stream", self.last_frame)
                
        except Exception as e:
            logging.warning(f"帧 {frame_id} 解码失败: {str(e)}")
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            exit(0)

    def _show_perf_panel(self):
        """实时显示性能指标"""
        current_time = time.time()
        if current_time - PERF_MONITOR["last_frame_time"] >= 1:
            fps = PERF_MONITOR["recv_fps"] / (current_time - PERF_MONITOR["last_frame_time"])
            PERF_MONITOR["recv_fps"] = 0
            PERF_MONITOR["last_frame_time"] = current_time
            
            # 在画面左上角显示性能数据
            display_frame = self.last_frame.copy()
            cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Reconn: {self.reconnect_count}", (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("Video Stream", display_frame)
            print(fps)

    def _send_heartbeat(self):
        while True:
            try:
                self.sock.sendto(b'HEARTBEAT', (self.server_ip, self.port))
            except:
                self._reconnect()
            time.sleep(1)

    def _reconnect(self):
        self.reconnect_count += 1
        self.sock.close()
        time.sleep(1)
        logging.warning(f"第{self.reconnect_count}次重连...")
        self._init_socket()

if __name__ == "__main__":
    client = VideoClient(server_ip="127.0.0.1")
    client.start()