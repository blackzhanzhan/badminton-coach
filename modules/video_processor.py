import cv2
import numpy as np
from typing import List, Tuple
import os

class VideoProcessor:
    """视频处理器，用于提取视频帧和基本处理"""
    
    def __init__(self):
        self.supported_formats = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
    
    def extract_frames(self, video_path: str, max_frames: int = None, skip_frames: int = 1) -> List[np.ndarray]:
        """
        从视频中提取帧
        
        Args:
            video_path: 视频文件路径
            max_frames: 最大提取帧数，None表示提取所有帧
            skip_frames: 跳帧间隔，1表示不跳帧
        
        Returns:
            提取的帧列表
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"视频文件不存在: {video_path}")
        
        # 检查文件格式
        file_ext = os.path.splitext(video_path)[1].lower()
        if file_ext not in self.supported_formats:
            raise ValueError(f"不支持的视频格式: {file_ext}")
        
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频文件: {video_path}")
        
        try:
            frame_count = 0
            extracted_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 跳帧处理
                if frame_count % skip_frames == 0:
                    frames.append(frame.copy())
                    extracted_count += 1
                    
                    # 检查是否达到最大帧数
                    if max_frames and extracted_count >= max_frames:
                        break
                
                frame_count += 1
        
        finally:
            cap.release()
        
        return frames
    
    def get_video_info(self, video_path: str) -> dict:
        """
        获取视频信息
        
        Args:
            video_path: 视频文件路径
        
        Returns:
            包含视频信息的字典
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"视频文件不存在: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频文件: {video_path}")
        
        try:
            info = {
                'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'fps': cap.get(cv2.CAP_PROP_FPS),
                'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                'duration_seconds': 0,
                'codec': None
            }
            
            # 计算时长
            if info['fps'] > 0:
                info['duration_seconds'] = info['frame_count'] / info['fps']
            
            # 获取编码信息
            fourcc = cap.get(cv2.CAP_PROP_FOURCC)
            if fourcc:
                info['codec'] = "".join([chr((int(fourcc) >> 8 * i) & 0xFF) for i in range(4)])
        
        finally:
            cap.release()
        
        return info
    
    def resize_frame(self, frame: np.ndarray, target_size: Tuple[int, int] = None, 
                    scale_factor: float = None) -> np.ndarray:
        """
        调整帧大小
        
        Args:
            frame: 输入帧
            target_size: 目标尺寸 (width, height)
            scale_factor: 缩放因子
        
        Returns:
            调整大小后的帧
        """
        if target_size:
            return cv2.resize(frame, target_size)
        elif scale_factor:
            height, width = frame.shape[:2]
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            return cv2.resize(frame, (new_width, new_height))
        else:
            return frame
    
    def save_frame(self, frame: np.ndarray, output_path: str) -> bool:
        """
        保存单帧图像
        
        Args:
            frame: 要保存的帧
            output_path: 输出路径
        
        Returns:
            是否保存成功
        """
        try:
            # 确保输出目录存在
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            return cv2.imwrite(output_path, frame)
        except Exception as e:
            print(f"保存帧失败: {e}")
            return False
    
    def create_video_from_frames(self, frames: List[np.ndarray], output_path: str, 
                                fps: float = 30.0, codec: str = 'mp4v') -> bool:
        """
        从帧列表创建视频
        
        Args:
            frames: 帧列表
            output_path: 输出视频路径
            fps: 帧率
            codec: 编码器
        
        Returns:
            是否创建成功
        """
        if not frames:
            return False
        
        try:
            # 确保输出目录存在
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # 获取帧尺寸
            height, width = frames[0].shape[:2]
            
            # 创建视频写入器
            fourcc = cv2.VideoWriter_fourcc(*codec)
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            if not out.isOpened():
                return False
            
            # 写入帧
            for frame in frames:
                out.write(frame)
            
            out.release()
            return True
        
        except Exception as e:
            print(f"创建视频失败: {e}")
            return False
    
    def extract_frames_at_timestamps(self, video_path: str, timestamps: List[float]) -> List[np.ndarray]:
        """
        在指定时间戳提取帧
        
        Args:
            video_path: 视频文件路径
            timestamps: 时间戳列表（秒）
        
        Returns:
            提取的帧列表
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"视频文件不存在: {video_path}")
        
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频文件: {video_path}")
        
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            for timestamp in timestamps:
                # 计算帧号
                frame_number = int(timestamp * fps)
                
                # 跳转到指定帧
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                
                ret, frame = cap.read()
                if ret:
                    frames.append(frame.copy())
                else:
                    # 如果无法读取，添加空帧
                    frames.append(None)
        
        finally:
            cap.release()
        
        return frames
    
    def get_frame_at_time(self, video_path: str, timestamp: float) -> np.ndarray:
        """
        获取指定时间的帧
        
        Args:
            video_path: 视频文件路径
            timestamp: 时间戳（秒）
        
        Returns:
            指定时间的帧
        """
        frames = self.extract_frames_at_timestamps(video_path, [timestamp])
        return frames[0] if frames and frames[0] is not None else None
    
    def validate_video_file(self, video_path: str) -> Tuple[bool, str]:
        """
        验证视频文件是否有效
        
        Args:
            video_path: 视频文件路径
        
        Returns:
            (是否有效, 错误信息)
        """
        if not os.path.exists(video_path):
            return False, "文件不存在"
        
        file_ext = os.path.splitext(video_path)[1].lower()
        if file_ext not in self.supported_formats:
            return False, f"不支持的文件格式: {file_ext}"
        
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return False, "无法打开视频文件"
            
            # 尝试读取第一帧
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                return False, "无法读取视频帧"
            
            return True, "视频文件有效"
        
        except Exception as e:
            return False, f"验证失败: {str(e)}"