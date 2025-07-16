import cv2
import numpy as np
import os
import urllib.request
import sys

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class PoseDetector:
    # 支持的模型类型
    MODEL_OPENPOSE_BODY_25 = "openpose_body_25"
    MODEL_OPENPOSE_COCO = "openpose_coco"
    MODEL_MEDIAPIPE = "mediapipe"

    def __init__(self, model_type="mediapipe", min_detection_confidence=0.2, device="cpu"):
        """
        初始化姿势检测器
        
        Args:
            model_type: 要使用的模型类型 ("openpose_coco", "openpose_body_25", "mediapipe")
            min_detection_confidence: 最小检测置信度 (已降低默认值以提高检出率)
            device: 计算设备 ('cpu' 或 'gpu')
        """
        self.model_type = model_type
        self.min_detection_confidence = min_detection_confidence
        self.device = device.lower()
        self.initialization_error = None # 用于存储初始化过程中的错误信息
        self.frame_timestamp_ms = 0 # 为视频模式增加时间戳
        
        # 通用关键点索引定义
        self.NOSE, self.NECK = 0, 1
        self.RIGHT_SHOULDER, self.RIGHT_ELBOW, self.RIGHT_WRIST = 2, 3, 4
        self.LEFT_SHOULDER, self.LEFT_ELBOW, self.LEFT_WRIST = 5, 6, 7
        self.RIGHT_HIP, self.RIGHT_KNEE, self.RIGHT_ANKLE = 8, 9, 10
        self.LEFT_HIP, self.LEFT_KNEE, self.LEFT_ANKLE = 11, 12, 13
        
        # 为分析器模块提供关键点名称到索引的映射
        self.model_landmarks_info = {
            "NOSE": self.NOSE, "NECK": self.NECK,
            "RIGHT_SHOULDER": self.RIGHT_SHOULDER, "RIGHT_ELBOW": self.RIGHT_ELBOW, "RIGHT_WRIST": self.RIGHT_WRIST,
            "LEFT_SHOULDER": self.LEFT_SHOULDER, "LEFT_ELBOW": self.LEFT_ELBOW, "LEFT_WRIST": self.LEFT_WRIST,
            "RIGHT_HIP": self.RIGHT_HIP, "RIGHT_KNEE": self.RIGHT_KNEE, "RIGHT_ANKLE": self.RIGHT_ANKLE,
            "LEFT_HIP": self.LEFT_HIP, "LEFT_KNEE": self.LEFT_KNEE, "LEFT_ANKLE": self.LEFT_ANKLE,
        }
        
        # 通用骨架连接
        self.pose_pairs = [
            [self.NECK, self.RIGHT_SHOULDER], [self.RIGHT_SHOULDER, self.RIGHT_ELBOW],
            [self.RIGHT_ELBOW, self.RIGHT_WRIST], [self.NECK, self.LEFT_SHOULDER],
            [self.LEFT_SHOULDER, self.LEFT_ELBOW], [self.LEFT_ELBOW, self.LEFT_WRIST],
            [self.NECK, self.RIGHT_HIP], [self.RIGHT_HIP, self.RIGHT_KNEE],
            [self.RIGHT_KNEE, self.RIGHT_ANKLE], [self.NECK, self.LEFT_HIP],
            [self.LEFT_HIP, self.LEFT_KNEE], [self.LEFT_KNEE, self.LEFT_ANKLE],
            [self.NOSE, self.NECK]
        ]

        self._initialize_model()

    def get_landmarks_info(self):
        """
        返回当前模型使用的关键点名称->索引的映射字典。
        这是一个公共接口，供外部模块（如PoseAnalyzer）调用。
        """
        return self.model_landmarks_info

    def _initialize_model(self):
        """根据选择的模型类型初始化模型"""
        if self.model_type.startswith("openpose"):
            self._init_openpose()
        elif self.model_type == self.MODEL_MEDIAPIPE:
            self._init_mediapipe()
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")

        # HOG行人检测器作为备选
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def _init_mediapipe(self):
        """初始化MediaPipe Pose模型，优先使用heavy模型，如果模型不存在则自动下载。"""
        
        # 定义模型，优先使用heavy
        model_options = [
            ("pose_landmarker_heavy.task", "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"),
            ("pose_landmarker_full.task", "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task")
        ]
        
        models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
        os.makedirs(models_dir, exist_ok=True)
        
        model_path = None
        
        for model_name, model_url in model_options:
            current_path = os.path.join(models_dir, model_name)
            if os.path.exists(current_path):
                model_path = current_path
                print(f"找到已存在的模型: {model_name}")
                break
            else:
                print(f"未找到模型 '{model_name}'。正在尝试下载...")
                try:
                    def show_progress(block_num, block_size, total_size):
                        downloaded = block_num * block_size
                        percent = min(100.0, downloaded * 100 / total_size)
                        progress = int(percent / 2)
                        bar = '[' + '=' * progress + ' ' * (50 - progress) + ']'
                        sys.stdout.write(f"\r{bar} {percent:.1f}%")
                        sys.stdout.flush()

                    urllib.request.urlretrieve(model_url, current_path, show_progress)
                    print(f"\n模型 '{model_name}' 下载成功。")
                    model_path = current_path
                    break 
                except Exception as e:
                    print(f"\n自动下载模型 '{model_name}' 失败: {e}")
                    # 如果下载失败，继续尝试下一个模型
                    continue
        
        if not model_path:
            print("错误：所有MediaPipe模型都无法找到或下载。请检查网络连接或手动下载。")
            self.landmarker = None
            return

        # 后续的模型加载逻辑
        try:
            # 读取模型文件到内存缓冲区，以避免非ASCII路径问题
            with open(model_path, 'rb') as f:
                model_buffer = f.read()
            
            # --- 根据设备选择进行初始化 ---
            if self.device == 'gpu':
                print("正在尝试使用 GPU 加速...")
                try:
                    base_options = python.BaseOptions(
                        model_asset_buffer=model_buffer,
                        delegate=python.BaseOptions.Delegate.GPU
                    )
                    options = vision.PoseLandmarkerOptions(
                        base_options=base_options,
                        running_mode=vision.RunningMode.VIDEO,
                        output_segmentation_masks=True,
                        min_pose_detection_confidence=self.min_detection_confidence,
                        min_pose_presence_confidence=self.min_detection_confidence)
                    self.landmarker = vision.PoseLandmarker.create_from_options(options)
                    print("GPU 加速初始化成功！模型将在 GPU 上运行。")
                    self.initialization_error = None
                except Exception as e:
                    self.landmarker = None
                    self.initialization_error = (
                        "GPU模式初始化失败！\n\n"
                        "请检查以下几点：\n"
                        "1. 您的电脑是否拥有支持CUDA的NVIDIA显卡。\n"
                        "2. 是否已正确安装最新的NVIDIA显卡驱动。\n"
                        "3. 是否已安装与您的库版本兼容的CUDA Toolkit和cuDNN。\n\n"
                        f"详细错误信息: {e}"
                    )
                    print(self.initialization_error)
            else:
                print("正在使用 CPU 运行...")
                base_options = python.BaseOptions(model_asset_buffer=model_buffer)
                options = vision.PoseLandmarkerOptions(
                    base_options=base_options,
                    running_mode=vision.RunningMode.VIDEO, 
                    output_segmentation_masks=True,
                    min_pose_detection_confidence=self.min_detection_confidence,
                    min_pose_presence_confidence=self.min_detection_confidence)
                self.landmarker = vision.PoseLandmarker.create_from_options(options)
                print("模型已在 CPU 上成功初始化。")
                self.initialization_error = None

        except Exception as e:
            self.landmarker = None
            self.initialization_error = f"读取模型文件失败: {e}"
            print(self.initialization_error)

    def _init_openpose(self):
        """初始化OpenPose DNN模型"""
        model_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
        os.makedirs(model_folder, exist_ok=True)

        model_files = {
            self.MODEL_OPENPOSE_COCO: ("pose_deploy_linevec.prototxt", "pose_iter_440000.caffemodel", 18),
            self.MODEL_OPENPOSE_BODY_25: ("pose_deploy.prototxt", "pose_iter_584000.caffemodel", 25)
        }
        
        proto, weights, n_points = model_files.get(self.model_type)
        self.proto_file = os.path.join(model_folder, proto)
        self.weights_file = os.path.join(model_folder, weights)
        self.n_points = n_points

        self.use_openpose = os.path.exists(self.proto_file) and os.path.exists(self.weights_file)
        
        if self.use_openpose:
            self.net = cv2.dnn.readNetFromCaffe(self.proto_file, self.weights_file)
            if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            
            self.in_width, self.in_height, self.scale = 368, 368, 1.0 / 255
            print(f"{self.model_type} 模型初始化成功。")
        else:
            print(f"警告：{self.model_type} 模型文件不存在，将使用备用检测方法。")
            print(f"请下载模型文件 '{proto}' 和 '{weights}' 并放置在 '{model_folder}' 目录下。")

    def detect_pose(self, image, timestamp_ms: int = None):
        """
        检测图像中的人体姿势, 返回处理后的图像和关键点数据
        
        Args:
            image: 输入图像
            timestamp_ms: (可选) 视频帧的时间戳 (毫秒)，用于MediaPipe视频模式以提高跟踪稳定性
        """
        landmarks = {}
        if self.model_type == self.MODEL_MEDIAPIPE:
            if self.landmarker:
                # 视频模式需要一个单调递增的时间戳
                # 如果外部提供了精确的时间戳，则使用它，否则使用内部计数器
                if timestamp_ms is None:
                    self.frame_timestamp_ms += 33  # 假设约30FPS的帧率
                    current_timestamp = self.frame_timestamp_ms
                else:
                    current_timestamp = timestamp_ms
                
                landmarks = self._detect_pose_mediapipe(image, current_timestamp)
        elif self.model_type.startswith("openpose"):
            if hasattr(self, 'use_openpose') and self.use_openpose:
                landmarks = self._detect_pose_openpose(image)
        
        # 如果主模型检测效果不佳, 尝试HOG后备方案
        # if len(landmarks) < 4:
        #     fallback_landmarks = self._detect_pose_fallback(image)
        #     # 只用后备方案补充未检测到的点
        #     for k, v in fallback_landmarks.items():
        #         if k not in landmarks:
        #             landmarks[k] = v
        
        # 在图像上绘制所有检测结果
        processed_image = self.draw_pose(image.copy(), landmarks)

        return processed_image, landmarks

    def _detect_pose_mediapipe(self, image, timestamp_ms):
        """使用MediaPipe检测姿势 (Tasks API - 视频模式)"""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        
        try:
            detection_result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        except Exception as e:
            print(f"MediaPipe 检测出错: {e}")
            return {}

        landmarks = {}
        # Assuming one person in the image for simplicity
        if detection_result.pose_landmarks:
            pose_landmarks_list = detection_result.pose_landmarks[0]
            h, w, _ = image.shape
            
            # Map mediapipe landmarks to our internal format
            # The new 'tasks' API uses integer indices for landmarks, not enums.
            mp_index_to_our_map = {
                0: self.NOSE,
                12: self.RIGHT_SHOULDER,
                14: self.RIGHT_ELBOW,
                16: self.RIGHT_WRIST,
                11: self.LEFT_SHOULDER,
                13: self.LEFT_ELBOW,
                15: self.LEFT_WRIST,
                24: self.RIGHT_HIP,
                26: self.RIGHT_KNEE,
                28: self.RIGHT_ANKLE,
                23: self.LEFT_HIP,
                25: self.LEFT_KNEE,
                27: self.LEFT_ANKLE,
            }

            for mp_idx, our_idx in mp_index_to_our_map.items():
                if mp_idx < len(pose_landmarks_list):
                    landmark = pose_landmarks_list[mp_idx]
                    # The new API provides landmark.visibility and landmark.presence
                    # We can use visibility as confidence
                    if landmark.visibility > self.min_detection_confidence:
                        landmarks[our_idx] = {'x': int(landmark.x * w), 'y': int(landmark.y * h), 'confidence': landmark.visibility}

            # Estimate neck position
            if self.RIGHT_SHOULDER in landmarks and self.LEFT_SHOULDER in landmarks:
                 landmarks[self.NECK] = {
                    'x': (landmarks[self.RIGHT_SHOULDER]['x'] + landmarks[self.LEFT_SHOULDER]['x']) // 2,
                    'y': (landmarks[self.RIGHT_SHOULDER]['y'] + landmarks[self.LEFT_SHOULDER]['y']) // 2,
                    'confidence': (landmarks[self.RIGHT_SHOULDER]['confidence'] + landmarks[self.LEFT_SHOULDER]['confidence']) / 2
                }
        return landmarks

    def _detect_pose_openpose(self, image):
        """使用OpenPose检测姿势"""
        h, w = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, self.scale, (self.in_width, self.in_height), (0, 0, 0), swapRB=False, crop=False)
        self.net.setInput(blob)
        output = self.net.forward()
        
        detected_keypoints = []
        for i in range(self.n_points):
            prob_map = output[0, i, :, :]
            _, conf, _, point = cv2.minMaxLoc(prob_map)
            if conf > self.min_detection_confidence:
                x, y = int(point[0] * w / output.shape[3]), int(point[1] * h / output.shape[2])
                detected_keypoints.append({'point': (x, y), 'confidence': conf, 'id': i})
            else:
                detected_keypoints.append(None)
        
        openpose_map = {
            self.MODEL_OPENPOSE_COCO: {0:self.NOSE, 1:self.NECK, 2:self.RIGHT_SHOULDER, 3:self.RIGHT_ELBOW, 4:self.RIGHT_WRIST, 5:self.LEFT_SHOULDER, 6:self.LEFT_ELBOW, 7:self.LEFT_WRIST, 8:self.RIGHT_HIP, 9:self.RIGHT_KNEE, 10:self.RIGHT_ANKLE, 11:self.LEFT_HIP, 12:self.LEFT_KNEE, 13:self.LEFT_ANKLE},
            self.MODEL_OPENPOSE_BODY_25: {0:self.NOSE, 1:self.NECK, 2:self.RIGHT_SHOULDER, 3:self.RIGHT_ELBOW, 4:self.RIGHT_WRIST, 5:self.LEFT_SHOULDER, 6:self.LEFT_ELBOW, 7:self.LEFT_WRIST, 9:self.RIGHT_HIP, 10:self.RIGHT_KNEE, 11:self.RIGHT_ANKLE, 12:self.LEFT_HIP, 13:self.LEFT_KNEE, 14:self.LEFT_ANKLE}
        }[self.model_type]

        landmarks = {}
        for data in detected_keypoints:
            if data and data['id'] in openpose_map:
                our_idx = openpose_map[data['id']]
                landmarks[our_idx] = {'x': data['point'][0], 'y': data['point'][1], 'confidence': data['confidence']}
        return landmarks
        
    def _detect_pose_fallback(self, image):
        """使用HOG检测器作为备用方案, 仅返回估算的关键点"""
        h, w = image.shape[:2]
        scale = min(400 / w, 400 / h) if w > 0 and h > 0 else 1.0
        small_img = cv2.resize(image, (0, 0), fx=scale, fy=scale)
        
        rects, _ = self.hog.detectMultiScale(small_img, winStride=(4, 4), padding=(8, 8), scale=1.05)
        
        landmarks = {}
        if len(rects) > 0:
            x, y, w_rect, h_rect = [int(c / scale) for c in rects[0]]
            
            landmarks = {
                'box': (x, y, w_rect, h_rect),
                self.NOSE: {'x': x + w_rect // 2, 'y': y + h_rect // 6, 'confidence': 0.6},
                self.NECK: {'x': x + w_rect // 2, 'y': y + h_rect // 5, 'confidence': 0.6},
                self.RIGHT_SHOULDER: {'x': x + w_rect // 3, 'y': y + h_rect // 4, 'confidence': 0.6},
                self.LEFT_SHOULDER: {'x': x + w_rect * 2 // 3, 'y': y + h_rect // 4, 'confidence': 0.6},
                self.RIGHT_ELBOW: {'x': x + w_rect // 4, 'y': y + h_rect * 2 // 5, 'confidence': 0.5},
                self.LEFT_ELBOW: {'x': x + w_rect * 3 // 4, 'y': y + h_rect * 2 // 5, 'confidence': 0.5},
                self.RIGHT_WRIST: {'x': x + w_rect // 5, 'y': y + h_rect // 2, 'confidence': 0.4},
                self.LEFT_WRIST: {'x': x + w_rect * 4 // 5, 'y': y + h_rect // 2, 'confidence': 0.4},
                self.RIGHT_HIP: {'x': x + w_rect // 3, 'y': y + h_rect * 3 // 5, 'confidence': 0.5},
                self.LEFT_HIP: {'x': x + w_rect * 2 // 3, 'y': y + h_rect * 3 // 5, 'confidence': 0.5},
                self.RIGHT_KNEE: {'x': x + w_rect // 3, 'y': y + h_rect * 4 // 5, 'confidence': 0.4},
                self.LEFT_KNEE: {'x': x + w_rect * 2 // 3, 'y': y + h_rect * 4 // 5, 'confidence': 0.4},
                self.RIGHT_ANKLE: {'x': x + w_rect // 3, 'y': y + h_rect - 10, 'confidence': 0.3},
                self.LEFT_ANKLE: {'x': x + w_rect * 2 // 3, 'y': y + h_rect - 10, 'confidence': 0.3},
            }
        return landmarks

    def draw_pose(self, image, landmarks):
        """在图像上绘制姿势关键点和骨架"""
        colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0)]
                  
        if not landmarks:
            cv2.putText(image, "未检测到人体", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return image

        if 'box' in landmarks:
            x, y, w_rect, h_rect = landmarks['box']
            cv2.rectangle(image, (x, y), (x + w_rect, y + h_rect), (255, 0, 0), 2)

        for idx, data in landmarks.items():
            if idx == 'box': continue
            cv2.circle(image, (data['x'], data['y']), 5, (0, 255, 255), -1, cv2.LINE_AA)
        
        for i, pair in enumerate(self.pose_pairs):
            if pair[0] in landmarks and pair[1] in landmarks:
                pt1 = (landmarks[pair[0]]['x'], landmarks[pair[0]]['y'])
                pt2 = (landmarks[pair[1]]['x'], landmarks[pair[1]]['y'])
                cv2.line(image, pt1, pt2, colors[i % len(colors)], 2, cv2.LINE_AA)

        # 计算并显示关键角度
        angles = {
            "R Elbow": (self.RIGHT_SHOULDER, self.RIGHT_ELBOW, self.RIGHT_WRIST),
            "R Shoulder": (self.RIGHT_HIP, self.RIGHT_SHOULDER, self.RIGHT_ELBOW),
            "R Knee": (self.RIGHT_HIP, self.RIGHT_KNEE, self.RIGHT_ANKLE),
        }
        for i, (name, points) in enumerate(angles.items()):
            angle = self.get_angle(landmarks, *points)
            if angle is not None:
                cv2.putText(image, f"{name}: {angle:.1f}", (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return image

    def get_angle(self, landmarks, p1_idx, p2_idx, p3_idx):
        """计算三个点之间的角度"""
        if not all(p in landmarks for p in [p1_idx, p2_idx, p3_idx]):
            return None
            
        a = np.array([landmarks[p1_idx]['x'], landmarks[p1_idx]['y']])
        b = np.array([landmarks[p2_idx]['x'], landmarks[p2_idx]['y']])
        c = np.array([landmarks[p3_idx]['x'], landmarks[p3_idx]['y']])
        
        ba, bc = a - b, c - b
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
        return angle

    def get_distance(self, landmarks, p1_idx, p2_idx):
        """计算两点之间的距离"""
        if not all(p in landmarks for p in [p1_idx, p2_idx]):
            return None
        
        p1 = np.array([landmarks[p1_idx]['x'], landmarks[p1_idx]['y']])
        p2 = np.array([landmarks[p2_idx]['x'], landmarks[p2_idx]['y']])
        return np.linalg.norm(p1 - p2) 