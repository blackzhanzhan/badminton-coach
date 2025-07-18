import math
import numpy as np
from enum import Enum
from modules.standard_poses import STANDARD_RECEIVE_PARAMS
import json
import math
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

class PoseAnalyzer:
    """
    此类负责分析输入的姿态关键点，并生成可读的反馈建议。
    它可以处理静态姿态（如准备姿势）和动态动作序列（如发球）。
    """
    class AnalysisMode(Enum):
        """定义分析模式的枚举"""
        STATIC_READY_STANCE = "静态-准备姿势"
        DYNAMIC_RECEIVE = "动态-接球分析"

    class ActionState(Enum):
        """定义动态动作分析的状态机"""
        IDLE = "空闲"
        PREPARING = "准备中"
        APPROACH = "接近"
        HITTING = "击球"
        RECOVERY = "恢复"
        ANALYZING = "分析中"

    def __init__(self, landmarks_info, analysis_mode=AnalysisMode.STATIC_READY_STANCE):
        """
        初始化姿态分析器。

        Args:
            landmarks_info (dict): 包含了关键点名称到索引的映射。
            analysis_mode (AnalysisMode): 要执行的分析模式。
        """
        self.landmarks_info = landmarks_info
        self.analysis_mode = analysis_mode
        self.feedback = []

        # 仅在动态分析模式下初始化状态机相关变量
        if self.analysis_mode == self.AnalysisMode.DYNAMIC_RECEIVE:
            self.action_state = self.ActionState.IDLE
            self.keyframes = {}
            self.prev_landmarks = None  # 存储上一帧 landmarks
            self.state_counter = 0  # 状态计数器
            self.min_frames_to_confirm = 3  # 最小确认帧数
            print(f"进入 '{self.analysis_mode.value}' 模式, 初始状态: {self.action_state.value}")

    def analyze_pose(self, landmarks):
        """
        分析给定的姿态关键点。
        这是外部调用的主入口。

        Args:
            landmarks (list): 包含所有关键点的列表。

        Returns:
            list: 包含分析建议的字符串列表。
        """
        if not landmarks:
            return ["画面中未检测到目标。"]

        self.feedback.clear()

        # 根据不同的分析模式，调用不同的处理逻辑
        if self.analysis_mode == self.AnalysisMode.STATIC_READY_STANCE:
            self._analyze_ready_stance(landmarks)
        elif self.analysis_mode == self.AnalysisMode.DYNAMIC_RECEIVE:
            self._analyze_receive_sequence(landmarks)
        else:
            self.feedback.append(f"错误：未知的分析模式 '{self.analysis_mode}'")
        
        return self.feedback.copy()

    def analyze_json_difference(self, standard_json_path, learner_json_path, api_key=None):
        """比较JSON并用大模型生成建议（带DTW对齐）"""
        try:
            with open(standard_json_path, 'r') as f:
                standard_data = json.load(f)
            with open(learner_json_path, 'r') as f:
                learner_data = json.load(f)
        except Exception as e:
            return [f"加载JSON失败: {e}"]

        if not standard_data or not learner_data:
            return ["JSON数据为空，无法比较。"]

        # 提取特征序列（示例：右肘角度 + 右肩角度，作为多维序列）
        def extract_angle_sequence(data):
            sequence = []
            for frame in data:
                landmarks = frame.get('landmarks', {})
                elbow_angle = self._calculate_angle(
                    self._get_landmark(landmarks, "RIGHT_SHOULDER"),
                    self._get_landmark(landmarks, "RIGHT_ELBOW"),
                    self._get_landmark(landmarks, "RIGHT_WRIST")
                ) or 0
                shoulder_angle = self._calculate_angle(
                    self._get_landmark(landmarks, "RIGHT_HIP"),
                    self._get_landmark(landmarks, "RIGHT_SHOULDER"),
                    self._get_landmark(landmarks, "RIGHT_ELBOW")
                ) or 0
                sequence.append([elbow_angle, shoulder_angle])  # 多维特征
            return np.array(sequence)

        std_seq = extract_angle_sequence(standard_data)
        learn_seq = extract_angle_sequence(learner_data)

        if len(std_seq) == 0 or len(learn_seq) == 0:
            return ["无法提取有效角度序列。"]

        # 应用DTW计算对齐距离和路径
        distance, path = fastdtw(std_seq, learn_seq, dist=euclidean)

        # 计算对齐后的平均偏差
        aligned_diffs = []
        for i, j in path:
            diff = np.linalg.norm(std_seq[i] - learn_seq[j])  # 欧氏距离
            aligned_diffs.append(diff)
        avg_aligned_diff = np.mean(aligned_diffs) if aligned_diffs else 0

        # 节奏差异：路径长度 vs. 序列长度
        duration_ratio = len(learn_seq) / len(std_seq) if len(std_seq) > 0 else 1
        rhythm_suggestion = f"动作节奏{'慢' if duration_ratio > 1.2 else '快'}了 {abs(duration_ratio - 1) * 100:.1f}%"

        # 用大模型生成建议
        prompt = f"作为羽毛球教练，基于以下接球动作差异给出纠正建议：平均对齐偏差 {avg_aligned_diff:.1f}°，{rhythm_suggestion}。重点关注挥拍和恢复阶段。"
        suggestions = ["默认建议: 动作相似，但节奏稍慢，建议加速准备阶段。"]

        if api_key:  # Groq API示例
            try:
                from groq import Groq
                client = Groq(api_key=api_key)
                response = client.chat.completions.create(
                    model="mixtral-8x7b-32768",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7
                )
                suggestions = [response.choices[0].message.content]
            except Exception as e:
                suggestions.append(f"API调用失败: {e}")

        return suggestions

    def _analyze_receive_sequence(self, landmarks):
        # 原状态机逻辑已注释掉，转向离线JSON分析
        self.feedback.append("动态模式：使用离线JSON比较进行分析。")
        return

    def _is_in_ready_stance(self, landmarks):
        """
        一个简化的判断是否处于准备姿势的函数。
        这里可以重用或修改 _analyze_ready_stance 的逻辑。
        返回 True 如果姿势大致正确，否则 False。
        """
        # 简化版检查：只要左右膝盖都处于弯曲状态就认为准备好了
        left_knee_angle = self._calculate_angle(
            self._get_landmark(landmarks, "LEFT_HIP"),
            self._get_landmark(landmarks, "LEFT_KNEE"),
            self._get_landmark(landmarks, "LEFT_ANKLE")
        )
        right_knee_angle = self._calculate_angle(
            self._get_landmark(landmarks, "RIGHT_HIP"),
            self._get_landmark(landmarks, "RIGHT_KNEE"),
            self._get_landmark(landmarks, "RIGHT_ANKLE")
        )
        if left_knee_angle is not None and right_knee_angle is not None:
            if 100 < left_knee_angle < 170 and 100 < right_knee_angle < 170:
                return True
        return False

    def _capture_keyframe_data(self, landmarks):
        """
        捕获并返回当前帧的所有相关角度数据。
        """
        # 在真实场景中，这里会计算所有需要的角度
        # 现在我们只返回一个示例数据
        left_hip = self._get_landmark(landmarks, "LEFT_HIP")
        right_hip = self._get_landmark(landmarks, "RIGHT_HIP")
        left_shoulder = self._get_landmark(landmarks, "LEFT_SHOULDER")
        right_shoulder = self._get_landmark(landmarks, "RIGHT_SHOULDER")
        body_lean_angle = None
        if all([left_hip, right_hip, left_shoulder, right_shoulder]):
            if left_hip is not None and right_hip is not None and left_shoulder is not None and right_shoulder is not None:
                avg_hip = ((left_hip['x'] + right_hip['x']) / 2, (left_hip['y'] + right_hip['y']) / 2)
                avg_shoulder = ((left_shoulder['x'] + right_shoulder['x']) / 2, (left_shoulder['y'] + right_shoulder['y']) / 2)
                # 计算肩部到髋部的向量与垂直线的角度
                delta_x = avg_hip[0] - avg_shoulder[0]
                delta_y = avg_hip[1] - avg_shoulder[1]
                body_lean_angle = math.degrees(math.atan2(delta_x, delta_y))
                if body_lean_angle < 0:
                    body_lean_angle += 180  # 调整为正值
        return {
            'left_knee_angle': self._calculate_angle(
                self._get_landmark(landmarks, "LEFT_HIP"),
                self._get_landmark(landmarks, "LEFT_KNEE"),
                self._get_landmark(landmarks, "LEFT_ANKLE")
            ),
            'right_knee_angle': self._calculate_angle(
                self._get_landmark(landmarks, "RIGHT_HIP"),
                self._get_landmark(landmarks, "RIGHT_KNEE"),
                self._get_landmark(landmarks, "RIGHT_ANKLE")
            ),
            'right_elbow_angle': self._calculate_angle(
                self._get_landmark(landmarks, "RIGHT_SHOULDER"),
                self._get_landmark(landmarks, "RIGHT_ELBOW"),
                self._get_landmark(landmarks, "RIGHT_WRIST")
            ),
            'right_shoulder_angle': self._calculate_angle(
                self._get_landmark(landmarks, "RIGHT_HIP"),
                self._get_landmark(landmarks, "RIGHT_SHOULDER"),
                self._get_landmark(landmarks, "RIGHT_ELBOW")
            ),
            'body_lean_angle': body_lean_angle
        }

    def _analyze_ready_stance(self, landmarks):
        """
        分析静态的羽毛球准备姿势。
        (这是我们之前的旧逻辑)
        """
        # ... (这里是之前的所有静态分析代码，保持不变) ...
        # Golden angles for ready stance (example values)
        # ...
        # (为了简洁，省略了之前静态分析的完整代码)
        self.feedback.append("正在执行静态准备姿势分析...")
        # 膝盖角度
        left_knee_angle = self._calculate_angle(
            self._get_landmark(landmarks, "LEFT_HIP"),
            self._get_landmark(landmarks, "LEFT_KNEE"),
            self._get_landmark(landmarks, "LEFT_ANKLE")
        )
        right_knee_angle = self._calculate_angle(
            self._get_landmark(landmarks, "RIGHT_HIP"),
            self._get_landmark(landmarks, "RIGHT_KNEE"),
            self._get_landmark(landmarks, "RIGHT_ANKLE")
        )

        knee_golden_angle = (110, 140)
        
        if left_knee_angle and right_knee_angle:
            avg_knee_angle = (left_knee_angle + right_knee_angle) / 2
            if avg_knee_angle < knee_golden_angle[0]:
                self.feedback.append("提示: 膝盖弯曲过多，重心可能过低。")
            elif avg_knee_angle > knee_golden_angle[1]:
                self.feedback.append("提示: 膝盖弯曲不足，请再降低重心。")
            else:
                self.feedback.append("状态: 膝盖角度良好。")
        else:
            self.feedback.append("警告: 无法清晰识别膝盖角度。")

        # 身体前倾角度
        left_hip = self._get_landmark(landmarks, "LEFT_HIP")
        right_hip = self._get_landmark(landmarks, "RIGHT_HIP")
        left_shoulder = self._get_landmark(landmarks, "LEFT_SHOULDER")
        right_shoulder = self._get_landmark(landmarks, "RIGHT_SHOULDER")

        if all([left_hip, right_hip, left_shoulder, right_shoulder]):
            avg_hip_y = (left_hip['y'] + right_hip['y']) / 2
            avg_shoulder_y = (left_shoulder['y'] + right_shoulder['y']) / 2
            
            # 简单地通过y坐标差异判断，使用像素距离作为阈值
            body_lean_diff = avg_hip_y - avg_shoulder_y  # 图像坐标系中y越大越靠下
            if body_lean_diff > 20: # 阈值可能需要根据实际效果微调
                 self.feedback.append("状态: 身体前倾姿势良好。")
            else:
                 self.feedback.append("提示: 身体不够前倾，请收腹，上半身稍微前倾。")
        else:
            self.feedback.append("警告: 无法清晰识别肩部或髋部。")
             
    def _get_landmark(self, landmarks, name):
        """通过名称获取关键点坐标"""
        try:
            index = self.landmarks_info[name]
            # 使用 .get() 避免当landmarks中不存在某个索引时引发KeyError
            return landmarks.get(index)
        except KeyError:
            # 当 landmarks_info 中没有这个名字时
            # print(f"警告: 无法在landmarks_info中找到名为 '{name}' 的关键点定义。")
            return None

    def _calculate_angle(self, p1, p2, p3):
        """
        计算由三个点p1, p2, p3组成的角度，p2为顶点。
        点以 {'x': x, 'y': y, 'confidence': c} 的字典形式提供。
        """
        if not all([p1, p2, p3]):
            return None
        
        # 使用置信度进行检查，可以适当降低分析时的阈值
        if p1['confidence'] < 0.3 or p2['confidence'] < 0.3 or p3['confidence'] < 0.3:
            return None

        try:
            # 使用 numpy 进行矢量计算，更高效稳定
            p1_np = np.array([p1['x'], p1['y']])
            p2_np = np.array([p2['x'], p2['y']])
            p3_np = np.array([p3['x'], p3['y']])

            v1 = p1_np - p2_np
            v2 = p3_np - p2_np

            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)

            # 避免除以零
            if norm_v1 == 0 or norm_v2 == 0:
                return None

            cosine_angle = np.dot(v1, v2) / (norm_v1 * norm_v2)
            
            # 夹逼余弦值到[-1, 1]，防止浮点误差导致 acos 失败
            angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
            
            return angle
        except (ValueError, ZeroDivisionError, KeyError):
             # 增加KeyError以防字典缺少'x', 'y', 'confidence'
            return None 