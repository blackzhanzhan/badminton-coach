import math
import numpy as np
from enum import Enum

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

    def _analyze_receive_sequence(self, landmarks):
        """
        处理完整的接球动作序列的状态机。
        """
        # 类似逻辑，调整为接球阶段
        if self.action_state == self.ActionState.IDLE:
            # 假设当用户摆出准备姿势时，就进入PREPARING状态
            # (这里的判断条件可以很复杂，我们先用一个简单的)
            is_ready = self._is_in_ready_stance(landmarks)
            if is_ready:
                self.action_state = self.ActionState.PREPARING
                # 记录进入准备状态的关键帧数据
                self.keyframes['preparation'] = self._capture_keyframe_data(landmarks)
                print(f"状态转换: {self.ActionState.IDLE.value} -> {self.ActionState.PREPARING.value}")
                print(f"已捕获 'preparation' 关键帧数据: {self.keyframes['preparation']}")
        
        elif self.action_state == self.ActionState.PREPARING:
            # 在这里添加从 PREPARING 到 BACKSWING 的逻辑
            # 例如：检测到手臂开始向后摆动
            pass

        # ... 后续将实现其他状态的转换逻辑 ...

        # 在界面上显示当前状态，用于调试
        self.feedback.append(f"当前动作状态: {self.action_state.value}")
        if self.keyframes:
            self.feedback.append("已捕获的关键帧: " + ", ".join(self.keyframes.keys()))


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
        return {
            'left_knee_angle': self._calculate_angle(
                self._get_landmark(landmarks, "LEFT_HIP"),
                self._get_landmark(landmarks, "LEFT_KNEE"),
                self._get_landmark(landmarks, "LEFT_ANKLE")
            ),
            'right_elbow_angle': self._calculate_angle(
                self._get_landmark(landmarks, "RIGHT_SHOULDER"),
                self._get_landmark(landmarks, "RIGHT_ELBOW"),
                self._get_landmark(landmarks, "RIGHT_WRIST")
            )
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