import math
from enum import Enum

class PoseAnalyzer:
    """
    此类负责分析输入的姿态关键点，并生成可读的反馈建议。
    它可以处理静态姿态（如准备姿势）和动态动作序列（如发球）。
    """
    class AnalysisMode(Enum):
        """定义分析模式的枚举"""
        STATIC_READY_STANCE = "静态-准备姿势"
        DYNAMIC_SERVE = "动态-发球分析"

    class ActionState(Enum):
        """定义动态动作分析的状态机"""
        IDLE = "空闲"
        PREPARING = "准备中"
        BACKSWING = "后摆"
        HITTING = "击球"
        FOLLOW_THROUGH = "随挥"
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
        if self.analysis_mode == self.AnalysisMode.DYNAMIC_SERVE:
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
        elif self.analysis_mode == self.AnalysisMode.DYNAMIC_SERVE:
            self._analyze_serve_sequence(landmarks)
        else:
            self.feedback.append(f"错误：未知的分析模式 '{self.analysis_mode}'")
        
        return self.feedback.copy()

    def _analyze_serve_sequence(self, landmarks):
        """
        处理完整的发球动作序列的状态机。
        """
        # 在这里，我们将实现状态转换的逻辑
        # 目前，我们先实现一个简单的状态转换来验证框架
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
        avg_hip_y = (self._get_landmark(landmarks, "LEFT_HIP")[1] + self._get_landmark(landmarks, "RIGHT_HIP")[1]) / 2
        avg_shoulder_y = (self._get_landmark(landmarks, "LEFT_SHOULDER")[1] + self._get_landmark(landmarks, "RIGHT_SHOULDER")[1]) / 2
        
        # 简单地通过y坐标差异判断，之后可以换成更精确的角度计算
        body_lean = avg_hip_y - avg_shoulder_y 
        if body_lean > 0.1: # 这个阈值需要调整
             self.feedback.append("状态: 身体前倾姿势良好。")
        else:
             self.feedback.append("提示: 身体不够前倾，请收腹，上半身稍微前倾。")
             
    def _get_landmark(self, landmarks, name):
        """通过名称获取关键点坐标"""
        try:
            index = self.landmarks_info[name]
            return landmarks[index]
        except (KeyError, IndexError):
            # print(f"警告: 无法找到名为 '{name}' 的关键点。")
            return None

    def _calculate_angle(self, p1, p2, p3):
        """计算由三个点p1, p2, p3组成的角度，p2为顶点"""
        if not all([p1, p2, p3]):
            return None
        
        # 可见性检查
        if p1[2] < 0.5 or p2[2] < 0.5 or p3[2] < 0.5:
            return None

        try:
            a = math.sqrt((p2[0] - p3[0])**2 + (p2[1] - p3[1])**2)
            b = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
            c = math.sqrt((p1[0] - p3[0])**2 + (p1[1] - p3[1])**2)
            
            angle_rad = math.acos((a**2 + b**2 - c**2) / (2 * a * b))
            angle_deg = math.degrees(angle_rad)
            
            return angle_deg
        except (ValueError, ZeroDivisionError):
            return None 