"""
"""
羽毛球接球标准动作数据
包含各个关键阶段的标准姿势参数
"""

# 标准接球动作参数
STANDARD_RECEIVE_PARAMS = {
    'preparation': {
        'description': '准备阶段：膝盖弯曲，重心低，拍子在前方',
        'right_elbow_angle': (90, 120),  # 右肘角度范围
        'right_shoulder_angle': (30, 50),  # 右肩角度范围
        'right_knee_angle': (130, 150),  # 右膝角度范围
        'body_lean_angle': (70, 90),  # 身体前倾角度范围
    },
    'approach': {
        'description': '接近阶段：快速移动到球的位置，保持平衡',
        'right_elbow_angle': (100, 130),  # 右肘角度范围
        'right_shoulder_angle': (50, 80),  # 右肩角度范围
        'right_knee_angle': (140, 160),  # 右膝角度范围
        'body_rotation': (0, 20),  # 身体旋转角度范围
    },
    'hit': {
        'description': '击球阶段：挥拍击球，手臂伸展',
        'right_elbow_angle': (150, 180),  # 右肘角度范围
        'right_shoulder_angle': (90, 120),  # 右肩角度范围
        'right_knee_angle': (140, 160),  # 右膝角度范围
        'wrist_angle': (160, 180),  # 手腕角度范围
    },
    'recovery': {
        'description': '恢复阶段：快速回到准备姿势',
        'right_elbow_angle': (90, 120),  # 右肘角度范围
        'right_shoulder_angle': (30, 50),  # 右肩角度范围
        'right_knee_angle': (130, 150),  # 右膝角度范围
        'body_rotation': (0, 10),  # 身体旋转角度范围
    }
}

# 各阶段的关键检查点
RECEIVE_CHECKPOINTS = {
    'preparation': [
        '握拍正确，重心降低',
        '膝盖适当弯曲',
        '拍子置于前方',
        '双眼注视来球',
        '身体平衡',
    ],
    'approach': [
        '脚步移动迅速',
        '保持低重心',
        '手臂准备挥拍',
        '目光锁定球',
        '平衡不失',
    ],
    'hit': [
        '挥拍时机准确',
        '击球点合适',
        '发力均匀',
        '身体协调',
        '眼神跟随',
    ],
    'recovery': [
        '快速回位',
        '恢复准备姿势',
        '保持警惕',
        '呼吸平稳',
        '准备下一球',
    ]
}

def get_standard_receive_params():
    """获取标准接球动作参数"""
    return STANDARD_RECEIVE_PARAMS

def get_receive_checkpoints():
    """获取接球动作检查点"""
    return RECEIVE_CHECKPOINTS 