# TrueEmotion Pro v4.0

新一代中文情感AI系统 - 混合神经网络 + 规则系统 + **有血有肉的情感表达**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 核心特性

| 特性 | 说明 |
|------|------|
| **24种情感类别** | 基于Plutchik情感轮，8种主要+16种扩展情感 |
| **VAD三维模型** | Valence-Arousal-Dominance 连续情感坐标 |
| **共情回复生成** | 口语化、有血有肉、去AI化的情感化回复 |
| **长期记忆** | 跨会话用户画像、交互历史持久化 (Repository模式) |
| **持续进化** | 从对话中学习新模式，自动形成检测准则 |
| **零依赖** | 纯规则系统，无需PyTorch/Transformers |

## 架构

```
src/trueemotion/
├── api/                    # 统一API层
│   ├── routes.py          # TrueEmotionPro 主类
│   └── schemas.py         # 数据模型定义
├── core/                  # 核心领域
│   ├── emotions/
│   │   ├── plutchik24.py # 24色情感轮定义
│   │   └── detector.py   # 规则检测器
│   ├── analysis/
│   │   ├── analyzer.py   # 分析器门面
│   │   └── output.py     # 输出数据结构
│   └── response/
│       └── engine.py      # 共情回复引擎
├── memory/                # 记忆层
│   └── repository.py     # Repository模式实现
└── learning/             # 进化层
    └── evolution.py      # 进化管理器
```

## 安装

```bash
# 克隆仓库
git clone https://github.com/qaz154/trueemotion-pro.git
cd trueemotion-pro

# 添加到Python路径
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

## 快速开始

```python
import sys
sys.path.insert(0, 'src')

from trueemotion import TrueEmotionPro

# 初始化
pro = TrueEmotionPro()

# 分析文本
result = pro.analyze("今天太开心了！终于完成了项目！")

print(result.emotion.primary)       # joy
print(result.emotion.intensity)     # 0.85
print(result.emotion.vad)          # (0.8, 0.5, 0.7)
print(result.human_response.text)   # "太为你高兴了！说说怎么回事！"
```

## 情感类别

### 主要情感 (8种)

| 情感 | VAD坐标 | 说明 |
|------|---------|------|
| joy | (0.8, 0.5, 0.7) | 开心、高兴、快乐 |
| sadness | (-0.8, -0.3, -0.5) | 难过、伤心、失落 |
| anger | (-0.8, 0.7, 0.5) | 生气、愤怒 |
| fear | (-0.6, 0.6, -0.4) | 害怕、担心、紧张 |
| disgust | (-0.7, -0.1, -0.4) | 恶心、讨厌 |
| surprise | (0.3, 0.8, 0.3) | 惊讶、意外 |
| trust | (0.6, 0.3, 0.7) | 相信、依赖 |
| anticipation | (0.5, 0.6, 0.4) | 期待、希望 |

### 扩展情感 (16种)

- **强化**: ecstasy, grief, rage, terror, loathing, astonishment, admiration, vigilance
- **弱化**: serenity, penitence, annoyance, apprehension, boredom, distraction, acceptance, interest
- **复合**: love, guilt, envy, contempt, optimism, despair, desire, pride
- **其他**: anxiety, remorse

## API文档

### TrueEmotionPro.analyze()

```python
result = pro.analyze(
    text="工作好累啊，老加班",
    context=None,           # 对话上下文
    learn=True,            # 是否学习
    response="确实累...",   # AI回复
    feedback=0.8,          # 用户反馈 0-1
    user_id="user123"     # 用户ID
)
```

**返回 AnalysisResult:**

```python
{
    "version": "4.0.0",
    "engine": "rule-based + empathy",
    "emotion": {
        "primary": "sadness",
        "intensity": 0.6,
        "vad": (-0.8, -0.3, -0.5),
        "confidence": 0.85,
        "all_emotions": {"sadness": 0.6, "anxiety": 0.3}
    },
    "human_response": {
        "text": "先缓缓，我陪着你",
        "empathy_type": "comfort",
        "intensity_level": "moderate",
        "follow_up": "发生什么了？"
    },
    "user_profile": {
        "user_id": "user123",
        "total_interactions": 5,
        "dominant_emotion": "sadness",
        "relationship_level": 0.5,
        "learned_patterns": 2
    }
}
```

### 其他方法

```python
# 获取用户画像
pro.get_user_profile(user_id="user123")

# 获取记忆状态
pro.get_memory_status()
# {'total_users': 1, 'total_patterns': 5, 'memory_path': './memory'}

# 执行进化
pro.evolve()
# {'total_patterns_analyzed': 5, 'evolved_rules': [...]}

# 获取系统统计
pro.get_stats()
```

## 共情回复示例

| 用户输入 | TrueEmotion Pro 回复 |
|----------|---------------------|
| "工作好累啊" | "说实话确实累...换我我也烦成这样" |
| "我升职了！" | "太为你高兴了！说说怎么回事！" |
| "气死了！" | "确实气人！换我我也急！" |
| "被裁员了..." | "先缓缓，我陪着你" |

## 持续进化

系统支持从对话中持续学习和进化：

```python
# 学习新模式
pro.analyze(
    "工作好累啊",
    learn=True,
    response="确实累，注意身体",
    feedback=0.8,
    user_id="user1"
)

# 执行进化 - 分析模式，提取高反馈规则
result = pro.evolve()
print(result['evolved_rules'])
# [{'emotion': 'sadness', 'keywords': [...], 'confidence': 0.85}]
```

## 测试

```bash
# 运行所有测试
pytest tests/ -v

# 运行特定测试
pytest tests/test_emotion_detector.py -v

# 运行演示
python -m trueemotion --demo
```

## 依赖

- Python 3.8+
- 无外部依赖（纯规则系统）

## 版本历史

- **v4.0.0**: 全新架构，Repository模式记忆系统，持续进化，零外部依赖
- **v3.1.0**: 长期记忆 + 自动进化
- **v3.0.1**: 集成TrueEmotionLife表达模块
- **v3.0.0**: 混合神经网络 + 规则系统

## License

MIT License
