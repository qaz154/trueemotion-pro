# TrueEmotion Pro v1.11

**人性化情感AI系统** - 让AI拥有像人类一样丰富、复杂、真实的情感

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 核心理念

人类情感的丰富性来自：

| 维度 | 说明 |
|------|------|
| **情感复合** | 多种情感同时存在，如"喜极而泣"、"带泪的微笑" |
| **连续强度** | 情感不是0/1，而是0.0-1.0的连续光谱 |
| **情感记忆** | 过去的经历塑造当下的感受 |
| **性格建模** | 有人外放、有人内敛 |
| **关系感知** | 对不同人情感反应不同 |
| **情境依赖** | 同样的话在不同情境下含义不同 |

## 核心特性

| 特性 | 说明 |
|------|------|
| **35+情感类别** | 细腻情感分类，包含复合情感 |
| **复合情感检测** | 自动检测"悲喜交加"等复合情感 |
| **连续强度计算** | 0.0-1.0连续分数，强度标签（极微/微弱/轻微/中等/强烈/极致） |
| **VAD三维模型** | Valence-Arousal-Dominance 情感坐标 |
| **人性化共情回复** | 口语化、有温度、非模板化 |
| **性格与关系系统** | 个性化响应 |
| **持续进化** | 从对话中学习新模式 |
| **零依赖** | 纯规则系统，无需PyTorch |

## 架构

```
src/trueemotion/
├── api/                        # 统一API层
│   ├── routes.py              # TrueEmotionPro 主类
│   └── schemas.py             # 数据模型
├── core/                       # 核心领域
│   ├── emotions/
│   │   ├── plutchik24.py    # 35+情感定义及VAD
│   │   ├── detector.py       # 人性化检测器
│   │   └── personality.py     # 性格与关系系统
│   ├── analysis/
│   │   ├── analyzer.py       # 分析器门面
│   │   └── output.py          # 数据结构
│   └── response/
│       └── engine.py          # 人性化共情引擎
├── memory/                     # 记忆层 (Repository模式)
│   └── repository.py
└── learning/                  # 进化层
    └── evolution.py
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

print(f"主要情感: {result.emotion.primary}")           # joy
print(f"强度: {result.emotion.intensity}")           # 0.869
print(f"强度标签: {result.emotion.intensity_label}") # 强烈
print(f"VAD: {result.emotion.vad}")                  # (0.85, 0.50, 0.70)
print(f"情感混合: {result.emotion_mix}")             # 以喜悦为主，伴有轻微...
print(f"共情回复: {result.human_response.text}")    # "太为你高兴了！说说怎么回事！"
```

## 情感类别

### 主要情感 (8种)

| 情感 | VAD | 说明 |
|------|-----|------|
| joy | (0.85, 0.50, 0.70) | 开心、高兴、快乐 |
| sadness | (-0.85, -0.30, -0.50) | 难过、伤心、失落 |
| anger | (-0.85, 0.70, 0.50) | 生气、愤怒 |
| fear | (-0.65, 0.60, -0.40) | 害怕、担心、紧张 |
| disgust | (-0.75, -0.10, -0.40) | 恶心、讨厌 |
| surprise | (0.30, 0.80, 0.30) | 惊讶、意外 |
| trust | (0.65, 0.30, 0.70) | 相信、依赖 |
| anticipation | (0.50, 0.60, 0.40) | 期待、希望 |

### 复合情感 (15+种)

- **复合**: love, guilt, envy, contempt, optimism, despair, pride, etc.
- **细腻**: melancholy, nostalgia, longing, compassion, gratitude, regret, etc.
- **特殊**: bittersweet (悲喜交加), hope_fear (忐忑)

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
    "version": "1.11",
    "engine": "humanized-v1.11",
    "emotion": {
        "primary": "boredom",
        "intensity": 0.202,
        "vad": (-0.30, -0.50, -0.20),
        "confidence": 0.202,
        "intensity_label": "轻微",
        "all_emotions": {"boredom": 0.202},
        "compound_emotions": {},
        "emotion_mix": [("boredom", 0.202)],
    },
    "human_response": {
        "text": "嗯，听起来不太顺心",
        "empathy_type": "共情回应",
        "intensity_level": "low",
        "follow_up": "然后呢？",
        "empathy_depth": "温和",
        "tone": "温和",
    },
    "emotion_mix": "以倦怠为主",
}
```

## 共情回复示例

| 用户输入 | 情感 | TrueEmotion Pro 回复 |
|----------|------|---------------------|
| "太开心了！！！终于完成了项目！！" | joy (0.87) | "太为你高兴了！说说怎么回事！" |
| "被裁员了，感觉人生没有希望了..." | despair (0.24) | "先冷静一下，我陪着你" |
| "看着他成功了，既高兴又有点嫉妒" | joy+envy | "听起来挺复杂的，我理解那种感觉" |

## 持续进化

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
```

## 测试

```bash
# 运行所有测试
python test_api.py

# 运行pytest
pytest tests/ -v
```

## 依赖

- Python 3.8+
- 无外部依赖（纯规则系统）

## 版本历史

- **v1.11**: 人性化情感系统 - 复合情感、连续强度、性格建模、关系感知
- **v4.0**: 架构重构，零依赖
- **v3.x**: 混合神经网络 + 规则系统

## License

MIT License
