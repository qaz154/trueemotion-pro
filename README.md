# TrueEmotion Pro

新一代中文情感AI系统 - 混合神经网络 + 规则系统 + **有血有肉的情感表达**

[![GitHub stars](https://img.shields.io/github/stars/your-username/trueemotion?style=social)](https://github.com/your-username/trueemotion)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)

## 核心特性

| 特性 | 说明 |
|------|------|
| **15+情感类别** | joy, sadness, anger, fear, anxiety, surprise, love, trust, anticipation, optimism, guilt, envy, contempt, despair, disgust |
| **混合架构** | 神经网络 + 规则系统，兼顾泛化能力和精确性 |
| **VAD维度** | 输出Valence-Arousal-Dominance三维情感坐标 |
| **有血有肉回复** | 集成TrueEmotionLife，自动生成情感化、口语化、去AI化的回复 |
| **长期记忆** | 跨会话用户画像、交互历史持久化 |
| **自动进化** | 从对话中学习新模式，自动形成准则 |

## 安装

```bash
# 克隆仓库
git clone https://github.com/your-username/trueemotion.git
cd trueemotion

# 安装依赖
pip install torch transformers jieba

# 添加到Python路径
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

## 快速开始

```python
import sys
sys.path.insert(0, 'src')

from trueemotion.trueemotion_pro import TrueEmotionPro

# 初始化
pro = TrueEmotionPro()

# 分析单条文本
result = pro.analyze("今天涨工资了，太开心了！")
print(result['emotion']['primary'])  # joy
print(result['human_response']['text'])  # "太为你高兴了！说说怎么回事！"

# 开启学习和记忆
result = pro.analyze(
    "工作好累啊，老加班",
    learn=True,
    response="确实累，注意身体啊"
)
print(result['user_summary']['total_interactions'])  # 1
```

## v3.1.0 新功能

### 长期记忆 + 自动进化

```python
# 第一次对话 - 学习
pro.analyze("今天被裁员了...", learn=True, response="心疼你...")

# 第二次对话 - 系统记得用户的状态
summary = pro.get_user_summary()
print(summary['dominant_emotion'])  # 可能是fear/despair

# 查看记忆状态
print(pro.get_memory_status())
# {'total_users': 1, 'learned_patterns': 2, 'total_interactions': 5}
```

### 模式反馈学习

系统会从对话中自动学习新模式，并反哺到检测器：

```python
# 如果用户多次说"又被老板骂了"都是anger情感
# 系统会自动学习这个模式
# 下次检测"又被老板骂了"时会正确识别为anger
```

## 有血有肉的效果对比

| 用户输入 | 传统AI回复 | TrueEmotion Pro回复 |
|----------|------------|---------------------|
| "工作好累啊" | "我理解您的感受" | "说实话确实累...换我我也烦成这样" |
| "我升职了！" | "恭喜您" | "太为你高兴了！说说怎么回事！" |
| "气死了！" | "请您消消气" | "确实气人！换我我也急！" |
| "被裁员了..." | "很抱歉听到" | "先缓缓，我陪着你" |

## 情感类别

| 情感 | 中文 | VAD坐标 | 描述 |
|------|------|---------|------|
| joy | 喜悦 | (0.8, 0.5, 0.7) | 开心、高兴、快乐 |
| sadness | 悲伤 | (-0.8, -0.3, -0.5) | 难过、伤心、失落 |
| anger | 愤怒 | (-0.8, 0.7, 0.5) | 生气、愤怒，气愤 |
| fear | 恐惧 | (-0.6, 0.6, -0.4) | 害怕、担心、紧张 |
| anxiety | 焦虑 | (-0.5, 0.6, -0.4) | 焦虑、忧虑、着急 |
| surprise | 惊讶 | (0.3, 0.8, 0.3) | 意外、震惊、吃惊 |
| love | 爱 | (0.9, 0.4, 0.8) | 喜欢、爱慕、甜蜜 |
| trust | 信任 | (0.6, 0.3, 0.7) | 相信、依赖、放心 |
| anticipation | 期待 | (0.5, 0.6, 0.4) | 希望、盼望、憧憬 |
| optimism | 乐观 | (0.7, 0.5, 0.6) | 积极、阳光、信心 |
| guilt | 内疚 | (-0.5, 0.2, -0.4) | 愧疚、自责、后悔 |
| envy | 嫉妒 | (-0.4, 0.3, -0.3) | 羡慕、嫉妒、眼红 |
| contempt | 鄙视 | (-0.6, 0.2, 0.4) | 藐视、看不起 |
| despair | 绝望 | (-0.9, 0.3, -0.7) | 绝望、放弃、崩溃 |
| disgust | 厌恶 | (-0.7, -0.1, -0.4) | 恶心、讨厌、厌烦 |

## API文档

### TrueEmotionPro.analyze()

```python
result = pro.analyze(
    text="工作好累啊，老加班",
    context=None,           # 对话上下文
    learn=True,             # 是否学习
    response="确实累...",    # AI回复
    feedback=0.8,           # 用户反馈
    user_id="user123"      # 用户ID
)
```

**返回:**
```python
{
    "version": "3.1.0",
    "engine": "hybrid",
    "emotion": {
        "primary": "sadness",
        "intensity": 0.6,
        "vad": (-0.8, -0.3, -0.5),
        "confidence": 0.85
    },
    "human_response": {
        "text": "说实话确实累...换我我也烦成这样",
        "empathy_type": "support",
        "intensity_level": "high",
        "follow_up": "说说怎么回事？"
    },
    "user_summary": {
        "total_interactions": 5,
        "dominant_emotion": "sadness",
        "relationship_level": 0.5
    }
}
```

### 其他方法

```python
# 获取用户信息
pro.get_user_info(user_id="user123")

# 获取记忆状态
pro.get_memory_status()
# {'total_users': 1, 'learned_patterns': 3, 'current_user': 'default'}

# 执行进化
pro.evolve()
# {'emotion_evolution': {...}, 'rule_status': {...}}
```

## 系统架构

```
输入文本
    ↓
┌─────────────────────────────────────────────────────────┐
│ TrueEmotionPro v3.1.0                                  │
│                                                          │
│ ┌─────────────────────────────────────────────────────┐│
│ │ 情感检测层 (HybridEmotionAnalyzer)                    ││
│ │ ├── RuleBasedEmotionDetector (规则系统)              ││
│ │ │   └── LearnedPatterns (学习到的模式) ←──────┐     ││
│ │ └── CharEmotionAnalyzer (CNN神经网络)           │     ││
│ └─────────────────────────────────────────────────────┘│
│    ↓                                                     │
│ ┌─────────────────────────────────────────────────────┐│
│ │ 进化层 (EvolutionManager)                            ││
│ │ └── 从对话中学习，自动形成准则                        ││
│ └─────────────────────────────────────────────────────┘│
│    ↓                                                     │
│ ┌─────────────────────────────────────────────────────┐│
│ │ 记忆层 (MemorySystem)                               ││
│ │ ├── 用户画像持久化                                   ││
│ │ └── 学习模式反哺检测器 ────────────────────────┘     ││
│ └─────────────────────────────────────────────────────┘│
│    ↓                                                     │
│ ┌─────────────────────────────────────────────────────┐│
│ │ 表达层 (TrueEmotionLife Expression)                 ││
│ │ ├── EmpathyEngine (共情引擎)                        ││
│ │ ├── PersonalityExpressor (个性化表达)               ││
│ │ └── NaturalLanguageGenerator (口语化NLG)            ││
│ └─────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────┘
    ↓
情感输出 + 有血有肉的回复
```

## 评估结果

### 情感检测准确率

测试数据集: 5730个真实对话样本

| 情感 | 分类准确率 |
|------|-----------|
| guilt | 100% |
| envy | 100% |
| despair | 100% |
| trust | 100% |
| optimism | 100% |
| anticipation | 100% |
| joy | 100% |
| disgust | 100% |
| anger | 95.7% |
| sadness | 93.8% |
| fear | 93.3% |
| anxiety | 92.9% |
| surprise | 91.7% |
| contempt | 90.0% |
| love | 90.0% |
| **总体** | **96.0%** |

## 使用场景

- 对话系统情感识别 + 自然回复
- 社交媒体情感分析
- 客服对话质检
- 用户反馈分析
- 心理健康监测
- AI Agent情感化交互

## 依赖

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.0+
- Jieba 0.42+

## 版本历史

- **v3.1.0**: 长期记忆 + 自动进化 + 模式反馈学习
- **v3.0.1**: 集成TrueEmotionLife表达模块
- **v3.0.0**: 混合神经网络 + 规则系统，15+情感类别

## License

MIT License

## Citation

```bibtex
@misc{trueemotion,
  title={TrueEmotion Pro - 新一代中文情感AI系统},
  author={TrueEmotion Team},
  year={2026},
  url={https://github.com/your-username/trueemotion}
}
```
