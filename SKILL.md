# TrueEmotion Pro Skill

让 Agent 调用情感分析能力。

## 功能

- 分析用户输入的情感
- 生成有血有肉的共情回复
- 跨对话记忆用户情感状态
- 自动学习新模式

## 使用方法

### 方式1: 直接调用 API

```python
import sys
sys.path.insert(0, '/path/to/trueemotion/src')

from trueemotion.api import analyze

# 简单分析
result = analyze("工作好累啊")
print(result.emotion)  # sadness
print(result.reply)     # "说实话，哎...会好起来的呀"

# 带学习（让 Agent 记住这次交互）
result = analyze(
    "被裁员了...",
    learn=True,
    response="心疼你，先缓缓"
)
```

### 方式2: 使用 EmotionAPI 类

```python
from trueemotion.api import EmotionAPI

api = EmotionAPI()

# 分析情感
result = api.analyze("工作好累啊")
print(f"情感: {result.emotion}, 回复: {result.reply}")

# 获取用户历史
history = api.get_user_history(user_id="alice")
print(f"最近对话: {history}")

# 获取用户画像
info = api.get_user_info(user_id="alice")
print(f"用户状态: {info}")
```

## EmotionResult 属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `emotion` | str | 主要情感 (joy, sadness, anger...) |
| `intensity` | float | 情感强度 0-1 |
| `confidence` | float | 置信度 0-1 |
| `vad` | tuple | VAD 维度 (valence, arousal, dominance) |
| `reply` | str | 有血有肉的回复 |
| `empathy_type` | str | 共情类型 |
| `user_state` | dict | 用户状态摘要 |

## 支持的情感类别

- joy (喜悦)
- sadness (悲伤)
- anger (愤怒)
- fear (恐惧)
- anxiety (焦虑)
- surprise (惊讶)
- love (爱)
- trust (信任)
- anticipation (期待)
- optimism (乐观)
- guilt (内疚)
- envy (嫉妒)
- contempt (鄙视)
- despair (绝望)
- disgust (厌恶)
- neutral (中性)

## 共情类型

- `support`: 支持安慰
- `venting`: 发泄疏导
- `action`: 行动建议
- `sharing`: 分享快乐
- `understanding`: 理解共情
- `celebration`: 庆祝祝贺

## 示例场景

### 对话 Agent

```python
from trueemotion.api import EmotionAPI

api = EmotionAPI()

def handle_message(user_text: str, agent_response: str) -> str:
    # 分析用户情感
    result = api.analyze(
        user_text,
        learn=True,
        response=agent_response
    )

    # 生成共情回复
    return result.reply
```

### 情感状态跟踪

```python
# 检测到用户持续低落
result = api.analyze("又被老板骂了")
if result.emotion == 'anger' and result.intensity > 0.7:
    # 触发安慰模式
    print("检测到用户愤怒强度较高，使用安慰策略")
```
