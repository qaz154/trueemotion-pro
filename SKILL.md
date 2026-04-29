# TrueEmotion Pro Skill

让 Agent 调用情感分析能力。

## 功能

- 分析用户输入的情感（40+情感类别）
- 生成有血有肉的共情回复
- 跨对话记忆用户情感状态
- 自动学习新模式
- 持续进化优化
- **LLM 驱动的语义理解**（v1.14 新增）
- **动态响应生成**（v1.14 新增）
- **自动降级机制**（v1.14 新增）

## 使用方法

### 方式1: 直接调用 TrueEmotionPro

```python
import sys
sys.path.insert(0, '/path/to/trueemotion/src')

from trueemotion import TrueEmotionPro

# 规则引擎模式（默认）
pro = TrueEmotionPro()

# LLM 模式（需要 OpenAI API Key）
pro = TrueEmotionPro(
    llm_provider="openai",
    api_key="sk-...",
    llm_model="gpt-4o-mini"
)

# 简单分析
result = pro.analyze("工作好累啊")
print(result.emotion.primary)  # sadness
print(result.human_response.text)  # "确实累，注意身体啊"

# 带学习（让 Agent 记住这次交互）
result = pro.analyze(
    "被裁员了...",
    learn=True,
    response="心疼你，先缓缓",
    feedback=0.9
)
```

### 方式2: 使用 FastAPI Web服务

```bash
# 启动服务（规则引擎模式）
uvicorn trueemotion.api.server:app --reload --port 8000

# 启动服务（LLM 模式，设置环境变量）
export OPENAI_API_KEY="sk-..."
uvicorn trueemotion.api.server:app --reload --port 8000

# API调用
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "今天太开心了！"}'
```

## AnalysisResult 属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `version` | str | 版本号，如 "1.14" |
| `engine` | str | 引擎名，如 "llm-v1.14" 或 "rule-v1.14" |
| `emotion.primary` | str | 主要情感 (joy, sadness, anger...) |
| `emotion.intensity` | float | 情感强度 0-1 |
| `emotion.intensity_label` | str | 强度标签（极微/微弱/轻微/中等/强烈/极致） |
| `emotion.vad` | tuple | VAD 维度 (valence, arousal, dominance) |
| `emotion.confidence` | float | 置信度 0-1 |
| `emotion.all_emotions` | dict | 所有检测到的情感及强度 |
| `emotion.compound_emotions` | dict | 复合情感 |
| `human_response.text` | str | 共情回复文本 |
| `human_response.empathy_type` | str | 共情类型 |
| `human_response.follow_up` | str | 追问建议 |
| `user_profile` | dict | 用户画像 |

## 支持的情感类别

### 主要情感 (8种)
- joy (喜悦), sadness (悲伤), anger (愤怒), fear (恐惧)
- disgust (厌恶), surprise (惊讶), trust (信任), anticipation (期待)

### 复合情感 (15+种)
- bittersweet (悲喜交加), love (爱), hope (希望), despair (绝望)
- anxiety (焦虑), jealous_love (吃醋), happy_sadness (又开心又难过)
- painful_joy (喜极而泣), frustration_hopelessness (无奈绝望)

### 细腻情感
- melancholy (忧郁), nostalgia (怀旧), loneliness (孤独)
- gratitude (感激), pride (自豪), shame (羞耻)

## 共情类型

- `分享喜悦`: 正面情感
- `深度共情`: 负面情感
- `安抚情绪`: 愤怒情感
- `安全感提供`: 恐惧情感
- `缓解焦虑`: 焦虑情感
- `理解复杂`: 复合情感

## LLM 模式说明

v1.14 引入了 LLM 驱动的情感分析，相比规则引擎：

### 优势
1. **语义理解**: 能理解深层语义，如"画饼"理解为失望+愤世嫉俗
2. **动态生成**: 响应不是模板，而是根据情境动态生成
3. **复杂情感**: 能识别三元及以上的复杂情感组合
4. **上下文感知**: 能理解对话历史和用户关系

### 降级机制
- 当 LLM 不可用或调用失败时，自动降级到规则引擎
- 熔断器模式防止 LLM 故障导致服务不可用
- 连续 5 次失败后打开熔断器，60 秒后尝试恢复

### 配置选项

```python
# 环境变量
export OPENAI_API_KEY="sk-..."
export OPENAI_API_BASE="https://api.openai.com/v1"  # 可选，用于代理

# 代码配置
pro = TrueEmotionPro(
    llm_provider="openai",      # "openai" 或 None
    llm_model="gpt-4o-mini",   # 模型名称
    api_key="sk-...",          # 或从环境变量获取
    enable_llm=True,            # 是否启用 LLM
)
```

## 示例场景

### 对话 Agent

```python
from trueemotion import TrueEmotionPro

pro = TrueEmotionPro(
    llm_provider="openai",
    api_key="sk-..."
)

def handle_message(user_text: str, agent_response: str) -> str:
    # 分析用户情感
    result = pro.analyze(
        user_text,
        learn=True,
        response=agent_response,
        feedback=0.8
    )

    # 返回共情回复
    return result.human_response.text
```

### 情感状态跟踪

```python
# 检测到用户持续低落
result = pro.analyze("又被老板骂了")
if result.emotion.primary == 'anger' and result.emotion.intensity > 0.7:
    # 触发安慰模式
    print("检测到用户愤怒强度较高，使用安慰策略")
```

### 批量分析

```python
# 批量分析多条文本
texts = ["很开心！", "很难过...", "气死了"]
results = pro.analyze_batch(texts)
for r in results:
    print(f"{r.emotion.primary}: {r.human_response.text}")
```

### 降级模式使用

```python
# 不使用 LLM，只用规则引擎
pro = TrueEmotionPro(llm_provider=None)
result = pro.analyze("今天很开心")
print(f"Engine: {result.engine}")  # rule-v1.14
```
