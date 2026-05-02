# TrueEmotion Pro v1.15

**人性化情感AI系统** - 让AI拥有像人类一样丰富、复杂、真实的情感

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-1.14-blue.svg)]()

## 核心理念

人类情感的丰富性来自：

| 维度 | 说明 |
|------|------|
| **情感复合** | 多种情感同时存在，如"喜极而泣"、"带泪的微笑" |
| **连续强度** | 情感不是0/1，而是0.0-1.0的连续光谱 |
| **情感记忆** | 过去的经历塑造当下的感受，记忆持久且会衰减 |
| **性格建模** | 有人外放、有人内敛 |
| **关系感知** | 对不同人情感反应不同 |
| **情境依赖** | 同样的话在不同情境下含义不同 |

## 核心特性

| 特性 | 说明 |
|------|------|
| **40+情感类别** | 包含基础情感、细腻情感、复合情感 |
| **LLM 语义理解** | 使用 GPT-4o-mini 进行深度语义情感检测 |
| **动态响应生成** | LLM 驱动，口语化、个性化、非模板化 |
| **复合情感检测** | 自动检测"悲喜交加"、"吃醋"等复合情感 |
| **连续强度计算** | 0.0-1.0连续分数，强度标签（极微/微弱/轻微/中等/强烈/极致） |
| **VAD三维模型** | Valence-Arousal-Dominance 情感坐标系统 |
| **反讽检测** | 识别"挺好的"等反讽表达的真实情感 |
| **人性化共情回复** | 口语化、有温度，支持多种语气 |
| **智能记忆系统** | 关键词提取、相似度匹配、强化衰减机制 |
| **持续进化** | 从对话中学习新模式，多维度置信度计算 |
| **自动降级机制** | LLM 不可用时自动切换到规则引擎 |
| **FastAPI Web API** | 完整的REST接口，支持Web演示页面 |

## 架构

```
src/trueemotion/
├── api/                        # 统一API层
│   ├── routes.py              # TrueEmotionPro 主类
│   ├── server.py              # FastAPI Web服务
│   ├── schemas.py             # 数据模型
│   └── templates/
│       └── demo.html          # Web演示页面
├── core/                       # 核心领域
│   ├── llm/                   # LLM 驱动模块 (v1.15 新增)
│   │   ├── base.py           # LLM 抽象接口
│   │   ├── openai_client.py  # OpenAI 实现
│   │   ├── emotion_detector.py # LLM 情感检测
│   │   ├── response_generator.py # LLM 响应生成
│   │   ├── fallback.py       # 降级管理器
│   │   └── prompts.py        # Prompt 模板
│   ├── emotions/
│   │   ├── plutchik24.py     # 40+情感定义及VAD坐标
│   │   ├── detector.py       # 规则引擎检测器
│   │   ├── irony.py         # 反讽检测器
│   │   └── personality.py    # 性格与关系系统
│   ├── analysis/
│   │   ├── analyzer.py       # 分析器门面 (双引擎支持)
│   │   ├── context.py        # 上下文分析器
│   │   └── output.py        # 数据结构
│   └── response/
│       ├── engine.py          # 规则引擎共情响应
│       └── proactive.py       # 主动共情引擎
├── memory/                     # 记忆层 (Repository模式)
│   └── repository.py          # 智能记忆系统
└── learning/                  # 进化层
    └── evolution.py           # 多维度进化管理
```

## 安装

```bash
# 克隆仓库
git clone https://github.com/qaz154/trueemotion-pro.git
cd trueemotion-pro

# 安装依赖
pip install fastapi uvicorn pydantic

# 设置 Python 路径
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"  # Linux/Mac
set PYTHONPATH=%PYTHONPATH%;%cd%\src          # Windows
```

## 快速开始

### Python API - 规则引擎模式

```python
import sys
sys.path.insert(0, 'src')

from trueemotion import TrueEmotionPro

# 初始化（规则引擎模式，无需 API Key）
pro = TrueEmotionPro()

# 分析文本
result = pro.analyze("今天太开心了！终于完成了项目！")

print(f"主要情感: {result.emotion.primary}")           # joy
print(f"强度: {result.emotion.intensity}")             # 0.85
print(f"强度标签: {result.emotion.intensity_label}")  # 强烈
print(f"VAD: {result.emotion.vad}")                   # (0.85, 0.50, 0.70)
print(f"共情回复: {result.human_response.text}")      # "太为你高兴了！说说怎么回事！"
print(f"追问: {result.human_response.follow_up}")     # "然后呢？"
```

### Python API - LLM 模式

```python
import sys
sys.path.insert(0, 'src')

from trueemotion import TrueEmotionPro

# 初始化（LLM 模式，需要 OpenAI API Key）
pro = TrueEmotionPro(
    llm_provider="openai",
    api_key="sk-...",
    llm_model="gpt-4o-mini"
)

# LLM 模式能理解深层语义
result = pro.analyze("今天被老板画饼了，感觉很失落")
# 规则引擎可能识别为 joy（因为有"老板"等正面词）
# LLM 正确识别为 disappointment + frustration

print(f"主要情感: {result.emotion.primary}")  # disappointment
print(f"共情回复: {result.human_response.text}")  # "老板画饼确实让人很失落..."
print(f"引擎版本: {result.engine}")  # llm-v1.15
```

### Web API

```bash
# 启动服务（规则引擎模式）
uvicorn trueemotion.api.server:app --reload --port 8000

# 启动服务（LLM 模式）
export OPENAI_API_KEY="sk-..."
uvicorn trueemotion.api.server:app --reload --port 8000

# API文档
# http://localhost:8000/docs

# Web演示页面
# http://localhost:8000/demo
```

### REST API 端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/analyze` | POST | 分析文本情感 |
| `/analyze/batch` | POST | 批量分析文本情感 |
| `/profile/{user_id}` | GET | 获取用户画像 |
| `/memory/status` | GET | 获取记忆状态 |
| `/evolve` | POST | 触发情感进化 |
| `/stats` | GET | 获取系统统计 |

## LLM 模式 vs 规则引擎模式

| 特性 | 规则引擎 | LLM 模式 |
|------|----------|----------|
| 情感检测 | 关键词匹配 | 深度语义理解 |
| 响应生成 | 模板选择 | 动态生成 |
| 复杂语义 | 可能误判 | 准确理解 |
| API 依赖 | 无 | 需要 OpenAI API Key |
| 降级机制 | - | 失败自动切换规则引擎 |

### LLM 模式优势

1. **语义理解**: 能理解"画饼"、"扎心"等网络用语
2. **动态生成**: 响应不是模板，而是根据情境动态生成
3. **复杂情感**: 能识别三元及以上的复杂情感组合
4. **上下文感知**: 能理解对话历史和用户关系

### 降级机制

当 LLM 不可用或连续失败时：
- 自动切换到规则引擎
- 熔断器模式防止服务不可用
- 连续 5 次失败后打开熔断器
- 60 秒后尝试恢复

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

| 情感 | 组合 | 说明 |
|------|------|------|
| bittersweet | joy + sadness | 悲喜交加 |
| love | joy + trust | 爱慕 |
| hope | joy + anticipation | 希望 |
| despair | sadness + fear | 绝望 |
| anxiety | fear + anticipation | 焦虑 |
| jealous_love | love + envy | 吃醋 |
| happy_sadness | joy + sadness | 又开心又难过 |
| painful_joy | joy + sadness + trust | 喜极而泣 |
| frustration_hopelessness | sadness + anger + guilt | 无奈绝望 |
| hope_fear | joy + anticipation + fear | 忐忑不安 |

### 细腻情感

- **柔和**: serenity, pensiveness, boredom, acceptance, interest
- **复杂**: melancholy, nostalgia, longing, compassion, gratitude
- **极端**: ecstasy, grief, rage, terror, loathing, astonishment

## 智能记忆系统

### 核心特性

```python
# 学习新模式（自动提取关键词）
pro.analyze(
    text="工作好累啊",
    learn=True,
    response="确实累，注意身体",
    feedback=0.9,  # 高反馈会被强化
    user_id="user1"
)

# 查找相似模式
patterns = pro._memory.find_similar_patterns(
    user_id="user1",
    emotion="sadness",
    text="被老板骂了，心情很差"
)

# 触发进化
result = pro.evolve()
```

### 记忆强化衰减机制

- **高反馈(≥0.8)**: 模式自动同步到全局模式库，跨用户共享
- **使用时强化**: 每次使用增加 REINFORCEMENT_BOOST (0.15)
- **闲置衰减**: 长期未使用的模式会缓慢衰减
- **相似度匹配**: 60%关键词重叠认为是相似模式

## 共情回复示例

| 用户输入 | 情感 | TrueEmotion Pro 回复 |
|----------|------|---------------------|
| "太开心了！！！终于完成了项目！！" | joy (0.87) | "太为你高兴了！说说怎么回事！" |
| "被裁员了，感觉人生没有希望了..." | despair (0.24) | "先冷静一下，我陪着你" |
| "看着他成功了，既高兴又有点嫉妒" | envy+joy | "确实会羡慕呢，能理解" |
| "吃醋了，男朋友和别的女生说话" | jealous_love | "很正常的心情" |
| "又开心又难过，五味杂陈" | happy_sadness | "五味杂陈啊，这种感觉最难形容了" |

## 反讽检测

```python
# 检测反讽
result = pro.analyze("挺好的，你真行啊！")
# 如果检测到反讽：
# result.explanation["irony"] = {
#     "is_irony": True,
#     "surface_emotion": "joy",
#     "true_emotion": "contempt",
#     "confidence": 0.75
# }
```

## 测试

```bash
# 运行所有测试
python test_api.py

# 运行pytest
pytest tests/ -v

# 验证核心功能
python -c "
import sys; sys.path.insert(0, 'src')
from trueemotion import TrueEmotionPro
pro = TrueEmotionPro()
r = pro.analyze('今天太开心了！')
assert r.emotion.primary == 'joy'
print('所有测试通过!')
"
```

## 依赖

- Python 3.9+
- **核心系统**: 无外部依赖（纯规则系统）
- **LLM 模式**: 需要 OpenAI API Key
- **Web API**: fastapi, uvicorn, pydantic

## 环境变量

| 变量 | 说明 | 必填 |
|------|------|------|
| `OPENAI_API_KEY` | OpenAI API Key | LLM 模式必填 |
| `OPENAI_API_BASE` | API 基础 URL（可选，用于代理） | 否 |

## 版本历史

| 版本 | 说明 |
|------|------|
| **v1.15** | 大规模Bug修复与系统优化 - 进化系统生效、内存安全、版本统一、响应引擎增强 |
| **v1.14** | LLM 驱动升级 - 语义情感检测、动态响应生成、自动降级 |
| **v1.13** | 全面升级 - 智能记忆系统、增强复合情感、多维度进化 |
| **v1.12** | FastAPI Web API与Web演示页面 |
| **v1.11** | 人性化情感系统 - 复合情感、连续强度、性格建模 |

## License

MIT License - 详见 [LICENSE](LICENSE) 文件

---

**TrueEmotion Pro** - 让AI拥有像人类一样丰富的情感
