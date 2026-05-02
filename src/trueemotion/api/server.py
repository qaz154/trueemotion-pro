"""
TrueEmotion Pro v1.15 FastAPI Server
====================================
人性化的情感AI Web服务

v1.15 新特性:
- 修复所有已知严重Bug
- 进化系统真正生效
- 内存系统线程安全与原子写入
- 响应引擎优化
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, List

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel, Field

from trueemotion import TrueEmotionPro

logger = logging.getLogger(__name__)


def _serialize_result(result) -> dict:
    """序列化分析结果"""
    response_dict = {
        "version": result.version,
        "engine": result.engine,
        "emotion": {
            "primary": result.emotion.primary,
            "intensity": round(result.emotion.intensity, 3),
            "intensity_label": result.emotion.intensity_label,
            "vad": {
                "valence": result.emotion.vad[0],
                "arousal": result.emotion.vad[1],
                "dominance": result.emotion.vad[2],
            },
            "confidence": round(result.emotion.confidence, 3),
            "all_emotions": result.emotion.all_emotions,
            "compound_emotions": result.emotion.compound_emotions,
            "emotion_mix": result.emotion.emotion_mix,
        },
        "human_response": {
            "text": result.human_response.text,
            "empathy_type": result.human_response.empathy_type,
            "intensity_level": result.human_response.intensity_level,
            "follow_up": result.human_response.follow_up,
            "empathy_depth": result.human_response.empathy_depth,
            "tone": result.human_response.tone,
        },
    }
    if result.compound_emotions:
        response_dict["compound_emotions"] = result.compound_emotions
    if result.user_profile:
        response_dict["user_profile"] = {
            "user_id": result.user_profile.user_id,
            "total_interactions": result.user_profile.total_interactions,
            "dominant_emotion": result.user_profile.dominant_emotion,
            "relationship_level": round(result.user_profile.relationship_level, 3),
            "learned_patterns": result.user_profile.learned_patterns,
            "last_emotion": result.user_profile.last_emotion,
            "emotional_state": result.user_profile.emotional_state,
            "interaction_style": result.user_profile.interaction_style,
        }
    return response_dict


# Pydantic Models for API
class AnalyzeRequest(BaseModel):
    """分析请求模型"""
    text: str = Field(..., min_length=1, max_length=5000, description="待分析的文本")
    user_id: str = Field(default="default", description="用户ID")
    context: Optional[str] = Field(default=None, description="对话上下文")
    learn: bool = Field(default=False, description="是否学习此次交互")
    response: Optional[str] = Field(default=None, description="AI回复（用于学习）")
    feedback: float = Field(default=0.5, ge=0.0, le=1.0, description="用户反馈 0-1")


class LearnRequest(BaseModel):
    """学习请求模型"""
    text: str = Field(..., description="用户输入")
    emotion: str = Field(..., description="情感类型")
    response: str = Field(..., description="回复文本")
    feedback: float = Field(default=0.5, ge=0.0, le=1.0, description="反馈分数")
    user_id: str = Field(default="default", description="用户ID")


class ResetRequest(BaseModel):
    """重置请求模型"""
    user_id: str = Field(..., description="用户ID")


class BatchAnalyzeRequest(BaseModel):
    """批量分析请求模型"""
    texts: List[str] = Field(..., min_length=1, max_length=100, description="待分析的文本列表")
    user_id: str = Field(default="default", description="用户ID")


def _get_pro(request: Request) -> TrueEmotionPro:
    """从 app.state 获取实例"""
    pro = getattr(request.app.state, "pro", None)
    if pro is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    return pro


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    app.state.pro = TrueEmotionPro()
    yield
    app.state.pro = None


# Create FastAPI app
app = FastAPI(
    title="TrueEmotion Pro API",
    description="""
## TrueEmotion Pro v1.15 - 人性化情感AI系统 (LLM驱动)

让AI拥有像人类一样丰富、复杂、真实的情感。

### 核心功能
- **情感分析**: 识别文本中的情感，包括复合情感（LLM语义理解）
- **共情回复**: 生成人性化的共情回复（LLM动态生成）
- **持续学习**: 从交互中学习用户偏好
- **情感进化**: 分析模式并优化回复策略

### 引擎模式
- **LLM模式**: 使用 GPT-4o-mini 进行深度语义理解和动态响应生成
- **规则模式**: 传统规则引擎（当LLM不可用时自动降级）

### 情感类型
支持35+种情感：joy, sadness, anger, fear, anticipation, surprise, disgust, trust
以及复合情感如：bittersweet, hope_fear, love_hope 等
    """,
    version="1.15",
    lifespan=lifespan,
)

# CORS配置 - 生产环境应限制来源
CORS_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:8000",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:8000",
]

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/", tags=["Info"])
async def root():
    """服务信息"""
    return JSONResponse({
        "name": "TrueEmotion Pro",
        "version": "1.15",
        "description": "人性化情感AI系统 (LLM驱动)",
        "docs": "/docs",
        "health": "/health",
        "demo": "/demo",
    })


@app.get("/demo", response_class=HTMLResponse, tags=["Info"])
async def demo_page():
    """Web演示页面"""
    template_path = Path(__file__).parent / "templates" / "demo.html"
    return template_path.read_text(encoding="utf-8")


@app.get("/health", tags=["Info"])
async def health_check(request: Request):
    """健康检查"""
    pro = _get_pro(request)
    return JSONResponse({
        "status": "healthy",
        "version": "1.15",
        "engine": "llm-v1.15" if pro.is_llm_enabled else "rule-v1.15",
    })


@app.post("/analyze", tags=["Analysis"])
async def analyze_text(body: AnalyzeRequest, request: Request):
    """
    分析文本情感并生成共情回复

    - **text**: 待分析的文本
    - **user_id**: 用户ID（用于记忆和学习）
    - **context**: 对话上下文（可选）
    - **learn**: 是否学习此次交互
    - **response**: AI回复（用于学习）
    - **feedback**: 用户反馈 0-1
    """
    pro = _get_pro(request)

    try:
        result = pro.analyze(
            text=body.text,
            context=body.context,
            learn=body.learn,
            response=body.response,
            feedback=body.feedback,
            user_id=body.user_id,
        )

        data = _serialize_result(result)
        data["context_used"] = result.context_used
        data["explanation"] = result.explanation
        return JSONResponse({
            "success": True,
            "data": data,
        })
    except Exception as e:
        logger.exception("Error in /analyze endpoint")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/analyze/batch", tags=["Analysis"])
async def analyze_batch_text(body: BatchAnalyzeRequest, request: Request):
    """
    批量分析文本情感

    - **texts**: 待分析的文本列表（最多100条）
    - **user_id**: 用户ID（用于记忆和学习）
    """
    pro = _get_pro(request)

    try:
        results = pro.analyze_batch(
            texts=body.texts,
            user_id=body.user_id,
        )

        return JSONResponse({
            "success": True,
            "data": [_serialize_result(result) for result in results],
        })
    except Exception as e:
        logger.exception("Error in /analyze/batch endpoint")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/profile/{user_id}", tags=["User"])
async def get_user_profile(user_id: str, request: Request):
    """获取用户画像"""
    pro = _get_pro(request)

    try:
        profile = pro.get_user_profile(user_id)
        return JSONResponse({"success": True, "data": profile})
    except Exception as e:
        logger.exception("Error in /profile/%s endpoint", user_id)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/memory/status", tags=["Memory"])
async def get_memory_status(request: Request):
    """获取记忆状态"""
    pro = _get_pro(request)

    try:
        status = pro.get_memory_status()
        return JSONResponse({"success": True, "data": status})
    except Exception as e:
        logger.exception("Error in /memory/status endpoint")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/memory/reset", tags=["Memory"])
async def reset_user_memory(body: ResetRequest, request: Request):
    """重置用户记忆"""
    pro = _get_pro(request)

    try:
        pro.reset_user(body.user_id)
        return JSONResponse({
            "success": True,
            "message": f"用户 {body.user_id} 的记忆已重置",
        })
    except Exception as e:
        logger.exception("Error in /memory/reset endpoint")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/evolve", tags=["Evolution"])
async def trigger_evolution(request: Request):
    """执行情感进化"""
    pro = _get_pro(request)

    try:
        result = pro.evolve()
        return JSONResponse({"success": True, "data": result})
    except Exception as e:
        logger.exception("Error in /evolve endpoint")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/evolution/status", tags=["Evolution"])
async def get_evolution_status(request: Request):
    """获取进化状态"""
    pro = _get_pro(request)

    try:
        status = pro.get_evolution_status()
        return JSONResponse({"success": True, "data": status})
    except Exception as e:
        logger.exception("Error in /evolution/status endpoint")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/stats", tags=["Info"])
async def get_system_stats(request: Request):
    """获取系统统计"""
    pro = _get_pro(request)

    try:
        stats = pro.get_stats()
        return JSONResponse({"success": True, "data": stats})
    except Exception as e:
        logger.exception("Error in /stats endpoint")
        raise HTTPException(status_code=500, detail="Internal server error")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
