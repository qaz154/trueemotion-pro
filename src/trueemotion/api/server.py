"""
TrueEmotion Pro v1.13 FastAPI Server
====================================
人性化的情感AI Web服务
"""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel, Field

from trueemotion import TrueEmotionPro


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


# Global instance
_pro_instance: Optional[TrueEmotionPro] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global _pro_instance
    _pro_instance = TrueEmotionPro()
    yield
    _pro_instance = None


# Create FastAPI app
app = FastAPI(
    title="TrueEmotion Pro API",
    description="""
## TrueEmotion Pro v1.13 - 人性化情感AI系统

让AI拥有像人类一样丰富、复杂、真实的情感。

### 核心功能
- **情感分析**: 识别文本中的情感，包括复合情感
- **共情回复**: 生成人性化的共情回复
- **持续学习**: 从交互中学习用户偏好
- **情感进化**: 分析模式并优化回复策略

### 情感类型
支持35+种情感：joy, sadness, anger, fear, anticipation, surprise, disgust, trust
以及复合情感如：bittersweet, hope_fear, love_hope 等
    """,
    version="1.13",
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
        "version": "1.13",
        "description": "人性化情感AI系统",
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
async def health_check():
    """健康检查"""
    if _pro_instance is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    return JSONResponse({"status": "healthy", "version": "1.13"})


@app.post("/analyze", tags=["Analysis"])
async def analyze_text(request: AnalyzeRequest):
    """
    分析文本情感并生成共情回复

    - **text**: 待分析的文本
    - **user_id**: 用户ID（用于记忆和学习）
    - **context**: 对话上下文（可选）
    - **learn**: 是否学习此次交互
    - **response**: AI回复（用于学习）
    - **feedback**: 用户反馈 0-1
    """
    if _pro_instance is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        result = _pro_instance.analyze(
            text=request.text,
            context=request.context,
            learn=request.learn,
            response=request.response,
            feedback=request.feedback,
            user_id=request.user_id,
        )

        return JSONResponse({
            "success": True,
            "data": {
                "version": result.version,
                "engine": result.engine,
                "emotion": {
                    "primary": result.emotion.primary,
                    "intensity": result.emotion.intensity,
                    "intensity_label": result.emotion.intensity_label,
                    "vad": {
                        "valence": result.emotion.vad[0],
                        "arousal": result.emotion.vad[1],
                        "dominance": result.emotion.vad[2],
                    },
                    "confidence": result.emotion.confidence,
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
                "user_profile": {
                    "user_id": result.user_profile.user_id,
                    "total_interactions": result.user_profile.total_interactions,
                    "dominant_emotion": result.user_profile.dominant_emotion,
                    "relationship_level": result.user_profile.relationship_level,
                    "learned_patterns": result.user_profile.learned_patterns,
                    "last_emotion": result.user_profile.last_emotion,
                    "emotional_state": result.user_profile.emotional_state,
                    "interaction_style": result.user_profile.interaction_style,
                },
                "context_used": result.context_used,
                "explanation": result.explanation,
            },
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze/batch", tags=["Analysis"])
async def analyze_batch_text(request: BatchAnalyzeRequest):
    """
    批量分析文本情感

    - **texts**: 待分析的文本列表（最多100条）
    - **user_id**: 用户ID（用于记忆和学习）
    """
    if _pro_instance is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        results = _pro_instance.analyze_batch(
            texts=request.texts,
            user_id=request.user_id,
        )

        return JSONResponse({
            "success": True,
            "data": [
                {
                    "version": result.version,
                    "engine": result.engine,
                    "emotion": {
                        "primary": result.emotion.primary,
                        "intensity": result.emotion.intensity,
                        "intensity_label": result.emotion.intensity_label,
                        "vad": {
                            "valence": result.emotion.vad[0],
                            "arousal": result.emotion.vad[1],
                            "dominance": result.emotion.vad[2],
                        },
                        "confidence": result.emotion.confidence,
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
                for result in results
            ],
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/profile/{user_id}", tags=["User"])
async def get_user_profile(user_id: str):
    """获取用户画像"""
    if _pro_instance is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        profile = _pro_instance.get_user_profile(user_id)
        return JSONResponse({"success": True, "data": profile})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/memory/status", tags=["Memory"])
async def get_memory_status():
    """获取记忆状态"""
    if _pro_instance is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        status = _pro_instance.get_memory_status()
        return JSONResponse({"success": True, "data": status})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/memory/reset", tags=["Memory"])
async def reset_user_memory(request: ResetRequest):
    """重置用户记忆"""
    if _pro_instance is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        _pro_instance.reset_user(request.user_id)
        return JSONResponse({
            "success": True,
            "message": f"用户 {request.user_id} 的记忆已重置",
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/evolve", tags=["Evolution"])
async def trigger_evolution():
    """执行情感进化"""
    if _pro_instance is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        result = _pro_instance.evolve()
        return JSONResponse({"success": True, "data": result})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/evolution/status", tags=["Evolution"])
async def get_evolution_status():
    """获取进化状态"""
    if _pro_instance is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        status = _pro_instance.get_evolution_status()
        return JSONResponse({"success": True, "data": status})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats", tags=["Info"])
async def get_system_stats():
    """获取系统统计"""
    if _pro_instance is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        stats = _pro_instance.get_stats()
        return JSONResponse({"success": True, "data": stats})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
