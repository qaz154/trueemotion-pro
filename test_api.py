# -*- coding: utf-8 -*-
"""
TrueEmotion Pro API 测试
"""
import sys
sys.path.insert(0, 'src')

from trueemotion.api import analyze, EmotionAPI

print("=" * 50)
print("TrueEmotion Pro v3.1.0 API 测试")
print("=" * 50)

# 测试用例
tests = [
    "工作好累啊，老加班",
    "我升职了！太开心了！",
    "气死了！领导又批评我！",
    "被裁员了，感觉人生没有希望了...",
    "今天吃什么好呢？",
]

print("\n1. 情感检测:")
for t in tests:
    r = analyze(t)
    print("   {} -> {} ({:.2f})".format(t[:15], r.emotion, r.intensity))

print("\n2. 共情回复:")
for t in tests:
    r = analyze(t)
    print("   [{}] {}".format(r.emotion, r.reply))

print("\n3. 用户记忆:")
api = EmotionAPI()
result = api.analyze("我今天加班到很晚", learn=True, response="辛苦了，注意身体")
print("   用户状态: {}".format(result.user_state))

print("\n测试完成!")
