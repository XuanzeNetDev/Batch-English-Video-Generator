#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Demo 视频生成脚本
用于生成展示用的卡拉OK视频（电脑版 + 手机版）
"""

import sys
sys.path.append('.')

from karaoke_alignment_generator import KaraokeAlignmentGenerator
from karaoke_mobile_generator import KaraokeAlignmentGeneratorMobileV3Simple
from pathlib import Path


def main():
    """生成 Demo 视频（电脑版 + 手机版）"""
    print("=" * 60)
    print("🎬 卡拉OK视频生成器 - Demo 版本")
    print("=" * 60)
    print()
    
    # 检查音频文件
    audio_file = Path("Stories_audio/Story_01_A_Day_at_the_Park.wav")
    if not audio_file.exists():
        print("❌ 错误：找不到音频文件")
        print(f"   请确保音频文件存在: {audio_file}")
        print()
        print("📝 提示：")
        print("   1. 使用 TTS 工具生成英文音频")
        print("   2. 将音频保存为: Stories_audio/Story_01_A_Day_at_the_Park.wav")
        print("   3. 音频格式: WAV, 采样率: 16000Hz 或更高")
        return
    
    print(f"✅ 找到音频文件: {audio_file}")
    print()
    
    # 故事1配置
    story_number = 1
    
    # ========== 生成电脑版 ==========
    print("=" * 60)
    print(f"📺 开始生成故事{story_number:02d} - 电脑版（1920x1080）")
    print("=" * 60)
    print()
    
    try:
        generator_desktop = KaraokeAlignmentGenerator()
        generator_desktop.generate(story_num=story_number, use_forced_alignment=True)
        print()
        print("✅ 电脑版视频生成完成!")
        print("📁 输出: karaoke_alignment_videos/Story_01_Karaoke_Complete.mp4")
        print()
    except Exception as e:
        print()
        print("❌ 电脑版生成失败")
        print(f"错误信息: {e}")
        print()
        import traceback
        traceback.print_exc()
        return
    
    # ========== 生成手机版 ==========
    print("=" * 60)
    print(f"📱 开始生成故事{story_number:02d} - 手机版（1080x1920）")
    print("=" * 60)
    print()
    
    try:
        generator_mobile = KaraokeAlignmentGeneratorMobileV3Simple()
        generator_mobile.generate(story_num=story_number, use_forced_alignment=True)
        print()
        print("✅ 手机版视频生成完成!")
        print("📁 输出: karaoke_alignment_videos_mobile/Story_01_Karaoke_Mobile_V3_Simple.mp4")
        print()
    except Exception as e:
        print()
        print("❌ 手机版生成失败")
        print(f"错误信息: {e}")
        print()
        import traceback
        traceback.print_exc()
        return
    
    # ========== 完成 ==========
    print("=" * 60)
    print("🎉 所有 Demo 视频生成完成！")
    print("=" * 60)
    print()
    print("📁 输出文件:")
    print("   📺 电脑版: karaoke_alignment_videos/Story_01_Karaoke_Complete.mp4")
    print("   📱 手机版: karaoke_alignment_videos_mobile/Story_01_Karaoke_Mobile_V3_Simple.mp4")
    print()


if __name__ == "__main__":
    main()
