#!/usr/bin/env python3
"""
卡拉OK对齐字幕生成器 - 逐词高亮跟踪
解决字幕精度问题 + 添加颜色跟踪朗读效果
使用 torchaudio Forced Alignment 实现精确对齐
"""

import os
import numpy as np
from pathlib import Path
import librosa
import re
import moviepy as mp
from PIL import Image, ImageDraw, ImageFont
import cv2
from difflib import SequenceMatcher
import platform

# PyTorch for forced alignment
import torch
import torchaudio


class KaraokeAlignmentGenerator:
    """卡拉OK对齐字幕生成器"""
    
    def __init__(self):
        self.output_dir = Path("karaoke_alignment_videos")
        self.output_dir.mkdir(exist_ok=True)
        
        self.temp_dir = Path("temp_karaoke_alignment")
        self.temp_dir.mkdir(exist_ok=True)
        
        self.audio_dir = Path("Stories_audio")
        self.english_dir = Path("English_Stories")
        self.chinese_dir = Path("Chinese_Stories")
        
        # 颜色配置 - 第一个故事用冰蓝色，其他随机
        self.color_scheme = None
        
        # 设备配置
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print("🎤 卡拉OK对齐字幕生成器初始化")
        print(f"   设备: {self.device}")
    
    def get_system_font(self, size):
        """跨平台获取系统字体"""
        system = platform.system()
        
        try:
            if system == "Windows":
                # Windows 字体
                font_paths = [
                    "C:\\Windows\\Fonts\\simsun.ttc",
                    "C:\\Windows\\Fonts\\msyh.ttc",
                    "C:\\Windows\\Fonts\\arial.ttf"
                ]
            elif system == "Darwin":  # macOS
                font_paths = [
                    "/System/Library/Fonts/PingFang.ttc",
                    "/System/Library/Fonts/Helvetica.ttc",
                    "/Library/Fonts/Arial Unicode.ttf"
                ]
            else:  # Linux
                font_paths = [
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
                    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
                ]
            
            # 尝试加载字体
            for font_path in font_paths:
                if os.path.exists(font_path):
                    return ImageFont.truetype(font_path, size)
            
            # 如果都失败，使用默认字体
            print(f"⚠️ 未找到系统字体，使用默认字体")
            return ImageFont.load_default()
            
        except Exception as e:
            print(f"⚠️ 字体加载失败: {e}，使用默认字体")
            return ImageFont.load_default()
    
    def extract_word_timestamps_with_forced_alignment(self, audio_path: str, english_text: str) -> list:
        """使用 torchaudio Forced Alignment 提取精确词级时间戳
        
        关键改进：使用原文文本进行强制对齐，而不是依赖 Whisper 的识别结果
        这样可以解决数字识别不一致的问题（如 "two hundred" vs "200"）
        """
        print("🎤 使用 torchaudio Forced Alignment 提取词级时间戳...")
        
        # 加载 wav2vec2 模型
        print("   📝 Step 1: 加载 wav2vec2 对齐模型...")
        bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
        model = bundle.get_model().to(self.device)
        labels = bundle.get_labels()
        
        # 构建字典
        dictionary = {c.lower(): i for i, c in enumerate(labels)}
        
        # 加载音频
        print("   📝 Step 2: 加载音频...")
        import soundfile as sf
        audio_data, sample_rate = sf.read(audio_path)
        waveform = torch.tensor(audio_data).float()
        if len(waveform.shape) == 1:
            waveform = waveform.unsqueeze(0)
        elif waveform.shape[1] == 2:  # stereo to mono
            waveform = waveform.mean(dim=1, keepdim=True).T
        
        # 重采样到模型需要的采样率
        if sample_rate != bundle.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)
        
        waveform = waveform.to(self.device)
        
        # 获取模型输出
        print("   📝 Step 3: 获取声学特征...")
        with torch.inference_mode():
            emissions, _ = model(waveform)
            emissions = torch.log_softmax(emissions, dim=-1)
        
        emission = emissions[0].cpu().detach()
        
        # 准备文本
        # 将文本转换为 token 序列
        transcript = self._prepare_transcript(english_text, dictionary)
        tokens = [dictionary.get(c, 0) for c in transcript]
        
        print(f"   📝 Step 4: 执行强制对齐 ({len(tokens)} tokens)...")
        
        # 构建 trellis 矩阵
        trellis = self._get_trellis(emission, tokens)
        
        # 回溯找到最佳路径
        path = self._backtrack(trellis, emission, tokens)
        
        if path is None:
            print("   ⚠️ 强制对齐失败，回退到 Whisper 方法")
            return self.extract_word_timestamps(audio_path)
        
        # 合并重复的字符
        segments = self._merge_repeats(path, transcript)
        
        # 将字符级时间戳转换为词级时间戳
        word_segments = self._chars_to_words(segments, english_text, emission.shape[0], bundle.sample_rate)
        
        print(f"   ✅ 提取 {len(word_segments)} 个词的精确时间戳")
        
        # 清理
        del model
        if self.device == "cuda":
            torch.cuda.empty_cache()
        
        return word_segments
    
    def _prepare_transcript(self, text: str, dictionary: dict) -> str:
        """准备用于对齐的转录文本"""
        # 转换为小写，用 | 表示空格
        result = []
        text = text.lower()
        
        for char in text:
            if char == ' ':
                result.append('|')
            elif char in dictionary:
                result.append(char)
            # 跳过不在字典中的字符（标点等）
        
        return ''.join(result)
    
    def _get_trellis(self, emission, tokens, blank_id=0):
        """构建 trellis 矩阵"""
        num_frame = emission.size(0)
        num_tokens = len(tokens)
        
        trellis = torch.zeros((num_frame, num_tokens))
        trellis[1:, 0] = torch.cumsum(emission[1:, blank_id], 0)
        trellis[0, 1:] = -float("inf")
        trellis[-num_tokens + 1:, 0] = float("inf")
        
        for t in range(num_frame - 1):
            trellis[t + 1, 1:] = torch.maximum(
                trellis[t, 1:] + emission[t, blank_id],
                trellis[t, :-1] + emission[t, tokens[1:]],
            )
        return trellis
    
    def _backtrack(self, trellis, emission, tokens, blank_id=0):
        """回溯找到最佳路径"""
        t, j = trellis.size(0) - 1, trellis.size(1) - 1
        
        path = [{'token_index': j, 'time_index': t, 'score': emission[t, blank_id].exp().item()}]
        
        while j > 0:
            if t <= 0:
                return None
            
            p_stay = emission[t - 1, blank_id]
            p_change = emission[t - 1, tokens[j]]
            
            stayed = trellis[t - 1, j] + p_stay
            changed = trellis[t - 1, j - 1] + p_change
            
            t -= 1
            if changed > stayed:
                j -= 1
            
            prob = (p_change if changed > stayed else p_stay).exp().item()
            path.append({'token_index': j, 'time_index': t, 'score': prob})
        
        while t > 0:
            prob = emission[t - 1, blank_id].exp().item()
            path.append({'token_index': j, 'time_index': t - 1, 'score': prob})
            t -= 1
        
        return path[::-1]
    
    def _merge_repeats(self, path, transcript):
        """合并重复的字符"""
        segments = []
        i1, i2 = 0, 0
        
        while i1 < len(path):
            while i2 < len(path) and path[i1]['token_index'] == path[i2]['token_index']:
                i2 += 1
            
            score = sum(p['score'] for p in path[i1:i2]) / (i2 - i1)
            
            segments.append({
                'label': transcript[path[i1]['token_index']],
                'start': path[i1]['time_index'],
                'end': path[i2 - 1]['time_index'] + 1,
                'score': score
            })
            
            i1 = i2
        
        return segments
    
    def _chars_to_words(self, char_segments, original_text: str, num_frames: int, sample_rate: int) -> list:
        """将字符级时间戳转换为词级时间戳"""
        # 计算时间比例
        ratio = len(original_text) / num_frames if num_frames > 0 else 1
        
        # 分词
        words = original_text.split()
        word_segments = []
        
        char_idx = 0
        for word in words:
            word_lower = word.lower()
            
            # 找到这个词对应的字符段
            word_start = None
            word_end = None
            word_score = []
            
            for char in word_lower:
                if char in 'abcdefghijklmnopqrstuvwxyz':
                    # 在 char_segments 中查找
                    while char_idx < len(char_segments):
                        seg = char_segments[char_idx]
                        if seg['label'] == char:
                            if word_start is None:
                                word_start = seg['start']
                            word_end = seg['end']
                            word_score.append(seg['score'])
                            char_idx += 1
                            break
                        elif seg['label'] == '|':
                            char_idx += 1
                        else:
                            char_idx += 1
            
            # 转换为秒
            if word_start is not None and word_end is not None:
                # 每帧约 20ms (50 fps)
                frame_duration = 0.02
                word_segments.append({
                    'word': word,
                    'start': word_start * frame_duration,
                    'end': word_end * frame_duration,
                    'score': sum(word_score) / len(word_score) if word_score else 0.5
                })
            else:
                # 如果找不到，使用估算
                if word_segments:
                    last_end = word_segments[-1]['end']
                else:
                    last_end = 0
                
                word_segments.append({
                    'word': word,
                    'start': last_end,
                    'end': last_end + 0.3,
                    'score': 0.3
                })
        
        return word_segments
    
    def extract_word_timestamps(self, audio_path: str) -> list:
        """提取词级时间戳 - 使用 Whisper（旧方法，保留兼容）"""
        print("🎤 提取词级时间戳（Whisper方法）...")
        
        import whisper
        model = whisper.load_model("base")
        result = model.transcribe(
            audio_path, 
            word_timestamps=True, 
            language='en',
            temperature=0.0
        )
        
        word_segments = []
        for segment in result["segments"]:
            if "words" in segment:
                for word_info in segment["words"]:
                    word_segments.append({
                        'word': word_info['word'].strip(),
                        'start': word_info['start'],
                        'end': word_info['end']
                    })
        
        print(f"   ✅ 提取 {len(word_segments)} 个时间戳")
        return word_segments
    
    def load_stories(self, story_num: int) -> tuple:
        """加载原文 - 英文和中文（支持一句一行格式）"""
        print("📝 加载原文...")
        
        # 英文原文
        eng_files = sorted(list(self.english_dir.glob("*.txt")))
        with open(eng_files[story_num - 1], 'r', encoding='utf-8') as f:
            eng_lines = f.readlines()
        
        # 中文翻译
        chi_files = sorted(list(self.chinese_dir.glob("*.txt")))
        with open(chi_files[story_num - 1], 'r', encoding='utf-8') as f:
            chi_lines = f.readlines()
        
        # 提取句子（跳过标题和空行）
        # 格式：标题 + 空行 + 每句一行
        eng_sentences = []
        chi_sentences = []
        
        # 跳过第一行（标题）和第二行（空行）
        for line in eng_lines[2:]:
            line = line.strip()
            if line:  # 跳过空行
                eng_sentences.append(line)
        
        for line in chi_lines[2:]:
            line = line.strip()
            if line:  # 跳过空行
                chi_sentences.append(line)
        
        print(f"   ✅ {len(eng_sentences)} 个英文句子")
        print(f"   ✅ {len(chi_sentences)} 个中文句子")
        
        # 检查句子数量是否匹配
        if len(eng_sentences) != len(chi_sentences):
            print(f"   ⚠️  警告: 中英文句子数量不匹配!")
            print(f"      英文: {len(eng_sentences)}句")
            print(f"      中文: {len(chi_sentences)}句")
        
        return eng_sentences, chi_sentences
    
    def align_sentences_with_forced_alignment(self, word_timestamps: list, eng_sentences: list, chi_sentences: list) -> list:
        """使用 Forced Alignment 结果进行句子对齐
        
        word_timestamps 已经是精确对齐到原文的时间戳，直接按句子分组即可
        """
        print("🎯 使用 Forced Alignment 结果对齐句子...")
        
        # 构建词到句子的映射
        all_original_words = []
        word_to_sentence = []
        
        for i, sent in enumerate(eng_sentences):
            words = sent.split()
            for word in words:
                all_original_words.append(word)
                word_to_sentence.append(i)
        
        print(f"   📊 原文总词数: {len(all_original_words)}")
        print(f"   📊 对齐时间戳数: {len(word_timestamps)}")
        
        # 检查词数是否匹配
        if len(word_timestamps) != len(all_original_words):
            print(f"   ⚠️ 词数不完全匹配，尝试智能对齐...")
            # 如果不匹配，使用模糊匹配
            return self.align_sentences_fuzzy(word_timestamps, eng_sentences, chi_sentences, all_original_words, word_to_sentence)
        
        # 词数匹配，直接分配
        aligned_words = []
        for i, (word, ts) in enumerate(zip(all_original_words, word_timestamps)):
            aligned_words.append({
                'word': word,  # 使用原文的词
                'start': ts['start'],
                'end': ts['end'],
                'score': ts.get('score', 1.0),
                'sentence_idx': word_to_sentence[i]
            })
        
        # 按句子组织
        aligned = []
        for i, eng_sent in enumerate(eng_sentences):
            sentence_words = [w for w in aligned_words if w['sentence_idx'] == i]
            
            if sentence_words:
                start_time = sentence_words[0]['start']
                end_time = sentence_words[-1]['end']
            else:
                start_time = 0
                end_time = 0
            
            chi_text = chi_sentences[i] if i < len(chi_sentences) else ""
            
            aligned.append({
                'index': i + 1,
                'start': start_time,
                'end': end_time,
                'english': eng_sent,
                'chinese': chi_text,
                'words': sentence_words,
                'score': 1.0,
                'word_start_idx': 0,
                'word_end_idx': len(sentence_words) - 1
            })
            
            if i < 5 or i >= len(eng_sentences) - 3:
                print(f"   🎯 {i+1}: {start_time:.2f}s-{end_time:.2f}s, {len(sentence_words)} 词")
        
        print(f"   ✅ {len(aligned)} 个句子对齐完成")
        return aligned
    
    def align_sentences_fuzzy(self, word_timestamps: list, eng_sentences: list, chi_sentences: list, 
                               all_original_words: list, word_to_sentence: list) -> list:
        """模糊对齐 - 当词数不完全匹配时使用"""
        print("   🔄 使用模糊匹配对齐...")
        
        # 使用序列匹配找到最佳对齐
        ts_words = [ts['word'].lower().strip('.,!?;:"\'') for ts in word_timestamps]
        orig_words = [w.lower().strip('.,!?;:"\'') for w in all_original_words]
        
        # 构建对齐映射
        aligned_words = []
        ts_idx = 0
        
        for i, orig_word in enumerate(all_original_words):
            orig_clean = orig_word.lower().strip('.,!?;:"\'')
            
            # 在时间戳中查找匹配
            best_match_idx = ts_idx
            best_score = 0
            
            # 在当前位置附近搜索
            search_range = min(5, len(word_timestamps) - ts_idx)
            for j in range(search_range):
                if ts_idx + j >= len(word_timestamps):
                    break
                ts_clean = ts_words[ts_idx + j]
                
                # 计算相似度
                if orig_clean == ts_clean:
                    score = 1.0
                elif orig_clean in ts_clean or ts_clean in orig_clean:
                    score = 0.8
                else:
                    score = SequenceMatcher(None, orig_clean, ts_clean).ratio()
                
                if score > best_score:
                    best_score = score
                    best_match_idx = ts_idx + j
            
            # 使用找到的时间戳
            if best_match_idx < len(word_timestamps):
                ts = word_timestamps[best_match_idx]
                aligned_words.append({
                    'word': orig_word,
                    'start': ts['start'],
                    'end': ts['end'],
                    'score': best_score,
                    'sentence_idx': word_to_sentence[i]
                })
                ts_idx = best_match_idx + 1
            else:
                # 没有更多时间戳，使用估算
                if aligned_words:
                    last_end = aligned_words[-1]['end']
                    avg_duration = 0.3  # 平均词时长
                    aligned_words.append({
                        'word': orig_word,
                        'start': last_end,
                        'end': last_end + avg_duration,
                        'score': 0.5,
                        'sentence_idx': word_to_sentence[i]
                    })
        
        # 按句子组织
        aligned = []
        for i, eng_sent in enumerate(eng_sentences):
            sentence_words = [w for w in aligned_words if w['sentence_idx'] == i]
            
            if sentence_words:
                start_time = sentence_words[0]['start']
                end_time = sentence_words[-1]['end']
            else:
                start_time = aligned[-1]['end'] if aligned else 0
                end_time = start_time
            
            chi_text = chi_sentences[i] if i < len(chi_sentences) else ""
            
            aligned.append({
                'index': i + 1,
                'start': start_time,
                'end': end_time,
                'english': eng_sent,
                'chinese': chi_text,
                'words': sentence_words,
                'score': sum(w['score'] for w in sentence_words) / len(sentence_words) if sentence_words else 0,
                'word_start_idx': 0,
                'word_end_idx': len(sentence_words) - 1
            })
            
            if i < 5 or i >= len(eng_sentences) - 3:
                avg_score = aligned[-1]['score']
                print(f"   🎯 {i+1}: {start_time:.2f}s-{end_time:.2f}s, {len(sentence_words)} 词, 置信度 {avg_score:.0%}")
        
        print(f"   ✅ {len(aligned)} 个句子对齐完成")
        return aligned
    
    def align_sentences(self, word_timestamps: list, eng_sentences: list, chi_sentences: list) -> list:
        """智能对齐 - 使用Whisper时间戳，动态分配给原文词"""
        print("🎯 智能对齐（动态时间分配）...")
        
        # 提取所有原文词
        all_original_words = []
        word_to_sentence = []
        
        for i, sent in enumerate(eng_sentences):
            words = sent.split()
            for word in words:
                all_original_words.append(word)
                word_to_sentence.append(i)
        
        print(f"   📊 原文总词数: {len(all_original_words)}")
        print(f"   📊 Whisper时间戳数: {len(word_timestamps)}")
        
        # 使用Whisper的总时长，按原文词数分配
        if len(word_timestamps) > 0:
            total_duration = word_timestamps[-1]['end'] - word_timestamps[0]['start']
            avg_duration_per_word = total_duration / len(all_original_words)
            
            print(f"   📊 总时长: {total_duration:.2f}秒")
            print(f"   📊 平均每词: {avg_duration_per_word:.2f}秒")
        
        # 为每个原文词分配时间
        aligned_words = []
        current_time = word_timestamps[0]['start'] if word_timestamps else 0
        
        for i, word in enumerate(all_original_words):
            # 使用平均时长
            start_time = current_time
            end_time = current_time + avg_duration_per_word
            
            # 但如果有对应的Whisper时间戳，优先使用
            if i < len(word_timestamps):
                # 使用Whisper的时间戳作为参考
                whisper_duration = word_timestamps[i]['end'] - word_timestamps[i]['start']
                end_time = start_time + whisper_duration
            
            aligned_words.append({
                'word': word,
                'start': start_time,
                'end': end_time,
                'sentence_idx': word_to_sentence[i]
            })
            
            current_time = end_time
        
        # 按句子重新组织
        aligned = []
        
        for i, eng_sent in enumerate(eng_sentences):
            sentence_words = [w for w in aligned_words if w['sentence_idx'] == i]
            
            if sentence_words:
                start_time = sentence_words[0]['start']
                end_time = sentence_words[-1]['end']
            else:
                start_time = 0
                end_time = 0
            
            chi_text = chi_sentences[i] if i < len(chi_sentences) else ""
            
            aligned.append({
                'index': i + 1,
                'start': start_time,
                'end': end_time,
                'english': eng_sent,
                'chinese': chi_text,
                'words': sentence_words,
                'score': 1.0,
                'word_start_idx': 0,
                'word_end_idx': len(sentence_words) - 1
            })
            
            if i < 5 or i >= len(eng_sentences) - 3:
                print(f"   🎯 {i+1}: {start_time:.2f}s-{end_time:.2f}s, {len(sentence_words)} 词")
        
        print(f"   ✅ {len(aligned)} 个句子对齐完成")
        
        total_assigned = sum(len(s['words']) for s in aligned)
        print(f"   📊 分配了 {total_assigned}/{len(all_original_words)} 个词")
        
        return aligned
        """智能对齐 - 确保所有词都被使用"""
        print("🎯 智能对齐...")
        
        aligned = []
        word_idx = 0
        total_words = len(word_segments)
        total_sentences = len(eng_sentences)
        
        for i, eng_sent in enumerate(eng_sentences):
            # 获取句子的词
            sent_words = eng_sent.split()
            expected_words = len(sent_words)
            
            # 计算剩余的词和句子
            remaining_sentences = total_sentences - i
            remaining_words = max(0, total_words - word_idx)
            
            # 如果词已经用完，但还有句子，给一个空的对齐
            if word_idx >= len(word_segments):
                aligned.append({
                    'index': i + 1,
                    'start': aligned[-1]['end'] if aligned else 0,
                    'end': aligned[-1]['end'] if aligned else 0,
                    'english': eng_sent,
                    'chinese': chi_sentences[i] if i < len(chi_sentences) else "",
                    'words': [],
                    'score': 0,
                    'word_start_idx': len(word_segments) - 1,
                    'word_end_idx': len(word_segments) - 1
                })
                continue
            
            # 从当前位置开始匹配
            best_start = word_idx
            best_end = word_idx
            matched_count = 0
            
            # 尝试匹配句子中的每个词
            current_idx = word_idx
            for sent_word in sent_words:
                if current_idx >= len(word_segments):
                    break
                
                # 清理标点
                sent_word_clean = sent_word.lower().strip('.,!?;:"\'')
                
                # 在接下来的3个词中查找匹配
                found = False
                for look_ahead in range(3):
                    if current_idx + look_ahead >= len(word_segments):
                        break
                    
                    whisper_word = word_segments[current_idx + look_ahead]['word'].lower().strip('.,!?;:"\'')
                    
                    if (sent_word_clean == whisper_word or 
                        sent_word_clean in whisper_word or 
                        whisper_word in sent_word_clean):
                        matched_count += 1
                        current_idx = current_idx + look_ahead + 1
                        best_end = current_idx - 1
                        found = True
                        break
                
                if not found:
                    # 没找到，跳过这个词
                    current_idx += 1
            
            # 如果匹配率太低，使用平均分配策略
            match_rate = matched_count / len(sent_words) if sent_words else 0
            
            if match_rate < 0.3 or best_end < best_start:
                # 按剩余词数平均分配
                if remaining_sentences > 0:
                    allocated_words = max(
                        expected_words,  # 至少分配期望的词数
                        int(remaining_words / remaining_sentences)  # 或者平均分配
                    )
                else:
                    allocated_words = remaining_words
                
                best_end = min(word_idx + allocated_words - 1, len(word_segments) - 1)
            
            # 边界检查
            best_start = max(0, min(best_start, len(word_segments) - 1))
            best_end = max(best_start, min(best_end, len(word_segments) - 1))
            
            # 获取词级详细信息
            sentence_words = []
            for j in range(best_start, best_end + 1):
                if j < len(word_segments):
                    sentence_words.append(word_segments[j])
            
            # 获取对应的中文
            chi_text = chi_sentences[i] if i < len(chi_sentences) else ""
            
            # 计算时间
            if len(sentence_words) > 0:
                start_time = sentence_words[0]['start']
                end_time = sentence_words[-1]['end']
            else:
                start_time = 0
                end_time = 0
            
            aligned.append({
                'index': i + 1,
                'start': start_time,
                'end': end_time,
                'english': eng_sent,
                'chinese': chi_text,
                'words': sentence_words,
                'score': match_rate,
                'word_start_idx': best_start,
                'word_end_idx': best_end
            })
            
            # 更新word_idx
            word_idx = best_end + 1
            
            if i < 5 or i >= len(eng_sentences) - 3:
                print(f"   🎯 {i+1}: {start_time:.2f}s-{end_time:.2f}s, {len(sentence_words)} 词, 匹配 {match_rate*100:.0f}%")
        
        # 检查是否还有剩余的词
        if word_idx < len(word_segments):
            remaining = len(word_segments) - word_idx
            print(f"   ⚠️ 还有 {remaining} 个词未使用，添加到最后一句")
            
            # 把剩余的词添加到最后一句
            if len(aligned) > 0:
                last_sentence = aligned[-1]
                for j in range(word_idx, len(word_segments)):
                    last_sentence['words'].append(word_segments[j])
                
                # 更新结束时间
                if len(last_sentence['words']) > 0:
                    last_sentence['end'] = last_sentence['words'][-1]['end']
                    last_sentence['word_end_idx'] = len(word_segments) - 1
        
        print(f"   ✅ {len(aligned)} 个句子对齐完成")
        return aligned
    
    def create_karaoke_subtitle(self, segment: dict, next_start: float = None) -> list:
        """创建卡拉OK字幕 - 逐词高亮"""
        clips = []
        
        # 计算结束时间
        end_time = segment['end']
        if next_start is not None:
            end_time = max(end_time, min(next_start, end_time + 0.3))
        
        duration = end_time - segment['start']
        
        # 创建英文卡拉OK效果
        eng_clip = self.create_word_highlight_clip(
            segment['english'],
            segment['words'],
            segment['start'],
            duration,
            y_pos=350,
            is_english=True
        )
        
        # 创建中文字幕（固定颜色）- 调低位置避免重叠
        chi_clip = self.create_static_subtitle(
            segment['chinese'],
            segment['start'],
            duration,
            y_pos=620
        )
        
        return [eng_clip, chi_clip]
    
    def create_word_highlight_clip(self, text: str, words: list, start_time: float, 
                                   duration: float, y_pos: int, is_english: bool):
        """创建逐词高亮动画 - 居中对齐 + 智能换行"""
        
        def make_frame(t):
            # 创建透明背景 - 增加高度支持3行
            img = Image.new('RGBA', (1920, 280), (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)
            
            # 加载字体
            font = self.get_system_font(48)
            
            # 文本已经在load_stories中清理过引号了
            text_words = text.split()
            
            # 智能分行 - 最多3行，不拆分单词
            max_width = 1920 - 200  # 左右各100像素边距
            lines = []
            current_line = []
            current_width = 0
            
            for word in text_words[:len(words)]:
                bbox = draw.textbbox((0, 0), word + " ", font=font)
                word_width = bbox[2] - bbox[0]
                
                # 如果当前行放不下，且未超过3行，才换行
                if current_width + word_width > max_width and current_line and len(lines) < 3:
                    lines.append(current_line)
                    current_line = [word]
                    current_width = word_width
                else:
                    current_line.append(word)
                    current_width += word_width
            
            if current_line:
                lines.append(current_line)
            
            # 如果超过3行，智能合并
            if len(lines) > 3:
                all_words = [w for line in lines for w in line]
                third = len(all_words) // 3
                lines = [
                    all_words[:third],
                    all_words[third:third*2],
                    all_words[third*2:]
                ]
            
            # 计算当前时间对应的词索引
            current_time = start_time + t
            current_word_idx = -1
            
            for i, word_info in enumerate(words):
                if current_time >= word_info['start'] and current_time <= word_info['end']:
                    current_word_idx = i
                    break
                elif current_time < word_info['start']:
                    current_word_idx = max(0, i - 1)
                    break
            
            if current_word_idx == -1 and words:
                current_word_idx = len(words) - 1
            
            # 绘制每行文字 - 居中对齐
            line_height = 58
            start_y = (280 - len(lines) * line_height) // 2
            
            word_global_idx = 0
            for line_idx, line_words in enumerate(lines):
                # 计算这一行的总宽度
                line_text = ' '.join(line_words)
                bbox = draw.textbbox((0, 0), line_text, font=font)
                line_width = bbox[2] - bbox[0]
                
                # 居中起始位置
                x_offset = (1920 - line_width) // 2
                y_pos_line = start_y + line_idx * line_height
                
                # 绘制这一行的每个词
                for word in line_words:
                    # 根据是否是当前词选择颜色 - 使用对比色方案
                    colors = self.color_scheme
                    if word_global_idx < current_word_idx:
                        # 已读 - 使用 read 颜色（不太暗，保持可读）
                        color = (*colors.get('read', (180, 180, 180)), 255)
                    elif word_global_idx == current_word_idx:
                        # 正在读 - 使用 highlight 高亮色（非常醒目）
                        color = (*colors.get('highlight', (255, 255, 100)), 255)
                    else:
                        color = (255, 255, 255, 255)  # 未读 - 白色
                    
                    # 描边
                    for dx in [-2, -1, 0, 1, 2]:
                        for dy in [-2, -1, 0, 1, 2]:
                            if abs(dx) + abs(dy) <= 2:
                                draw.text((x_offset + dx, y_pos_line + dy), word, 
                                        font=font, fill=(0, 0, 0, 255))
                    
                    # 主文字
                    draw.text((x_offset, y_pos_line), word, font=font, fill=color)
                    
                    # 更新位置
                    bbox = draw.textbbox((0, 0), word + " ", font=font)
                    x_offset += (bbox[2] - bbox[0])
                    word_global_idx += 1
            
            return np.array(img)
        
        clip = mp.VideoClip(make_frame, duration=duration)
        return clip.with_position(('center', y_pos)).with_start(start_time)
    
    def create_static_subtitle(self, text: str, start_time: float, duration: float, y_pos: int):
        """创建静态字幕 - 居中对齐 + 智能分行"""
        font = self.get_system_font(44)
        
        # 清理文本 - 移除多余的引号
        clean_text = text.replace('"', '').replace('"', '').replace('"', '').strip()
        
        # 创建图像用于测量
        img = Image.new('RGBA', (1920, 200), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # 智能分行 - 考虑左右边距，支持最多3行
        max_width = 1920 - 200  # 左右各100像素边距
        
        # 检查单行是否超宽
        bbox = draw.textbbox((0, 0), clean_text, font=font)
        text_width = bbox[2] - bbox[0]
        
        if text_width <= max_width:
            lines = [clean_text]
        else:
            # 需要分行 - 在标点处优先分割
            # 尝试分成2行
            if len(clean_text) <= 40:
                mid = len(clean_text) // 2
                best_split = mid
                
                # 在中间附近找标点符号
                for i in range(mid - 8, mid + 9):
                    if i > 0 and i < len(clean_text) and clean_text[i] in '，。！？、；':
                        best_split = i + 1
                        break
                
                line1 = clean_text[:best_split].strip()
                line2 = clean_text[best_split:].strip()
                
                # 检查第二行是否还超宽
                bbox2 = draw.textbbox((0, 0), line2, font=font)
                if bbox2[2] - bbox2[0] <= max_width:
                    lines = [line1, line2]
                else:
                    # 需要3行
                    third = len(clean_text) // 3
                    split1 = third
                    split2 = third * 2
                    
                    # 在分割点附近找标点
                    for i in range(split1 - 5, split1 + 6):
                        if i > 0 and i < len(clean_text) and clean_text[i] in '，。！？、；':
                            split1 = i + 1
                            break
                    
                    for i in range(split2 - 5, split2 + 6):
                        if i > split1 and i < len(clean_text) and clean_text[i] in '，。！？、；':
                            split2 = i + 1
                            break
                    
                    lines = [
                        clean_text[:split1].strip(),
                        clean_text[split1:split2].strip(),
                        clean_text[split2:].strip()
                    ]
            else:
                # 长文本直接分3行
                third = len(clean_text) // 3
                lines = [
                    clean_text[:third].strip(),
                    clean_text[third:third*2].strip(),
                    clean_text[third*2:].strip()
                ]
        
        # 绘制字幕 - 居中对齐
        line_height = 55
        start_y = (200 - len(lines) * line_height) // 2
        
        for i, line in enumerate(lines):
            bbox = draw.textbbox((0, 0), line, font=font)
            line_width = bbox[2] - bbox[0]
            
            # 居中位置
            x = (1920 - line_width) // 2
            y = start_y + i * line_height
            
            # 描边
            for dx in [-2, -1, 0, 1, 2]:
                for dy in [-2, -1, 0, 1, 2]:
                    if abs(dx) + abs(dy) <= 2:
                        draw.text((x + dx, y + dy), line, font=font, fill=(0, 0, 0, 255))
            
            # 主文字 - 使用 subtitle 对比色（和主题色形成反差）
            subtitle_color = self.color_scheme.get('subtitle', (255, 215, 100))
            draw.text((x, y), line, font=font, fill=(*subtitle_color, 255))
        
        img_clip = mp.ImageClip(np.array(img), duration=duration)
        return img_clip.with_position(('center', y_pos)).with_start(start_time)
    
    def get_color_scheme(self, story_num: int) -> dict:
        """获取颜色方案 - 每个主题有独特的配色和背景风格
        
        设计原则：
        1. highlight（高亮）- 要非常亮、醒目，用于当前朗读的词
        2. subtitle（字幕）- 用金黄色，最醒目最好看
        3. read（已读）- 稍暗但不要太暗，保持可读性
        4. primary（主色）- 用于频谱条等装饰
        """
        import random
        random.seed(story_num)
        
        # 定义丰富的主题配色方案 - 强调对比度和醒目度
        schemes = [
            {
                'name': '冰蓝极光',
                'primary': (100, 200, 255),      # 主色 - 冰蓝（频谱）
                'highlight': (255, 255, 0),      # 高亮色 - 亮黄（最醒目）
                'subtitle': (255, 215, 0),       # 中文字幕 - 金黄色
                'read': (150, 200, 220),         # 已读 - 淡蓝
                'bg_style': 'aurora',
                'bg_colors': [(10, 20, 50), (20, 40, 80), (30, 60, 100)],
                'star_color': (180, 220, 255),
                'r_base': 50, 'r_range': 205, 'g_base': 100, 'g_range': 155, 'b_base': 255, 'b_range': 0
            },
            {
                'name': '梦幻紫罗兰',
                'primary': (180, 100, 255),      # 主色 - 紫罗兰
                'highlight': (255, 255, 0),      # 高亮色 - 亮黄
                'subtitle': (255, 215, 0),       # 中文字幕 - 金黄色
                'read': (200, 180, 220),         # 已读 - 淡紫
                'bg_style': 'fantasy',
                'bg_colors': [(30, 10, 50), (50, 20, 80), (40, 15, 60)],
                'star_color': (200, 180, 255),
                'r_base': 138, 'r_range': 117, 'g_base': 43, 'g_range': 170, 'b_base': 226, 'b_range': 30
            },
            {
                'name': '翡翠极光',
                'primary': (80, 255, 180),       # 主色 - 翡翠绿
                'highlight': (255, 255, 0),      # 高亮色 - 亮黄
                'subtitle': (255, 215, 0),       # 中文字幕 - 金黄色
                'read': (180, 220, 200),         # 已读 - 淡绿
                'bg_style': 'aurora',
                'bg_colors': [(5, 25, 20), (10, 40, 35), (15, 50, 40)],
                'star_color': (180, 255, 220),
                'r_base': 46, 'r_range': 134, 'g_base': 213, 'g_range': 42, 'b_base': 152, 'b_range': 103
            },
            {
                'name': '烈焰红',
                'primary': (255, 80, 80),        # 主色 - 鲜红（高亮红）
                'highlight': (255, 255, 0),      # 高亮色 - 亮黄（最醒目）
                'subtitle': (255, 215, 0),       # 中文字幕 - 金黄色
                'read': (255, 180, 180),         # 已读 - 淡红
                'bg_style': 'fire',              # 背景风格 - 火焰
                'bg_colors': [(40, 10, 10), (60, 15, 15), (50, 12, 12)],
                'star_color': (255, 200, 150),
                'r_base': 255, 'r_range': 0, 'g_base': 50, 'g_range': 150, 'b_base': 50, 'b_range': 100
            },
            {
                'name': '金色暖阳',
                'primary': (255, 200, 100),      # 主色 - 金色
                'highlight': (255, 255, 255),    # 高亮色 - 纯白（最醒目）
                'subtitle': (255, 215, 0),       # 中文字幕 - 金黄色
                'read': (230, 210, 180),         # 已读 - 淡金
                'bg_style': 'sunset',
                'bg_colors': [(40, 20, 10), (60, 30, 15), (50, 25, 12)],
                'star_color': (255, 220, 180),
                'r_base': 255, 'r_range': 0, 'g_base': 140, 'g_range': 115, 'b_base': 0, 'b_range': 100
            },
            {
                'name': '薰衣草梦',
                'primary': (200, 150, 255),      # 主色 - 薰衣草
                'highlight': (255, 255, 0),      # 高亮色 - 亮黄
                'subtitle': (255, 215, 0),       # 中文字幕 - 金黄色
                'read': (210, 200, 230),         # 已读 - 淡紫
                'bg_style': 'fantasy',
                'bg_colors': [(25, 20, 40), (40, 35, 60), (35, 30, 50)],
                'star_color': (220, 200, 255),
                'r_base': 200, 'r_range': 55, 'g_base': 150, 'g_range': 80, 'b_base': 255, 'b_range': 0
            },
            {
                'name': '海洋深蓝',
                'primary': (80, 150, 255),       # 主色 - 深蓝
                'highlight': (255, 255, 0),      # 高亮色 - 亮黄
                'subtitle': (255, 215, 0),       # 中文字幕 - 金黄色
                'read': (160, 190, 230),         # 已读 - 淡蓝
                'bg_style': 'ocean',
                'bg_colors': [(5, 15, 35), (10, 25, 50), (8, 20, 40)],
                'star_color': (150, 200, 255),
                'r_base': 50, 'r_range': 150, 'g_base': 100, 'g_range': 100, 'b_base': 255, 'b_range': 0
            },
            {
                'name': '森林绿意',
                'primary': (100, 220, 120),      # 主色 - 森林绿
                'highlight': (255, 255, 0),      # 高亮色 - 亮黄
                'subtitle': (255, 215, 0),       # 中文字幕 - 金黄色
                'read': (180, 210, 180),         # 已读 - 淡绿
                'bg_style': 'forest',
                'bg_colors': [(10, 25, 15), (15, 35, 20), (12, 30, 18)],
                'star_color': (180, 230, 190),
                'r_base': 80, 'r_range': 100, 'g_base': 180, 'g_range': 50, 'b_base': 100, 'b_range': 80
            },
        ]
        
        # 第一个故事固定用冰蓝极光
        if story_num == 1:
            return schemes[0]
        else:
            return schemes[(story_num - 1) % len(schemes)]
    
    def create_background(self) -> str:
        """创建主题背景 - 根据配色方案生成独特背景（专业去色带版本）"""
        import random
        
        colors = self.color_scheme
        bg_style = colors.get('bg_style', 'aurora')
        bg_colors = colors.get('bg_colors', [(10, 20, 50), (20, 40, 80), (30, 60, 100)])
        star_color = colors.get('star_color', (200, 200, 255))
        
        # 使用numpy创建超平滑渐变（专业去色带）
        width, height = 1920, 1080
        
        # 根据风格创建不同的背景
        if bg_style == 'aurora':
            # 极光效果 - 简洁渐变
            color_start = np.array(bg_colors[0], dtype=np.float64)
            color_end = np.array(bg_colors[1], dtype=np.float64)
            
        elif bg_style == 'fantasy':
            # 幻境效果 - 柔和的紫色/粉色渐变
            color_start = np.array(bg_colors[0], dtype=np.float64)
            color_end = np.array([
                bg_colors[1][0] * 0.8,
                bg_colors[1][1] * 0.8,
                bg_colors[1][2] * 0.8
            ], dtype=np.float64)
            
        elif bg_style == 'romantic':
            # 浪漫粉色天空
            color_start = np.array(bg_colors[0], dtype=np.float64)
            color_end = np.array([
                bg_colors[0][0] + (bg_colors[1][0] - bg_colors[0][0]) * 0.5,
                bg_colors[0][1] + (bg_colors[1][1] - bg_colors[0][1]) * 0.5,
                bg_colors[0][2] + (bg_colors[1][2] - bg_colors[0][2]) * 0.5
            ], dtype=np.float64)
            
        elif bg_style == 'fire':
            # 火焰效果 - 深红渐变
            color_start = np.array(bg_colors[0], dtype=np.float64)
            color_end = np.array([
                bg_colors[1][0],
                bg_colors[1][1] * 0.5,
                bg_colors[1][2] * 0.3
            ], dtype=np.float64)
            
        elif bg_style == 'sunset':
            # 日落效果
            color_start = np.array(bg_colors[0], dtype=np.float64)
            color_end = np.array([
                bg_colors[1][0],
                bg_colors[1][1] * 0.7,
                bg_colors[1][2] * 0.5
            ], dtype=np.float64)
            
        else:  # ocean, forest 等
            # 默认渐变
            color_start = np.array(bg_colors[0], dtype=np.float64)
            color_end = np.array(bg_colors[1], dtype=np.float64)
        
        # 创建高精度渐变数组
        gradient = np.zeros((height, width, 3), dtype=np.float64)
        
        # 使用超平滑的渐变函数（smootherstep）
        for y in range(height):
            ratio = y / height
            # smootherstep: 6t^5 - 15t^4 + 10t^3
            smooth_ratio = ratio * ratio * ratio * (ratio * (ratio * 6 - 15) + 10)
            color = color_start * (1 - smooth_ratio) + color_end * smooth_ratio
            gradient[y, :] = color
        
        # 添加Bayer矩阵抖动（专业去色带技术）
        bayer_matrix = np.array([
            [ 0, 32,  8, 40,  2, 34, 10, 42],
            [48, 16, 56, 24, 50, 18, 58, 26],
            [12, 44,  4, 36, 14, 46,  6, 38],
            [60, 28, 52, 20, 62, 30, 54, 22],
            [ 3, 35, 11, 43,  1, 33,  9, 41],
            [51, 19, 59, 27, 49, 17, 57, 25],
            [15, 47,  7, 39, 13, 45,  5, 37],
            [63, 31, 55, 23, 61, 29, 53, 21]
        ], dtype=np.float64) / 64.0 - 0.5
        
        # 创建全尺寸的抖动矩阵
        tile_h = height // 8 + 1
        tile_w = width // 8 + 1
        dither = np.tile(bayer_matrix, (tile_h, tile_w))[:height, :width]
        dither = dither[:, :, np.newaxis]
        dither = np.repeat(dither, 3, axis=2)
        
        # 应用抖动
        gradient_dithered = gradient + dither * 4.0
        
        # 转换为uint8
        gradient_uint8 = np.clip(gradient_dithered, 0, 255).astype(np.uint8)
        
        # 非常轻微的高斯模糊
        gradient_final = cv2.GaussianBlur(gradient_uint8, (3, 3), 0.3)
        
        # 转换为PIL图像
        img = Image.fromarray(gradient_final, mode='RGB')
        draw = ImageDraw.Draw(img)
        
        # 添加主题色星光粒子
        random.seed(42)
        num_stars = 100
        
        for _ in range(num_stars):
            x = random.randint(0, 1920)
            y = random.randint(0, 800)
            size = random.choice([1, 1, 1, 2, 2, 3])
            brightness = random.uniform(0.6, 1.0)
            
            # 使用主题星星颜色
            sr = int(star_color[0] * brightness)
            sg = int(star_color[1] * brightness)
            sb = int(star_color[2] * brightness)
            
            if size == 1:
                draw.point((x, y), fill=(sr, sg, sb))
            elif size == 2:
                # 十字星
                for dx, dy in [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]:
                    if 0 <= x + dx < 1920 and 0 <= y + dy < 1080:
                        draw.point((x + dx, y + dy), fill=(sr, sg, sb))
            else:
                # 大星星
                for dx in range(-2, 3):
                    for dy in range(-2, 3):
                        if abs(dx) + abs(dy) <= 2:
                            fade = 1 - (abs(dx) + abs(dy)) * 0.2
                            if 0 <= x + dx < 1920 and 0 <= y + dy < 1080:
                                draw.point((x + dx, y + dy), fill=(
                                    int(sr * fade), int(sg * fade), int(sb * fade)
                                ))
        
        bg_path = self.temp_dir / "karaoke_bg.png"
        img.save(bg_path)
        return str(bg_path)
    
    def create_visualizer(self, audio_path: str, bg_path: str, duration: float):
        """创建音频可视化 + 动态星光"""
        y, sr = librosa.load(audio_path)
        stft = librosa.stft(y, hop_length=256, n_fft=1024)
        magnitude = np.abs(stft)
        db = librosa.amplitude_to_db(magnitude, ref=np.max)
        times = librosa.times_like(stft, sr=sr, hop_length=256)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=1024)
        
        freq_mask = (freqs >= 80) & (freqs <= 4000)
        db_filtered = db[freq_mask]
        
        n_bars = 45
        db_bars = np.zeros((n_bars, db.shape[1]))
        for i in range(n_bars):
            start = i * len(db_filtered) // n_bars
            end = (i + 1) * len(db_filtered) // n_bars
            db_bars[i] = np.mean(db_filtered[start:end], axis=0)
        
        # 生成动态星星（会闪烁）
        import random
        random.seed(123)
        stars = []
        for _ in range(60):  # 60个动态星星
            stars.append({
                'x': random.randint(0, 1920),
                'y': random.randint(0, 700),
                'speed': random.uniform(0.5, 2.0),  # 闪烁速度
                'phase': random.uniform(0, 6.28),  # 初始相位
                'size': random.choice([1, 2, 3])
            })
        
        # 获取颜色方案
        colors = self.color_scheme
        star_color = colors.get('star_color', (200, 200, 255))
        
        def make_frame(t):
            bg = cv2.imread(bg_path)
            bg = cv2.resize(bg, (1920, 1080))
            
            # 绘制动态星光 - 使用主题星星颜色
            for star in stars:
                brightness = 0.5 + 0.5 * np.sin(star['speed'] * t + star['phase'])
                brightness = max(0.4, min(1.0, brightness))
                
                x, y = int(star['x']), int(star['y'])
                size = star['size']
                
                # 使用主题星星颜色
                sr = int(star_color[0] * brightness)
                sg = int(star_color[1] * brightness)
                sb = int(star_color[2] * brightness)
                
                if size == 1:
                    if 0 <= x < 1920 and 0 <= y < 1080:
                        bg[y, x] = [sb, sg, sr]  # BGR格式
                elif size == 2:
                    for dx, dy in [(0,0), (1,0), (-1,0), (0,1), (0,-1)]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < 1920 and 0 <= ny < 1080:
                            bg[ny, nx] = [sb, sg, sr]
                else:
                    for dx in range(-2, 3):
                        for dy in range(-2, 3):
                            if abs(dx) + abs(dy) <= 2:
                                nx, ny = x + dx, y + dy
                                if 0 <= nx < 1920 and 0 <= ny < 1080:
                                    fade = 1 - (abs(dx) + abs(dy)) * 0.15
                                    bg[ny, nx] = [int(sb * fade), int(sg * fade), int(sr * fade)]
            
            idx = np.argmin(np.abs(times - t))
            spectrum = np.clip((db_bars[:, idx] + 60) / 60, 0, 1)
            
            # 绘制频谱 - 使用配色方案
            h, w = bg.shape[:2]
            bar_w = 18
            spacing = 4
            total_w = n_bars * (bar_w + spacing)
            start_x = (w - total_w) // 2
            
            for i, amp in enumerate(spectrum):
                x = start_x + i * (bar_w + spacing)
                bar_h = int(amp * 220)
                bar_h = max(3, bar_h)
                
                for h_offset in range(bar_h):
                    ratio = h_offset / max(bar_h, 1)
                    r = int(colors['r_base'] + ratio * colors['r_range'])
                    g = int(colors['g_base'] + ratio * colors['g_range'])
                    b = int(colors['b_base'] + ratio * colors['b_range'])
                    
                    y_pos = h - 100 - h_offset
                    cv2.rectangle(bg, (x, y_pos), (x + bar_w, y_pos + 1), (b, g, r), -1)
            
            # 进度条 - 使用配色方案
            bar_y = h - 45
            bar_start = 150
            bar_end = w - 150
            bar_width = bar_end - bar_start
            
            cv2.rectangle(bg, (bar_start, bar_y - 3), (bar_end, bar_y + 3), (30, 30, 40), -1)
            
            progress = t / duration
            prog_x = int(bar_start + bar_width * progress)
            
            for x in range(bar_start, prog_x, 3):
                ratio = (x - bar_start) / bar_width
                r = int(colors['r_base'] + ratio * colors['r_range'])
                g = int(colors['g_base'] + ratio * colors['g_range'])
                b = int(colors['b_base'] + ratio * colors['b_range'])
                cv2.rectangle(bg, (x, bar_y - 3), (x + 3, bar_y + 3), (b, g, r), -1)
            
            cv2.circle(bg, (prog_x, bar_y), 7, (255, 255, 255), -1)
            cv2.circle(bg, (prog_x, bar_y), 9, (255, 220, 100), 2)
            
            return cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)
        
        return mp.VideoClip(make_frame, duration=duration)
    
    def generate(self, story_num: int = 1, use_forced_alignment: bool = True):
        """生成卡拉OK视频 - 使用 Forced Alignment 精确对齐
        
        Args:
            story_num: 故事编号
            use_forced_alignment: 是否使用 WhisperX Forced Alignment（推荐True）
        """
        # 设置颜色方案
        self.color_scheme = self.get_color_scheme(story_num)
        
        print(f"\n{'='*60}")
        print(f"🎤 生成故事 {story_num} - 卡拉OK高亮版本")
        print(f"{'='*60}\n")
        
        print("特点:")
        print("  🎤 词级时间戳 - 最精确")
        print("  🌈 逐词高亮 - 卡拉OK效果")
        print("  🎯 Forced Alignment - 解决数字/缩写识别问题" if use_forced_alignment else "  🎯 文本对齐 - 中英文1:1对应")
        print("  ✅ 真实中文 - Chinese_Stories目录")
        print(f"  🎨 配色方案 - {self.color_scheme['name']}")
        print("  ⭐ 星光特效 - 动态粒子")
        print()
        
        # 加载音频
        audio_files = sorted(list(self.audio_dir.glob("*.wav")))
        audio_path = str(audio_files[story_num - 1])
        
        # 加载原文（英文和中文）
        eng_sentences, chi_sentences = self.load_stories(story_num)
        
        # 获取完整英文文本（用于 Forced Alignment）
        full_english_text = ' '.join(eng_sentences)
        
        if use_forced_alignment:
            # 使用 WhisperX Forced Alignment 提取精确时间戳
            word_timestamps = self.extract_word_timestamps_with_forced_alignment(audio_path, full_english_text)
            # 使用新的对齐方法
            aligned = self.align_sentences_with_forced_alignment(word_timestamps, eng_sentences, chi_sentences)
        else:
            # 使用旧方法（保留兼容）
            word_timestamps = self.extract_word_timestamps(audio_path)
            aligned = self.align_sentences(word_timestamps, eng_sentences, chi_sentences)
        
        # 创建字幕
        print("\n🎬 创建卡拉OK字幕...")
        subtitle_clips = []
        for i, segment in enumerate(aligned):
            next_start = aligned[i + 1]['start'] if i + 1 < len(aligned) else None
            clips = self.create_karaoke_subtitle(segment, next_start)
            subtitle_clips.extend(clips)
            if i < 3:
                print(f"   🎤 句子 {i+1}: {segment['start']:.2f}s-{segment['end']:.2f}s")
        
        if len(aligned) > 3:
            print(f"   ... 还有 {len(aligned) - 3} 个")
        
        print(f"   ✅ {len(subtitle_clips)} 个字幕片段")
        
        # 创建可视化
        print("\n🎵 创建可视化...")
        bg_path = self.create_background()
        
        # 使用librosa获取时长
        import librosa
        y, sr = librosa.load(audio_path, sr=None)
        duration = len(y) / sr
        
        visualizer = self.create_visualizer(audio_path, bg_path, duration)
        
        # 合成
        print("\n🎬 合成最终视频...")
        final = mp.CompositeVideoClip([visualizer] + subtitle_clips)
        final = final.with_audio(mp.AudioFileClip(audio_path))
        
        output_path = self.output_dir / f"Story_{story_num:02d}_Karaoke_Complete.mp4"
        
        print(f"\n⚡ 使用无损编码（完全无色带，文件较大）...")
        final.write_videofile(
            str(output_path),
            fps=30,
            codec='libx264',
            audio_codec='aac',
            preset='veryslow',
            threads=8,
            ffmpeg_params=[
                '-crf', '0',  # CRF 0 = 完全无损
                '-pix_fmt', 'yuv444p',  # 4:4:4色度采样，无色度压缩
                '-qp', '0',  # 量化参数0 = 无损
                '-movflags', '+faststart'
            ]
        )
        
        final.close()
        
        size = os.path.getsize(output_path) / (1024 * 1024)
        
        print(f"\n{'='*60}")
        print("✅ 卡拉OK视频生成成功!")
        print(f"{'='*60}")
        print(f"📹 文件: {output_path}")
        print(f"📊 大小: {size:.1f} MB")
        print(f"⏱️ 时长: {duration:.1f}秒")
        print(f"🎤 特点: 逐词高亮 + 真实中文")
        
        return str(output_path)


def main():
    import sys
    
    print("="*60)
    print("🎤 卡拉OK对齐字幕生成系统")
    print("="*60)
    print()
    print("新功能:")
    print("  🌈 逐词高亮 - 卡拉OK跟踪效果")
    print("  🎯 超高精度 - 改进对齐算法")
    print("  ✅ 真实中文 - 使用Chinese_Stories")
    print("  🎵 专业可视化 - 45个频谱条")
    print()
    
    # 检查命令行参数
    if len(sys.argv) > 1:
        if sys.argv[1] == "all":
            # 批量生成所有故事
            print("📦 批量生成模式 - 生成所有故事视频")
            print()
            
            generator = KaraokeAlignmentGenerator()
            audio_dir = Path("Stiries_audio")
            audio_files = sorted(list(audio_dir.glob("*.wav")))
            total = len(audio_files)
            
            print(f"📊 发现 {total} 个音频文件")
            print(f"⏰ 预计总时长: {total * 8:.0f} 分钟")
            print()
            
            success_count = 0
            failed = []
            
            for i in range(1, total + 1):
                try:
                    print(f"\n{'='*60}")
                    print(f"🎬 [{i}/{total}] 正在生成故事 {i}")
                    print(f"{'='*60}")
                    
                    video_path = generator.generate(story_num=i)
                    
                    if video_path:
                        success_count += 1
                        print(f"✅ 故事 {i} 完成 ({success_count}/{total})")
                    else:
                        failed.append(i)
                        print(f"❌ 故事 {i} 失败")
                        
                except Exception as e:
                    failed.append(i)
                    print(f"❌ 故事 {i} 发生错误: {str(e)}")
                    continue
            
            # 汇总报告
            print(f"\n{'='*60}")
            print("📊 批量生成完成报告")
            print(f"{'='*60}")
            print(f"✅ 成功: {success_count}/{total}")
            if failed:
                print(f"❌ 失败: {len(failed)} 个 - {failed}")
            print(f"💾 输出目录: karaoke_alignment_videos/")
            print()
            
        else:
            # 生成指定编号的故事
            try:
                story_num = int(sys.argv[1])
                generator = KaraokeAlignmentGenerator()
                video_path = generator.generate(story_num=story_num)
                
                if video_path:
                    print(f"\n🎬 正在打开视频...")
                    try:
                        os.startfile(video_path)
                    except:
                        print(f"请手动打开: {video_path}")
                    
                    print("\n🎉 卡拉OK视频生成完成！")
                    print("💡 逐词高亮 + 真实中文 + 超精确对齐！")
            except ValueError:
                print("❌ 错误: 请提供有效的故事编号")
                print("用法: python karaoke_alignment_generator.py [编号|all]")
    else:
        # 默认生成第1个故事
        generator = KaraokeAlignmentGenerator()
        video_path = generator.generate(story_num=1)
        
        if video_path:
            print(f"\n🎬 正在打开视频...")
            try:
                os.startfile(video_path)
            except:
                print(f"请手动打开: {video_path}")
            
            print("\n🎉 卡拉OK视频生成完成！")
            print("💡 逐词高亮 + 真实中文 + 超精确对齐！")


if __name__ == "__main__":
    main()
