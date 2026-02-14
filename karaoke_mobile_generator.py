#!/usr/bin/env python3
"""
卡拉OK对齐字幕生成器 - 手机竖屏版 V3 (简约风格)
专为快手/抖音设计 - 左侧布局，避开右侧功能按钮
- 竖屏尺寸: 1080x1920
- 简约风格：左中高光，向上滚动
- 固定高光位置，字幕流动
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

# PyTorch for forced alignment
import torch
import torchaudio


class KaraokeAlignmentGeneratorMobileV3Simple:
    """卡拉OK对齐字幕生成器 - 手机竖屏版 V3 (简约风格)"""
    
    def __init__(self):
        self.output_dir = Path("karaoke_alignment_videos_mobile")
        self.output_dir.mkdir(exist_ok=True)
        
        self.temp_dir = Path("temp_karaoke_alignment_mobile")
        self.temp_dir.mkdir(exist_ok=True)
        
        self.audio_dir = Path("Stories_audio")
        self.english_dir = Path("English_Stories")
        self.chinese_dir = Path("Chinese_Stories")
        
        # 竖屏视频尺寸 (9:16)
        self.width = 1080
        self.height = 1920
        
        # 左侧布局 - 避开右侧按钮
        self.left_margin = 60  # 左边距
        self.right_safe_zone = 200  # 右侧安全区（避开按钮）
        self.text_width = self.width - self.left_margin - self.right_safe_zone  # 可用文本宽度
        
        # 高光固定位置（左中）
        self.highlight_y = 960  # 屏幕中心高度
        
        # 句子间距（增大间距，避免重叠）
        self.line_spacing = 200  # 每个句子对的间距（英文+中文+间隔）
        
        # 设备配置
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print("📱 卡拉OK对齐字幕生成器 - 手机竖屏版 V3 (简约风格)")
        print(f"   视频尺寸: {self.width}x{self.height} (9:16)")
        print(f"   布局: 左侧布局，避开右侧按钮")
        print(f"   风格: 简约风格，向上滚动")
        print(f"   设备: {self.device}")


    
    def extract_word_timestamps_with_forced_alignment(self, audio_path: str, english_text: str) -> list:
        """使用 torchaudio Forced Alignment 提取精确词级时间戳"""
        print("🎤 使用 torchaudio Forced Alignment 提取词级时间戳...")
        
        bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
        model = bundle.get_model().to(self.device)
        labels = bundle.get_labels()
        dictionary = {c.lower(): i for i, c in enumerate(labels)}
        
        import soundfile as sf
        audio_data, sample_rate = sf.read(audio_path)
        waveform = torch.tensor(audio_data).float()
        if len(waveform.shape) == 1:
            waveform = waveform.unsqueeze(0)
        elif waveform.shape[1] == 2:
            waveform = waveform.mean(dim=1, keepdim=True).T
        
        if sample_rate != bundle.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)
        
        waveform = waveform.to(self.device)
        
        with torch.inference_mode():
            emissions, _ = model(waveform)
            emissions = torch.log_softmax(emissions, dim=-1)
        
        emission = emissions[0].cpu().detach()
        transcript = self._prepare_transcript(english_text, dictionary)
        tokens = [dictionary.get(c, 0) for c in transcript]
        
        trellis = self._get_trellis(emission, tokens)
        path = self._backtrack(trellis, emission, tokens)
        
        if path is None:
            print("   ⚠️ 强制对齐失败")
            return []
        
        segments = self._merge_repeats(path, transcript)
        word_segments = self._chars_to_words(segments, english_text, emission.shape[0], bundle.sample_rate)
        
        print(f"   ✅ 提取 {len(word_segments)} 个词的精确时间戳")
        
        del model
        if self.device == "cuda":
            torch.cuda.empty_cache()
        
        return word_segments

    
    def _prepare_transcript(self, text: str, dictionary: dict) -> str:
        """准备用于对齐的转录文本"""
        result = []
        text = text.lower()
        for char in text:
            if char == ' ':
                result.append('|')
            elif char in dictionary:
                result.append(char)
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
        words = original_text.split()
        word_segments = []
        char_idx = 0
        for word in words:
            word_lower = word.lower()
            word_start = None
            word_end = None
            word_score = []
            for char in word_lower:
                if char in 'abcdefghijklmnopqrstuvwxyz':
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
            if word_start is not None and word_end is not None:
                frame_duration = 0.02
                word_segments.append({
                    'word': word,
                    'start': word_start * frame_duration,
                    'end': word_end * frame_duration,
                    'score': sum(word_score) / len(word_score) if word_score else 0.5
                })
            else:
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


    
    def load_stories(self, story_num: int) -> tuple:
        """加载原文 - 英文和中文"""
        print("📝 加载原文...")
        
        eng_files = sorted(list(self.english_dir.glob("*.txt")))
        with open(eng_files[story_num - 1], 'r', encoding='utf-8') as f:
            eng_lines = f.readlines()
        
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
        """使用 Forced Alignment 结果进行句子对齐"""
        print("🎯 使用 Forced Alignment 结果对齐句子...")
        
        all_original_words = []
        word_to_sentence = []
        
        for i, sent in enumerate(eng_sentences):
            words = sent.split()
            for word in words:
                all_original_words.append(word)
                word_to_sentence.append(i)
        
        print(f"   📊 原文总词数: {len(all_original_words)}")
        print(f"   📊 对齐时间戳数: {len(word_timestamps)}")
        
        if len(word_timestamps) != len(all_original_words):
            print(f"   ⚠️ 词数不完全匹配，尝试智能对齐...")
            return self.align_sentences_fuzzy(word_timestamps, eng_sentences, chi_sentences, all_original_words, word_to_sentence)
        
        aligned_words = []
        for i, (word, ts) in enumerate(zip(all_original_words, word_timestamps)):
            aligned_words.append({
                'word': word,
                'start': ts['start'],
                'end': ts['end'],
                'score': ts.get('score', 1.0),
                'sentence_idx': word_to_sentence[i]
            })
        
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
        
        print(f"   ✅ {len(aligned)} 个句子对齐完成")
        return aligned

    
    def align_sentences_fuzzy(self, word_timestamps: list, eng_sentences: list, chi_sentences: list, 
                               all_original_words: list, word_to_sentence: list) -> list:
        """模糊对齐"""
        print("   🔄 使用模糊匹配对齐...")
        
        ts_words = [ts['word'].lower().strip('.,!?;:"\'') for ts in word_timestamps]
        aligned_words = []
        ts_idx = 0
        
        for i, orig_word in enumerate(all_original_words):
            orig_clean = orig_word.lower().strip('.,!?;:"\'')
            best_match_idx = ts_idx
            best_score = 0
            search_range = min(5, len(word_timestamps) - ts_idx)
            for j in range(search_range):
                if ts_idx + j >= len(word_timestamps):
                    break
                ts_clean = ts_words[ts_idx + j]
                if orig_clean == ts_clean:
                    score = 1.0
                elif orig_clean in ts_clean or ts_clean in orig_clean:
                    score = 0.8
                else:
                    score = SequenceMatcher(None, orig_clean, ts_clean).ratio()
                if score > best_score:
                    best_score = score
                    best_match_idx = ts_idx + j
            
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
                if aligned_words:
                    last_end = aligned_words[-1]['end']
                    aligned_words.append({
                        'word': orig_word,
                        'start': last_end,
                        'end': last_end + 0.3,
                        'score': 0.5,
                        'sentence_idx': word_to_sentence[i]
                    })
        
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
        
        print(f"   ✅ {len(aligned)} 个句子对齐完成")
        return aligned


    
    def _calculate_sentence_height(self, segment, eng_font, chi_font, draw):
        """计算句子的实际高度（包括英文和中文的所有行）"""
        # 智能换行 - 英文（按80%宽度换行）
        max_line_width = int(self.text_width * 0.8)
        text_words = segment['english'].split()
        eng_line_count = 0
        current_line_width = 0
        
        for word in text_words:
            bbox = draw.textbbox((0, 0), word + " ", font=eng_font)
            word_width = bbox[2] - bbox[0]
            
            if current_line_width + word_width > max_line_width and current_line_width > 0:
                eng_line_count += 1
                current_line_width = word_width
            else:
                current_line_width += word_width
        
        if current_line_width > 0:
            eng_line_count += 1
        
        # 智能换行 - 中文（按80%宽度换行）
        chi_text = segment['chinese']
        chi_line_count = 0
        chi_current_line = ""
        
        for char in chi_text:
            test_line = chi_current_line + char
            bbox = draw.textbbox((0, 0), test_line, font=chi_font)
            line_width = bbox[2] - bbox[0]
            
            if line_width > max_line_width and chi_current_line:
                chi_line_count += 1
                chi_current_line = char
            else:
                chi_current_line = test_line
        
        if chi_current_line:
            chi_line_count += 1
        
        # 计算总高度
        eng_height = eng_line_count * 60  # 每行英文60px
        chi_height = chi_line_count * 55  # 每行中文55px
        gap = 10  # 英文和中文之间的间距
        
        total_height = eng_height + gap + chi_height
        return total_height
    
    def create_simple_subtitle_clip(self, aligned_segments: list, total_duration: float):
        """创建简约风格字幕 - 左侧布局，向上滚动，动态间距（优化版）"""
        
        # 预加载字体（只加载一次，避免每帧重复加载）
        print("   ⚡ 预加载字体...")
        try:
            eng_font = ImageFont.truetype("C:\\Windows\\Fonts\\arialbd.ttf", 52)
            chi_font = ImageFont.truetype("C:\\Windows\\Fonts\\simhei.ttf", 46)
        except:
            try:
                eng_font = ImageFont.truetype("C:\\Windows\\Fonts\\arial.ttf", 52)
                chi_font = ImageFont.truetype("C:\\Windows\\Fonts\\simsun.ttc", 46)
            except:
                eng_font = ImageFont.truetype("C:\\Windows\\Fonts\\simsun.ttc", 52)
                chi_font = eng_font
        
        # 预计算每个句子的高度
        print("   ⚡ 预计算句子高度...")
        temp_img = Image.new('RGBA', (self.width, self.height), (0, 0, 0, 0))
        temp_draw = ImageDraw.Draw(temp_img)
        
        for seg in aligned_segments:
            seg['height'] = self._calculate_sentence_height(seg, eng_font, chi_font, temp_draw)
        
        # 预计算所有句子的Y位置（避免每帧重复计算）
        print("   ⚡ 预计算句子位置...")
        all_y_positions = {}
        for current_idx in range(len(aligned_segments)):
            y_positions = {}
            cumulative_y = self.highlight_y
            
            # 从当前句子开始，向上计算
            for i in range(current_idx, -1, -1):
                y_positions[i] = cumulative_y
                if i > 0:
                    cumulative_y -= (aligned_segments[i-1]['height'] + 80)
            
            # 从当前句子开始，向下计算
            cumulative_y = self.highlight_y
            for i in range(current_idx, len(aligned_segments)):
                y_positions[i] = cumulative_y
                if i < len(aligned_segments) - 1:
                    cumulative_y += (aligned_segments[i]['height'] + 80)
            
            all_y_positions[current_idx] = y_positions
        
        print("   ✅ 预计算完成，开始渲染...")
        
        def make_frame(t):
            # 创建透明背景
            img = Image.new('RGBA', (self.width, self.height), (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)
            
            # 找到当前播放的句子
            current_idx = -1
            for i, seg in enumerate(aligned_segments):
                if t >= seg['start'] and t <= seg['end']:
                    current_idx = i
                    break
            
            if current_idx == -1:
                for i, seg in enumerate(aligned_segments):
                    if t < seg['start']:
                        current_idx = max(0, i - 1)
                        break
                if current_idx == -1:
                    current_idx = len(aligned_segments) - 1
            
            # 计算滚动偏移
            scroll_offset = 0
            if current_idx < len(aligned_segments):
                seg = aligned_segments[current_idx]
                
                if t > seg['end'] and current_idx + 1 < len(aligned_segments):
                    next_seg = aligned_segments[current_idx + 1]
                    gap_duration = next_seg['start'] - seg['end']
                    next_sentence_height = seg['height'] + 80
                    
                    if gap_duration > 0:
                        time_in_gap = t - seg['end']
                        progress = min(1.0, time_in_gap / gap_duration)
                        if progress < 0.5:
                            eased_progress = 4 * progress * progress * progress
                        else:
                            eased_progress = 1 - pow(-2 * progress + 2, 3) / 2
                        scroll_offset = eased_progress * next_sentence_height
                    else:
                        scroll_offset = next_sentence_height
            
            # 使用预计算的位置
            y_positions = all_y_positions.get(current_idx, {})
            visible_range = range(max(0, current_idx - 2), min(len(aligned_segments), current_idx + 3))
            
            for i in visible_range:
                segment = aligned_segments[i]
                y_pos = y_positions.get(i, self.highlight_y) - scroll_offset
                
                if y_pos < -300 or y_pos > self.height + 300:
                    continue
                
                is_highlight = (i == current_idx) and (t >= segment['start'] and t <= segment['end'])
                
                if is_highlight:
                    self._draw_sentence_with_karaoke(draw, segment, t, y_pos, eng_font, chi_font)
                else:
                    opacity = 0.5 if i != current_idx else 0.7
                    self._draw_sentence_static(draw, segment, y_pos, eng_font, chi_font, opacity)
            
            return np.array(img)
        
        return mp.VideoClip(make_frame, duration=total_duration)
    
    def _draw_sentence_with_karaoke(self, draw, segment, current_time, y_pos, eng_font, chi_font):
        """绘制带卡拉OK高亮的句子（左侧对齐，智能换行，使用主题颜色）"""
        # 找到当前高亮的词
        current_word_idx = -1
        words = segment['words']
        for i, word_info in enumerate(words):
            if current_time >= word_info['start'] and current_time <= word_info['end']:
                current_word_idx = i
                break
        
        # 使用主题颜色
        text_color = self.color_scheme['text']
        highlight_color = self.color_scheme['highlight']
        
        # 智能换行 - 英文（按80%宽度换行）
        max_line_width = int(self.text_width * 0.8)  # 80%宽度
        text_words = segment['english'].split()
        eng_lines = []
        current_line = []
        current_line_width = 0
        word_indices = []  # 记录每行的词索引
        
        for word_idx, word in enumerate(text_words[:len(words)]):
            bbox = draw.textbbox((0, 0), word + " ", font=eng_font)
            word_width = bbox[2] - bbox[0]
            
            if current_line_width + word_width > max_line_width and current_line:
                eng_lines.append(current_line)
                word_indices.append(list(range(len(word_indices) * 10, len(word_indices) * 10 + len(current_line))))
                current_line = [word]
                current_line_width = word_width
            else:
                current_line.append(word)
                current_line_width += word_width
        
        if current_line:
            eng_lines.append(current_line)
        
        # 绘制英文（逐词高亮，多行）
        y_offset = int(y_pos)
        word_idx = 0
        
        for line_words in eng_lines:
            x_offset = self.left_margin
            
            for word in line_words:
                if word_idx == current_word_idx:
                    # 高亮词 - 使用主题高亮色
                    color = highlight_color
                else:
                    # 普通词 - 使用主题文字色
                    color = text_color
                
                draw.text((x_offset, y_offset), word, font=eng_font, fill=color)
                bbox = draw.textbbox((0, 0), word + " ", font=eng_font)
                word_width = bbox[2] - bbox[0]
                x_offset += word_width
                word_idx += 1
            
            y_offset += 60  # 行间距
        
        # 英文和中文之间的间距
        y_offset += 10
        
        # 智能换行 - 中文（按80%宽度换行）
        chi_text = segment['chinese']
        chi_lines = []
        chi_current_line = ""
        
        for char in chi_text:
            test_line = chi_current_line + char
            bbox = draw.textbbox((0, 0), test_line, font=chi_font)
            line_width = bbox[2] - bbox[0]
            
            if line_width > max_line_width and chi_current_line:
                chi_lines.append(chi_current_line)
                chi_current_line = char
            else:
                chi_current_line = test_line
        
        if chi_current_line:
            chi_lines.append(chi_current_line)
        
        # 绘制中文（多行，使用主题文字色）
        for chi_line in chi_lines:
            draw.text((self.left_margin, y_offset), chi_line, font=chi_font, fill=text_color)
            y_offset += 55  # 行间距
    
    def _draw_sentence_static(self, draw, segment, y_pos, eng_font, chi_font, opacity):
        """绘制静态句子（左侧对齐，半透明，智能换行，使用主题颜色）"""
        # 使用主题文字颜色
        color = self.color_scheme['text']
        alpha = int(255 * opacity)
        color_with_alpha = (*color, alpha)
        
        # 智能换行 - 英文（按80%宽度换行）
        max_line_width = int(self.text_width * 0.8)  # 80%宽度
        text_words = segment['english'].split()
        eng_lines = []
        current_line = []
        current_line_width = 0
        
        for word in text_words:
            bbox = draw.textbbox((0, 0), word + " ", font=eng_font)
            word_width = bbox[2] - bbox[0]
            
            if current_line_width + word_width > max_line_width and current_line:
                eng_lines.append(' '.join(current_line))
                current_line = [word]
                current_line_width = word_width
            else:
                current_line.append(word)
                current_line_width += word_width
        
        if current_line:
            eng_lines.append(' '.join(current_line))
        
        # 绘制英文（多行）
        y_offset = int(y_pos)
        for line in eng_lines:
            draw.text((self.left_margin, y_offset), line, font=eng_font, fill=color_with_alpha)
            y_offset += 60  # 行间距
        
        # 英文和中文之间的间距
        y_offset += 10
        
        # 智能换行 - 中文（按80%宽度换行）
        chi_text = segment['chinese']
        chi_lines = []
        chi_current_line = ""
        
        for char in chi_text:
            test_line = chi_current_line + char
            bbox = draw.textbbox((0, 0), test_line, font=chi_font)
            line_width = bbox[2] - bbox[0]
            
            if line_width > max_line_width and chi_current_line:
                chi_lines.append(chi_current_line)
                chi_current_line = char
            else:
                chi_current_line = test_line
        
        if chi_current_line:
            chi_lines.append(chi_current_line)
        
        # 绘制中文（多行）
        for chi_line in chi_lines:
            draw.text((self.left_margin, y_offset), chi_line, font=chi_font, fill=color_with_alpha)
            y_offset += 55  # 行间距


    
    def create_simple_background(self, duration: float) -> mp.VideoClip:
        """创建简约背景 - 使用主题颜色（专业去色带版本）"""
        print("🎨 创建简约背景（专业去色带）...")
        
        import random
        
        # 使用numpy创建超平滑渐变
        color_start = np.array(self.color_scheme['bg_start'], dtype=np.float64)
        color_end = np.array(self.color_scheme['bg_end'], dtype=np.float64)
        
        # 创建高精度渐变数组
        gradient = np.zeros((self.height, self.width, 3), dtype=np.float64)
        
        # 使用超平滑的渐变函数（smootherstep - 比smoothstep更平滑）
        for y in range(self.height):
            ratio = y / self.height
            # smootherstep: 6t^5 - 15t^4 + 10t^3
            smooth_ratio = ratio * ratio * ratio * (ratio * (ratio * 6 - 15) + 10)
            color = color_start * (1 - smooth_ratio) + color_end * smooth_ratio
            gradient[y, :] = color
        
        # 添加Bayer矩阵抖动（专业去色带技术）
        # 8x8 Bayer矩阵
        bayer_matrix = np.array([
            [ 0, 32,  8, 40,  2, 34, 10, 42],
            [48, 16, 56, 24, 50, 18, 58, 26],
            [12, 44,  4, 36, 14, 46,  6, 38],
            [60, 28, 52, 20, 62, 30, 54, 22],
            [ 3, 35, 11, 43,  1, 33,  9, 41],
            [51, 19, 59, 27, 49, 17, 57, 25],
            [15, 47,  7, 39, 13, 45,  5, 37],
            [63, 31, 55, 23, 61, 29, 53, 21]
        ], dtype=np.float64) / 64.0 - 0.5  # 归一化到[-0.5, 0.5]
        
        # 创建全尺寸的抖动矩阵
        tile_h = self.height // 8 + 1
        tile_w = self.width // 8 + 1
        dither = np.tile(bayer_matrix, (tile_h, tile_w))[:self.height, :self.width]
        dither = dither[:, :, np.newaxis]  # 添加颜色通道维度
        dither = np.repeat(dither, 3, axis=2)  # 扩展到3个颜色通道
        
        # 应用抖动（强度为4.0，更强的去色带效果）
        gradient_dithered = gradient + dither * 4.0
        
        # 转换为uint8
        gradient_uint8 = np.clip(gradient_dithered, 0, 255).astype(np.uint8)
        
        # 非常轻微的高斯模糊（0.3sigma，几乎看不出模糊但能柔化抖动）
        gradient_final = cv2.GaussianBlur(gradient_uint8, (3, 3), 0.3)
        
        # 转换为PIL图像并保存（使用最高质量）
        img = Image.fromarray(gradient_final, mode='RGB')
        bg_path = str(self.temp_dir / "bg_simple.png")
        img.save(bg_path, quality=100, optimize=False)
        
        # 小星星粒子（只在左侧区域，使用主题颜色）
        num_stars = 50
        stars = []
        for i in range(num_stars):
            stars.append({
                'x': random.randint(self.left_margin, self.width - self.right_safe_zone),
                'y': random.randint(100, self.height - 100),
                'size': random.choice([1, 1, 2, 2, 3]),
                'speed': random.uniform(0.5, 2.0),
                'phase': random.uniform(0, 2 * np.pi)
            })
        
        def make_frame(t):
            bg = cv2.imread(bg_path)
            bg = cv2.resize(bg, (self.width, self.height))
            
            # 使用主题星光颜色
            star_color = self.color_scheme['star']
            sr, sg, sb = star_color
            
            for star in stars:
                brightness = 0.5 + 0.5 * np.sin(t * star['speed'] * 2 * np.pi + star['phase'])
                x = star['x']
                y = int(star['y'] + 15 * np.sin(t * star['speed'] + star['phase']))
                size = star['size']
                
                if self.left_margin <= x < self.width - self.right_safe_zone and 100 <= y < self.height - 100:
                    if size == 1:
                        bg[y, x] = [int(sb * brightness), int(sg * brightness), int(sr * brightness)]
                    elif size == 2:
                        for dx, dy in [(0,0), (1,0), (-1,0), (0,1), (0,-1)]:
                            nx, ny = x + dx, y + dy
                            if self.left_margin <= nx < self.width - self.right_safe_zone and 100 <= ny < self.height - 100:
                                bg[ny, nx] = [int(sb * brightness), int(sg * brightness), int(sr * brightness)]
                    else:
                        for dx in range(-2, 3):
                            for dy in range(-2, 3):
                                if abs(dx) + abs(dy) <= 2:
                                    nx, ny = x + dx, y + dy
                                    if self.left_margin <= nx < self.width - self.right_safe_zone and 100 <= ny < self.height - 100:
                                        fade = 1 - (abs(dx) + abs(dy)) * 0.15
                                        bg[ny, nx] = [int(sb * brightness * fade), 
                                                     int(sg * brightness * fade), 
                                                     int(sr * brightness * fade)]
            
            return cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)
        
        return mp.VideoClip(make_frame, duration=duration)

    
    def get_color_scheme(self, story_num: int) -> dict:
        """获取颜色方案 - 与电脑版保持一致"""
        import random
        random.seed(story_num)
        
        # 8个主题配色方案（与电脑版完全一致）
        schemes = [
            {
                'name': '冰蓝极光',
                'bg_start': (10, 20, 50),
                'bg_end': (30, 60, 100),
                'text': (220, 220, 255),
                'highlight': (255, 182, 193),
                'star': (180, 220, 255),
            },
            {
                'name': '梦幻紫罗兰',
                'bg_start': (30, 10, 50),
                'bg_end': (50, 20, 80),
                'text': (220, 220, 255),
                'highlight': (255, 182, 193),
                'star': (200, 180, 255),
            },
            {
                'name': '翡翠极光',
                'bg_start': (5, 25, 20),
                'bg_end': (15, 50, 40),
                'text': (220, 255, 240),
                'highlight': (255, 255, 150),
                'star': (180, 255, 220),
            },
            {
                'name': '烈焰红',
                'bg_start': (40, 10, 10),
                'bg_end': (60, 15, 15),
                'text': (255, 220, 220),
                'highlight': (255, 255, 100),
                'star': (255, 200, 150),
            },
            {
                'name': '金色暖阳',
                'bg_start': (40, 20, 10),
                'bg_end': (60, 30, 15),
                'text': (255, 240, 220),
                'highlight': (255, 255, 255),
                'star': (255, 220, 180),
            },
            {
                'name': '薰衣草梦',
                'bg_start': (25, 20, 40),
                'bg_end': (40, 35, 60),
                'text': (240, 230, 255),
                'highlight': (255, 200, 255),
                'star': (220, 200, 255),
            },
            {
                'name': '海洋深蓝',
                'bg_start': (5, 15, 35),
                'bg_end': (10, 25, 50),
                'text': (200, 230, 255),
                'highlight': (150, 255, 255),
                'star': (150, 200, 255),
            },
            {
                'name': '森林绿意',
                'bg_start': (10, 25, 15),
                'bg_end': (15, 35, 20),
                'text': (230, 255, 230),
                'highlight': (255, 255, 150),
                'star': (180, 230, 190),
            },
        ]
        
        # 第一个故事固定用冰蓝极光
        if story_num == 1:
            return schemes[0]
        else:
            return schemes[(story_num - 1) % len(schemes)]
    
    def generate(self, story_num: int = 1, use_forced_alignment: bool = True, theme_override: int = None):
        """生成卡拉OK视频 - 简约风格
        
        Args:
            story_num: 故事编号
            use_forced_alignment: 是否使用强制对齐
            theme_override: 强制使用指定主题 (1-8)，None则按story_num自动选择
        """
        print(f"\n{'='*60}")
        print(f"📱 生成故事 {story_num} - 手机竖屏版 V3 (简约风格)")
        print(f"{'='*60}\n")
        
        # 设置颜色方案
        if theme_override is not None:
            self.color_scheme = self.get_color_scheme(theme_override)
        else:
            self.color_scheme = self.get_color_scheme(story_num)
        
        print("特点:")
        print("  📱 竖屏格式 - 1080x1920 (9:16)")
        print("  📍 左侧布局 - 避开右侧按钮")
        print("  ⬆️ 向上滚动 - 高光固定在左中")
        print("  🎨 简约风格 - 紫色背景 + 白色字")
        print("  ✨ 小星星 - 左侧区域")
        print("  🎤 词级卡拉OK - 粉色高亮")
        print("  🎯 Forced Alignment - 精确对齐")
        print(f"  🎨 配色方案 - {self.color_scheme['name']}")
        
        # 加载原文
        eng_sentences, chi_sentences = self.load_stories(story_num)
        
        # 获取音频
        audio_files = sorted(list(self.audio_dir.glob("*.wav")))
        if not audio_files:
            audio_files = sorted(list(self.audio_dir.glob("*.mp3")))
        audio_path = str(audio_files[story_num - 1])
        
        # 提取词级时间戳
        if use_forced_alignment:
            full_english_text = ' '.join(eng_sentences)
            word_timestamps = self.extract_word_timestamps_with_forced_alignment(audio_path, full_english_text)
            aligned = self.align_sentences_with_forced_alignment(word_timestamps, eng_sentences, chi_sentences)
        else:
            print("   ⚠️ 必须使用 Forced Alignment")
            return
        
        # 创建字幕
        print("\n🎬 创建简约字幕...")
        audio_clip = mp.AudioFileClip(audio_path)
        duration = audio_clip.duration
        
        subtitle_clip = self.create_simple_subtitle_clip(aligned, duration)
        print(f"   ✅ 简约字幕创建完成")
        
        # 创建背景
        background = self.create_simple_background(duration)
        
        # 合成
        print("\n🎬 合成最终视频...")
        final = mp.CompositeVideoClip([background, subtitle_clip])
        final = final.with_audio(audio_clip)
        
        # 输出
        output_path = self.output_dir / f"Story_{story_num:02d}_Karaoke_Mobile_V3_Simple.mp4"
        
        print(f"\n⚡ 使用无损编码（完全无色带，文件较大）...")
        final.write_videofile(
            str(output_path),
            fps=60,  # 60fps流畅动画
            codec='libx264',
            audio_codec='aac',
            preset='veryslow',  # 最慢编码，最高质量
            threads=8,
            ffmpeg_params=[
                '-crf', '0',  # CRF 0 = 完全无损
                '-pix_fmt', 'yuv444p',  # 4:4:4色度采样，无色度压缩
                '-qp', '0',  # 量化参数0 = 无损
                '-movflags', '+faststart'
            ]
        )
        
        print(f"\n✅ 完成！视频已保存: {output_path}")
        print(f"   尺寸: {self.width}x{self.height} (9:16 竖屏)")
        print(f"   时长: {duration:.1f}秒")
        print(f"   风格: 简约风格 (快手/抖音优化)")
        print(f"   布局: 左侧布局，避开右侧按钮")


if __name__ == "__main__":
    generator = KaraokeAlignmentGeneratorMobileV3Simple()
    generator.generate(story_num=2, use_forced_alignment=True, theme_override=1)  # Story 2 使用冰蓝极光主题
