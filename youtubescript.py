import os
import yt_dlp
import whisper
import torch
import sys
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip
from moviepy.video.fx.all import mirror_x
from tqdm import tqdm
import numpy as np
from moviepy.config import change_settings
import shutil
from concurrent.futures import ThreadPoolExecutor
from termcolor import colored


# Настройка цветного вывода
def print_info(text): print(colored(text, 'cyan'))


def print_success(text): print(colored(text, 'green'))


def print_error(text): print(colored(text, 'red'))


def print_warning(text): print(colored(text, 'yellow'))


# Установка пути к ImageMagick
IMAGEMAGICK_PATH = shutil.which("magick") or r"C:\Users\marak\OneDrive\Desktop\LSP\ImageMagick-7.1.1-Q16-HDRI\magick.exe"
if os.path.exists(IMAGEMAGICK_PATH):
    change_settings({"IMAGEMAGICK_BINARY": IMAGEMAGICK_PATH})
else:
    print_error("⚠️ ImageMagick не найден. Установите его для корректной работы!")


def download_youtube_video(url, output_path):
    """Скачивание видео с YouTube с проверками."""
    if not url or not url.startswith(('http', 'https')):
        print_error("❌ Неверный URL! Введите корректную ссылку на YouTube видео.")
        return None

    try:
        os.makedirs(output_path, exist_ok=True)
        ydl_opts = {
            'format': 'bestvideo+bestaudio/best',
            'outtmpl': os.path.join(output_path, '%(title)s.%(ext)s'),
            'merge_output_format': 'mp4',
            'postprocessors': [{'key': 'FFmpegVideoConvertor', 'preferedformat': 'mp4'}]
        }
        print_info("⬇️ Начинаем загрузку видео...")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info).replace(".webm", ".mp4")
            print_success(f"✅ Видео успешно загружено: {os.path.basename(filename)}")
            return filename
    except Exception as e:
        print_error(f"❌ Ошибка при загрузке: {str(e)}")
        return None


def transcribe_video(video_path):
    """Транскрипция аудио с проверками."""
    if not os.path.exists(video_path):
        print_error("❌ Видео файл не найден!")
        return []

    try:
        # Указываем путь к моделям Whisper при запуске как .exe
        if getattr(sys, 'frozen', False):  # Если запущен как .exe
            BASE_PATH = os.path.dirname(sys.executable)
            whisper_model_dir = os.path.join(BASE_PATH, "whisper_models")
            os.environ["XDG_CACHE_HOME"] = BASE_PATH  # Whisper будет искать в .cache/whisper относительно BASE_PATH
            print_info(f"🔧 Установлен путь к моделям Whisper: {whisper_model_dir}")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print_info(f"🎙 Используем устройство: {device}")
        model = whisper.load_model("base", device=device)
        print_info("🔊 Извлекаем аудио из видео...")
        audio = whisper.load_audio(video_path)
        sample_rate = whisper.audio.SAMPLE_RATE
        segment_length = 30 * sample_rate
        num_segments = int(np.ceil(len(audio) / segment_length))
        print_info(f"📊 Найдено {num_segments} аудио сегментов")

        results = []
        with tqdm(total=num_segments, desc="🎤 Транскрипция", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}") as pbar:
            for i in range(num_segments):
                start = i * segment_length
                end = min(start + segment_length, len(audio))
                segment = audio[start:end]
                segment = whisper.pad_or_trim(segment)
                result = model.transcribe(segment, word_timestamps=True, language="ru")
                for seg in result["segments"]:
                    seg["start"] += i * 30
                    seg["end"] += i * 30
                results.extend(result["segments"])
                pbar.update(1)
        print_success("✅ Транскрипция успешно завершена!")
        return results
    except Exception as e:
        print_error(f"❌ Ошибка транскрипции: {str(e)}")
        return []


def format_video_for_shorts(video):
    """Форматирование видео для Shorts."""
    try:
        target_width, target_height = 1080, 1920
        aspect_ratio = video.w / video.h
        scale_factor = 1.2
        if aspect_ratio > target_width / target_height:
            new_width = int(target_width * scale_factor)
            video_resized = video.resize(width=new_width)
        else:
            new_height = int(target_height * scale_factor)
            video_resized = video.resize(height=new_height)
        return video_resized.on_color(size=(target_width, target_height), color=(0, 0, 0), pos="center")
    except Exception as e:
        print_error(f"❌ Ошибка форматирования видео: {str(e)}")
        return video


def create_text_clip(text, fontsize=60, color="yellow", position="bottom"):
    """Создание текстового клипа."""
    if not text:
        return None
    try:
        return TextClip(
            text.strip(),
            fontsize=fontsize,
            color=color,
            stroke_color="black",
            stroke_width=2,
            font="Arial-Bold",
            method="caption",
            size=(1000, None),
            align="center"
        )
    except Exception as e:
        print_error(f"❌ Ошибка создания субтитров: {str(e)}")
        return None


def process_chunk(i, input_path, segments, start_time, end_time, output_folder, watermark_text):
    """Обработка одного чанка видео."""
    try:
        video = VideoFileClip(input_path)
        video_chunk = video.subclip(start_time, end_time)
        video_chunk = video_chunk.fx(mirror_x)
        video_chunk = format_video_for_shorts(video_chunk)

        subtitle_clips = []
        for seg in segments:
            seg_start = seg["start"]
            seg_end = seg["end"]
            if seg_end > start_time and seg_start < end_time:
                clip_start = max(seg_start - start_time, 0)
                clip_duration = min(seg_end, end_time) - max(seg_start, start_time)
                if clip_duration > 0 and "text" in seg and seg["text"]:
                    txt_clip = create_text_clip(seg["text"], fontsize=60, color="yellow")
                    if txt_clip:
                        txt_clip = txt_clip.set_position(("center", 0.7), relative=True) \
                            .set_start(clip_start) \
                            .set_duration(clip_duration)
                        subtitle_clips.append(txt_clip)

        watermark = create_text_clip(f"{watermark_text}", fontsize=60, color="white")
        if watermark:
            watermark = watermark.set_position("center").set_duration(video_chunk.duration)

        clips = [video_chunk] + ([watermark] if watermark else []) + subtitle_clips
        final_clip = CompositeVideoClip(clips)

        output_path = os.path.join(output_folder, f"#shorts{i + 1} #кино #сериалы.mp4")
        print_info(f"🎬 Создаем видео: {os.path.basename(output_path)}")
        final_clip.write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac",
            fps=30,
            preset="ultrafast",
            threads=4,
            logger=None
        )
        print_success(f"✅ Видео сохранено: {os.path.basename(output_path)}")
        video.close()
    except Exception as e:
        print_error(f"❌ Ошибка обработки чанка {i + 1}: {str(e)}")


def split_video(input_path, output_folder, watermark_text="@GoodFilms", chunk_length=58):
    """Разрезка видео на части."""
    if not os.path.exists(input_path):
        print_error("❌ Исходное видео не найдено!")
        return

    os.makedirs(output_folder, exist_ok=True)
    segments = transcribe_video(input_path)
    if not segments:
        print_warning("⚠️ Не удалось получить субтитры, продолжаем без них")

    try:
        video = VideoFileClip(input_path)
        num_chunks = int(np.ceil(video.duration / chunk_length))
        video.close()
        print_info(f"✂ Разделяем видео на {num_chunks} частей по {chunk_length} секунд")

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for i in range(num_chunks):
                start_time = i * chunk_length
                end_time = min((i + 1) * chunk_length, video.duration)
                print_info(f"⏳ Планируем часть {i + 1}: {start_time:.1f} сек → {end_time:.1f} сек")
                futures.append(
                    executor.submit(process_chunk, i, input_path, segments, start_time, end_time, output_folder,
                                    watermark_text)
                )
            for future in futures:
                future.result()
        print_success("🎉 Все части видео успешно созданы!")
    except Exception as e:
        print_error(f"❌ Ошибка при разделении видео: {str(e)}")


if __name__ == "__main__":
    print_info("🎥 YouTube Shorts Creator v1.0")
    print_info("=================================")

    youtube_url = input(colored("📺 Введите URL видео с YouTube: ", 'cyan'))
    custom_watermark = input(
        colored("✏️ Введите текст для водяного знака (@GoodFilms по умолчанию): ", 'cyan')) or "@GoodFilms"
    output_dir = "shorts_videos"

    downloaded_video = download_youtube_video(youtube_url, output_dir)
    if downloaded_video:
        split_video(downloaded_video, output_dir, watermark_text=custom_watermark)
    else:
        print_error("❌ Не удалось продолжить из-за ошибки загрузки")