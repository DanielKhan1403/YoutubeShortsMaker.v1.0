# YouTube Shorts Creator

## 🌐 English Version

### Description
YouTube Shorts Creator is an automated tool for downloading, transcribing, formatting, and generating YouTube Shorts with subtitles and watermarks. This script ensures quick and seamless content creation, requiring minimal user input.

### Features
- 📹 **Download YouTube Videos**: Fetches high-quality video files.
- 🎤 **Audio Transcription**: Converts speech into subtitles using OpenAI's Whisper.
- 📝 **Subtitle Overlay**: Automatically adds readable subtitles to videos.
- 🎬 **Shorts Formatting**: Resizes and optimizes videos for YouTube Shorts (9:16 aspect ratio).
- ✨ **Watermark Support**: Allows adding custom branding to videos.
- ✅ **Automated Workflow**: No manual editing required.

### Requirements
- Python 3.8+
- FFmpeg
- ImageMagick
- Dependencies (install via `pip install -r requirements.txt`)

### Installation & Usage
1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Run the Script:**
   ```bash
   python youtubescript.py
   ```
3. **Follow On-Screen Instructions:**
   - Enter the YouTube video URL.
   - Specify a watermark (optional).
   - Wait for processing.

### Output
Processed Shorts videos will be saved in the `shorts_videos/` directory.

### Troubleshooting
- Ensure **FFmpeg** and **ImageMagick** are installed and accessible.
- Run with Python 3.8+.
- If errors occur, check terminal output for missing dependencies.

---

## 🇷🇺 Русская версия

### Описание
YouTube Shorts Creator — это автоматизированный скрипт для скачивания, транскрипции, форматирования и генерации YouTube Shorts с субтитрами и водяными знаками. Минимум ручного вмешательства!

### Возможности
- 📹 **Скачивание видео**: Загрузка высококачественных YouTube-роликов.
- 🎤 **Транскрипция**: Конвертация аудио в текст (благодаря OpenAI Whisper).
- 📝 **Субтитры**: Добавляются ваши автоматически сгенерированные тексты.
- 🎬 **Автоформат**: Расширение, обрезка до Shorts (9:16).
- ✨ **Водяные знаки**: Добавьте свой бренд.
- ✅ **Автоматизация**: Все процессы выполняются без монтажа.

### Установка
1. **Инсталляция**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Запуск**:
   ```bash
   python youtubescript.py
   ```
3. **Проследуйте инструкциям**

### Результат
Обработанные Shorts будут в `shorts_videos/`

