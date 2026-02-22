#!/usr/bin/env python3
"""
Trim video to a specified duration.
Usage: python3 trim.py input.mp4 15
       python3 trim.py input.mp4 15 -o output.mp4
       python3 trim.py input.mp4 15 --start 5   # від 5с до 20с
"""

import argparse
import subprocess
import sys
from pathlib import Path


def trim_video(input_path: str, duration: float, start: float = 0.0, output_path: str = None):
    input_file = Path(input_path)
    if not input_file.exists():
        print(f"❌ Файл не знайдено: {input_file}")
        sys.exit(1)

    if output_path is None:
        output_path = str(input_file.parent / f"{input_file.stem}_trimmed{input_file.suffix}")

    end = start + duration
    print(f"✂️  Обрізаємо: {start:.1f}s → {end:.1f}s ({duration:.1f}s)")

    cmd = (
        f'ffmpeg -y -ss {start:.3f} -i "{input_path}" '
        f'-t {duration:.3f} '
        f'-c:v libx264 -crf 18 -preset fast '
        f'-c:a aac "{output_path}"'
    )

    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Помилка:\n{result.stderr}")
        sys.exit(1)

    print(f"✅ Збережено: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Обрізка відео до заданої тривалості")
    parser.add_argument("input",    help="Вхідний відео файл")
    parser.add_argument("duration", type=float, help="Тривалість в секундах (напр. 15)")
    parser.add_argument("-s", "--start", type=float, default=0.0,
                        help="Початок обрізки в секундах (default: 0)")
    parser.add_argument("-o", "--output", default=None,
                        help="Вихідний файл (default: input_trimmed.mp4)")
    args = parser.parse_args()
    trim_video(args.input, args.duration, args.start, args.output)


if __name__ == "__main__":
    main()