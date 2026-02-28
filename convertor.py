import subprocess
import argparse
import sys
from pathlib import Path


def run(command: str, description: str):
    print(f"▶ {description}...")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {result.stderr}")
    return result


def convert_mkv_to_mp4(video_path: str, output_path: str = None) -> str:
    input_path = Path(video_path).resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if input_path.suffix.lower() != ".mkv":
        raise ValueError(f"Expected .mkv file, got: {input_path.suffix}")

    out_path = Path(output_path).resolve() if output_path else input_path.with_suffix(".mp4")

    run(
        f'ffmpeg -y -i "{input_path}" -vcodec copy -acodec aac -b:a 192k -movflags +faststart "{out_path}"',
        "Convert MKV to MP4",
    )

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"Done! → {out_path} ({size_mb:.1f} MB)")
    return str(out_path)


def batch_convert(input_dir: str, output_dir: str = None) -> list[str]:
    input_dir = Path(input_dir).resolve()
    mkv_files = sorted(input_dir.glob("*.mkv"))

    if not mkv_files:
        print(f"No .mkv files found in: {input_dir}")
        return []

    out_dir = None
    if output_dir:
        out_dir = Path(output_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    failed = []

    for i, mkv_file in enumerate(mkv_files, 1):
        print(f"\n[{i}/{len(mkv_files)}]", end=" ")
        out_path = (out_dir / mkv_file.with_suffix(".mp4").name) if out_dir else None

        try:
            result = convert_mkv_to_mp4(str(mkv_file), str(out_path) if out_path else None)
            results.append(result)
        except Exception as e:
            print(f"FAILED: {mkv_file.name} — {e}")
            failed.append(mkv_file.name)

    print(f"\n✅ Converted: {len(results)}/{len(mkv_files)} files")
    if failed:
        print(f"❌ Failed: {', '.join(failed)}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Convert MKV files to MP4 format",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python mkv_to_mp4.py video.mkv\n"
            "  python mkv_to_mp4.py video.mkv -o output.mp4\n"
            "  python mkv_to_mp4.py --batch ./videos/\n"
            "  python mkv_to_mp4.py --batch ./videos/ --output-dir ./converted/\n"
        )
    )
    parser.add_argument("input", nargs="?", help="Path to a single .mkv file")
    parser.add_argument("-o", "--output", help="Output file path (single file mode)")
    parser.add_argument("--batch", metavar="DIR", help="Directory to batch convert all .mkv files")
    parser.add_argument("--output-dir", metavar="DIR", help="Output directory (batch mode)")

    args = parser.parse_args()

    if args.batch:
        batch_convert(args.batch, args.output_dir)
    elif args.input:
        try:
            convert_mkv_to_mp4(args.input, args.output)
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()