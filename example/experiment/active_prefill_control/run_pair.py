import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable



def run_script(script_name: str) -> str:
    cmd = [PYTHON, str(ROOT / script_name)]
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    print(proc.stdout)
    if proc.stderr:
        print(proc.stderr, file=sys.stderr)
    match = re.search(r"OUTPUT_DIR=(.+)", proc.stdout)
    if match is None:
        raise RuntimeError(f"未在 {script_name} 输出中找到 OUTPUT_DIR")
    return match.group(1).strip()



def main() -> None:
    off_dir = run_script("baseline.py")
    on_dir = run_script("enabled.py")
    subprocess.run(
        [PYTHON, str(ROOT / "compare_runs.py"), off_dir, on_dir],
        check=True,
    )


if __name__ == "__main__":
    main()
