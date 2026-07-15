import os
import subprocess
import shutil
from pathlib import Path
import argparse

parser = argparse.ArgumentParser(description="Build and cache gsplat CUDA extensions.")
parser.add_argument(
    "--build-root",
    type=Path,
    required=True,
    help="Path to the central build cache directory (e.g. /shared/gsplat_builds)",
)

def get_commit_hash():
    return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()

def main():
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parent
    build_root: Path = args.build_root
    commit = get_commit_hash()
    build_dir = build_root / commit

    if build_dir.exists():
        print(f"✅ Build for {commit} already exists at {build_dir}")
        return

    print(f"🔨 Building gsplat extensions for commit {commit} ...")
    build_tmp = repo_root / "build_tmp"
    if build_tmp.exists():
        shutil.rmtree(build_tmp)

    os.environ["BUILD_CUDA"] = "1"
    subprocess.check_call(
        ["python", "setup.py", "build_ext", f"--build-lib={build_tmp}"],
        cwd=repo_root
    )

    build_dir.mkdir(parents=True, exist_ok=True)
    for so_file in build_tmp.rglob("*.so"):
        rel = so_file.relative_to(build_tmp)
        dest = build_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(so_file, dest)
        print(f"📦 Copied {rel}")

    shutil.rmtree(build_tmp)
    print(f"✅ Done. Stored build at: {build_dir}")

if __name__ == "__main__":
    main()
