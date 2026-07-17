import os
import shutil

PIPELINE_DEPLOYMENT = [
    'configs', 'dataset', 'models', 'scripts', 'utils',
    'requirements.txt', 'setup.py', 'README.md',
]


def save_pipeline_snapshot(deployment_list, dst_dir, src_dir='.'):
    """Copy pipeline source files to a snapshot directory."""
    if deployment_list is None:
        deployment_list = PIPELINE_DEPLOYMENT
    os.makedirs(dst_dir, exist_ok=True)
    for item in deployment_list:
        src_path = os.path.join(src_dir, item)
        dst_path = os.path.join(dst_dir, item)
        if os.path.exists(src_path):
            if os.path.isdir(src_path):
                shutil.copytree(src_path, dst_path, dirs_exist_ok=True,
                                ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.git'))
            else:
                shutil.copy2(src_path, dst_path)
