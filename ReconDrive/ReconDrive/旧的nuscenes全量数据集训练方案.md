# ReconDrive 全量数据集训练交付方案

## 二、配置文件修改

文件：`configs/nuscenes/recondrive.yaml`

```yaml
# ===== 必须修改 =====

# 1. 数据路径（服务器上的 nuscenes 路径）
data_path: '/path/to/nuscenes'          # 原值: ./data/nuscenes/

# 2. 版本切换为12Hz
nuscenes_version: 'interp_12Hz_trainval'        # 原值: v1.0-mini

# 3. 缓存目录（建议指到大磁盘路径）
cache_dir: '/path/to/cache/nuscenes-trainval'  # 原值: ./work_dirs/cache_data/nuscenes-mini

# 4. 帧率切换为 12Hz（全量数据包含高频 sweeps）
frame_rate: 12                           # 原值: 2

# 5. context_span 增大（全量场景更长，12Hz 下 6 帧 ≈ 0.5s 间隔）
context_span: 6                          # 原值: 1

# 6. 训练 epoch 数
train_epoch: 10                          # 原值: 2

# ===== 按需调整 =====

# 7. GPU 数量（按服务器配置）
devices: [0,1,2,3,4,5,6,7]              # 原值不变

# 8. batch_size 现在是 1，可以增大一点
```

修改后效果（仅列出有变化的行）：

<br />

```yaml
# 修改前                                # 修改后
train_epoch: 2                          train_epoch: 10
frame_rate: 2                           frame_rate: 12
cache_dir: './work_dirs/cache_data/...' cache_dir: '/path/to/cache/nuscenes-trainval'
data_path: './data/nuscenes/'           data_path: '/path/to/nuscenes'
nuscenes_version: 'v1.0-mini'           nuscenes_version: ' interp_12Hz_trainval'
context_span: 1                         context_span: 6
```

***

## 三、必传文件清单

⚠️ **注意**：本地 `checkpoints/` 是一个软链接 → `/data/public_data/ReconDrive_checkpoint`。需要在自己服务器上创建同样的软链接或直接替换为实际目录。

如果要用 `best_module-v2.ckpt` 做微调，仍需要把以下模型权重文件一并传过去（模型构造时就需要加载，即使后续会被 checkpoint 覆盖）：

| 文件                                                       | 大小       | 说明                                   |
| -------------------------------------------------------- | -------- | ------------------------------------ |
| `checkpoints/vggt.pt`                                    | \~2.5 GB | VGGT backbone 预训练权重（`__init__` 时就加载） |
| `checkpoints/sam2.1_hiera_small.pt`                      | \~150 MB | SAM2 分割模型权重（训练时懒加载）                  |
| `work_dirs/recondrive_training/ckpt/best_module-v2.ckpt` | \~4.3 GB | 微调用 checkpoint（可选，不传则从头训）            |

### 收到文件需要做的

```bash
cd /path/to/ReconDrive

# 方案1（推荐）：创建软链接指向服务器上的权重目录
ln -s /path/to/colleagues/checkpoint/dir checkpoints

# 方案2：直接创建目录，把收到的权重文件放进去
mkdir checkpoints
# 然后把 vggt.pt 和 sam2.1_hiera_small.pt 放进去
```

如果不挂载软链接或不放这些文件，训练时会直接报错找不到对应权重。

### 下载链接

VGGT 和 SAM2 的预训练权重可以从官方获取：

| 模型 | 下载地址 |
|------|---------|
| VGGT | `https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt`（另存为 `checkpoints/vggt.pt`） |
| SAM2 | 安装 `pip install sam2` 后，从安装目录中找到 `sam2.1_hiera_small.pt`，或从 GitHub Releases 下载 |

VGGT 下载后改名为 `vggt.pt`，SAM2 改名为 `sam2.1_hiera_small.pt`，放入 `checkpoints/` 目录即可。

***

## 四、训练命令

### 从 checkpoint 训练

```bash
conda activate recondrive

CUDA_PATH=/usr/local/cuda-12.8 \
TORCH_CUDA_ARCH_LIST="8.9;9.0" \
python scripts/trainer.py \
    --cfg_path configs/nuscenes/recondrive.yaml \
    --pretrained_ckpt /path/to/best_module.ckpt \
    --train_4d \
    --devices 4
```

### 单卡训练

```bash
conda activate recondrive

CUDA_VISIBLE_DEVICES=5 \
CUDA_PATH=/usr/local/cuda-12.8 \
TORCH_CUDA_ARCH_LIST="8.9;9.0" \
python scripts/trainer.py \
    --cfg_path configs/nuscenes/recondrive.yaml \
    --pretrained_ckpt /path/to/best_module.ckpt \
    --train_4d \
    --devices 1
```

***

## 五、训练后测试

```bash
CUDA_PATH=/usr/local/cuda-12.8 \
TORCH_CUDA_ARCH_LIST="8.9;9.0" \
python scripts/inference.py \
    --cfg_path configs/nuscenes/recondrive.yaml \
    --restore_ckpt work_dirs/recondrive_training/ckpt/best_module.ckpt \
    --output_dir work_dirs/inference_full \
    --device cuda:0 \
    --eval_resolution 280x518
```

### 可视化 LiDAR 结果

```bash
# 单样本可视化
python show_lidar/visualize_lidar.py work_dirs/inference_full/scene-0061/sample_0000/lidar

# 批量所有场景
python show_lidar/visualize_lidar.py work_dirs/inference_full --batch
```

***

### 目录结构

服务器上 nuscenes 全量数据需要申请访问权限，数据目录结构应为：

```
/path/to/nuscenes/
├── samples/
├── sweeps/
├── maps/
├── v1.0-trainval_meta/
└── v1.0-trainval/
```



***

## 六、注意事项

1. **首次运行会缓存数据**：全量 nuscenes 预处理生成缓存到 `cache_dir`，约 **50-100 GB**，耗时数小时
2. **训练时间预估**：700 场景 × 10 epoch，8 卡 H100 约 **2-3 天**
3. **显存 OOM**：`batch_size: 1` 已最小；`accumulate_grad_batches: 8` 可以降，但注意总 batch size = 1×8 = 8
4. **测试集**：还是那 10 个场景（scene-0061/0103/0553/0655/0757/0796/0916/1077/1094/1100），与全量 train/val 不重叠
5. **LiDAR 评估指标**：inference 完成后会在 JSON 中输出 `depth_l2`, `depth_median_l2`, `intensity_rmse`, `ray_drop_acc`, `chamfer_distance`

