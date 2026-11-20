# D405 眼在手上标定

棋盘固定，相机装在末端。点一次拍照，读一次末端位姿，拍一帧，检测并保存。机械臂保持不动，由人换下一个姿态。

## 安装

```bash
python -m pip install -e ".[gui]"
```

没有 OpenCV 时再装：`python -m pip install "opencv-python-headless>=4.8,<5"`。

## 开始标定

位姿从哪读，由配置文件决定。

ROS，读 `/tf` 里的 `base_link` → `r_end_link`：

```bash
./step.sh
```

对应 `config/d405_eye_in_hand.yaml`。左臂把里面的 `end` 改成 `l_end_link`。

Flexiv，用 RDK 读 `tcp_pose`（`Rizon4s-063191`）：

```bash
handeye doctor --config config/d405_flexiv_eye_in_hand.yaml --camera
handeye gui --config config/d405_flexiv_eye_in_hand.yaml --resume
```

## 操作

1. 棋盘完整入镜。绿色 0 点在每次拍照里都是同一个物理角。
2. 手臂停稳后点拍照或按空格，保存完再换姿态。训练大约 25 组，绕不同的轴转。
3. 另外采一组验证样本。
4. 点「计算 / 验证」，导出 PARK 或 TSAI。

导出的 YAML、JSON、CSV 在 `handeye_data` 当前会话的 `results/`。中断后用上面的 `--resume` 接着采。

离线重算：

```bash
handeye offline --session <会话目录> --method PARK --method TSAI
```

没有相机和机械臂时：

```bash
handeye gui --demo --config config/d405_eye_in_hand.yaml
```
