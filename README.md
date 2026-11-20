# D405 眼在手上标定

棋盘固定，相机装在末端。点一次拍照，读一次末端位姿，拍一帧，检测并保存。机械臂保持不动，由人换下一个姿态；也可以勾选自动换位，拍照成功后由机械臂自己转到下一点。

## 安装

```bash
python -m pip install -e ".[gui]"
```

没有 OpenCV 时再装：`python -m pip install "opencv-python-headless>=4.8,<5"`。

## 开始标定

位姿从哪读，由配置文件决定。

ROS，读 `/tf` 里的 `base_link` → `l_end_link`（左臂）：

```bash
./step.sh
```

对应 `config/d405_eye_in_hand.yaml`。右臂把里面的 `end` 改成 `r_end_link`，`auto_motion.move_group` 改成 `right_arm`。

`step.sh` 带 `--resume`，会接着上一个会话采。换了末端后旧会话不能接着用，启动时会报 `RESUME_MISMATCH`；这时不带 `--resume` 启动一次来新建会话：

```bash
handeye gui --config config/d405_eye_in_hand.yaml
```

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

会话目录和样本目录用创建时的本地时间命名，例如 `handeye_data/20261120_143805/samples/20261120_143912`。

## 自动换位（可选）

不勾选时和上面一样全手动。勾选后仍然由人按拍照，只有拍照成功才会动。

准备：

- 工控机上 `move_group` 在跑，能控制 `auto_motion.move_group` 对应的手臂；遥操作、键盘伺服这些会同时给手臂发指令的节点先停掉。
- 本机能通过 ROS 2 看到 `/compute_cartesian_path`、`/execute_trajectory` 和 `/trajectory_execution_event`，并装有 `moveit_msgs`（`sudo apt install ros-humble-moveit-msgs`）。`handeye doctor` 会检查这个包。
- `config/d405_eye_in_hand.yaml` 的 `auto_motion.board_center_m` 填棋盘中心在 `base_link` 下的位置。

步骤：

1. 手动把手臂摆到棋盘在画面中间的姿态，勾选「拍照成功后自动移动到下一点」。当前姿态就是起点。配置的棋盘距离和相机实测相差超过 0.3 m 时不会开启。
2. 点拍照。保存成功后手臂沿直线转到下一点，停稳后再点拍照，如此重复。
3. 规划时发现到不了的点自动跳过。棋盘拍不全或者不想要这个点，点「跳过此点，移动到下一点」。
4. 运动中点「取消当前任务」会让手臂停下。执行失败时也会停在原地等人处理，不会自己继续。
5. 所有点走完后自动关闭，回到手动。

各点是把起点姿态绕末端和棋盘中心连线上的一点转出来的：倾角按螺旋从小到大，最大 `max_tilt_deg`，同时绕视线滚转 ±`max_roll_deg`。棋盘近时绕棋盘中心转；棋盘远时这样转手臂够不着，转动中心会靠近末端，使末端离起点不超过 `max_travel_m`，代价是棋盘在画面里偏移，开启时界面会给出偏移角度。速度由 `velocity_scale` 控制。

离线重算：

```bash
handeye offline --session <会话目录> --method PARK --method TSAI
```

没有相机和机械臂时：

```bash
handeye gui --demo --config config/d405_eye_in_hand.yaml
```
