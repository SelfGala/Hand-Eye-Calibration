# D405 眼在手上标定

Python 核心、PyQt5 原生桌面窗口、D405 SDK 取图、ROS 2 按需位姿服务。按照原方案的坐标与算法实现，采样流程已按本次要求简化。

**点击拍照 → 查询一次实测末端位姿 → 取返回后到达的一帧新图 → 检测并保存。没有倒计时、画面稳定等待、停稳判定或两次读取。** 用户手动保持机械臂不动，完成后移动到下一个姿态。取下一帧是正常取图，不是等待画面稳定。不会控制机械臂运动。

方案审阅和取舍见 [REVIEW.md](REVIEW.md)，现场待补充信息见 [docs/pending_integration.md](docs/pending_integration.md)，实测结果见 [docs/verification.md](docs/verification.md)。原始方案不改写。

## 无硬件试运行

Python 3.10 及以上；先检查已有 OpenCV，已有可用 `cv2` 时保留现场安装。

```bash
python -m pip install -e ".[gui]"
# 仅在没有 cv2 时安装这一项；不要和现场 opencv / contrib 包混装。
python -m pip install "opencv-python-headless>=4.8,<5"

handeye doctor
handeye gui --demo --config config/d405_eye_in_hand.yaml
handeye offline --session examples/synthetic_session --method PARK --method TSAI
```

`--demo` 在标题和数据来源中标注合成演示，不访问机器人或相机。随附示例包含 20 组训练、5 组独立验证、已知真值以及报告，仅用于演示和复算。桌面窗口可以 `handeye gui --session examples/synthetic_session` 离线打开。示例中三维角点投影生成的测量与栅格渲染图并非完全同一个噪声模型，`--redetect` 会得到不同的亚像素误差。

如果 `handeye` 不在 PATH，可用 `python -m handeye_core.cli` 替代。支持额外生成合成数据：

```bash
handeye demo --output ./handeye_data --train 20 --validation 5 --noise-px 0.1
handeye offline --session <上条命令输出的会话目录> --method PARK
handeye offline --session <会话目录> --method PARK --redetect
```

## Orin 与工控机部署

先在两台设备使用同一 ROS 2 发行版和兼容的 Python，确认 DDS 网络互通。这里没有假定具体 JetPack 或 ROS 版本。下面使用 Bash，在每台机器的项目根目录执行；`$ROS_DISTRO` 必须来自现场已有环境。

```bash
source /opt/ros/$ROS_DISTRO/setup.bash
python3 -c "import sys, cv2, numpy, scipy, yaml, rclpy; print(sys.executable, cv2.__version__)"
# 使用与 rclpy 兼容的解释器；不要创建隔离到看不到系统 ROS 的普通虚拟环境。
python3 -m pip install --no-deps --no-build-isolation -e .
cd ros2_ws
colcon build --symlink-install
source install/setup.bash
cd ..
```

系统需提供 numpy、scipy、yaml 和 OpenCV 4.x；Orin 另需 PyQt5、pyrealsense2 及可用桌面。`--no-deps` 是为了保留现场依赖，缺包先按该系统可用方式补齐。Python 打包需 setuptools>=68。RealSense/Jetson 安装采用 [官方安装说明](https://github.com/realsenseai/librealsense/blob/master/doc/installation_jetson.md)，不在未知系统上自动换驱动。

工控机：优先把 `RobotPoseService` 嵌入持有现有机械臂连接的驱动进程，接入一次实测读取函数，参见 [tools/robot_reader_example.py](tools/robot_reader_example.py)。独立桥进程中的 reader factory 只能用于现场允许的接口或 IPC，导入驱动模块不会共享另一个进程的连接和锁。

```bash
# 在 config/robot_bridge.yaml 中填好真实 reader_factory 后使用：
ros2 launch handeye_robot_bridge pose_service.launch.py config_path:=$PWD/config/robot_bridge.yaml

# 尚无真实 reader 时，只能显式启动这个模拟服务：
ros2 run handeye_robot_bridge mock_pose_service
ros2 service call /mock_handeye_robot_bridge/query_counts std_srvs/srv/Trigger '{}'
```

模拟服务返回显式 `replay` 来源、SDK 调用数 0，可通过 ROS 参数修改 `translation_m` 和 `quaternion_xyzw`。它只验证通信和计数，恒定模拟位姿不具有标定所需旋转激励。完整数学演示使用 `gui --demo`。

Orin：相机由 SDK 独占，不要同时启动另一个占用同一 D405 的驱动。

```bash
handeye doctor --config config/d405_eye_in_hand.yaml --camera --ros-graph
# 二选一，均启动同一个 GUI 和后端：
handeye gui --config config/d405_eye_in_hand.yaml
ros2 launch handeye_ros calibrate.launch.py config_path:=$PWD/config/d405_eye_in_hand.yaml
```

`doctor --ros-graph` 仅发现服务，不查询位姿。默认 GUI 只创建 service client，无状态 topic 或 TF 订阅；预览、求解、导出不读取机器人。

## 操作

1. 设置棋盘的计数含义、列行数与单格边长。默认 9×6 内角点、54 点、43 mm；如果是 9×6 方格，选择“计数指方格数”，自动得到 8×5、40 个内角点。参数变化后新建会话，旧会话保留。
2. 固定棋盘，完整入镜。查看绿色 0 点和 X/Y 轴，保证不同姿态对应同一物理角。普通棋盘不能保证自动辨别 180° 方向；可在拍照前勾选编号旋转，也可选中已保存样本纠正。
3. 手动保持不动后点击拍照或空格；完成后再换姿态，绕至少两个非平行轴旋转。建议约 25 组训练，建议值不锁住计算按钮。
4. 单独换姿态采集验证组，不要把同一停留位置的相邻照片分到训练与验证。可启用/停用样本、切换角色；不会删除原图。
5. 点击“计算 / 验证”，分别查看 PARK 与 TSAI。验证不是计算或导出的前置条件；无验证时显示“未验证”。导出所选算法的 YAML、JSON、CSV、文本报告。

重复按键和任务并发只阻止重复提交，不判断机械臂是否稳定。模糊指标、覆盖率、PnP 误差和运动标志仅记录或提示，不拦截拍照。没有完整棋盘时无法产生有效 PnP 样本，仍可能已发生本次唯一的位姿读取；不做额外的图像预检等待。

## 数据和数学

`T_a_b` 把 b 下的点变到 a。`G=T_base_link_r_end_link`、`C=T_camera_optical_board`，输出 `X=T_r_end_link_camera_optical`，满足 `G X C=Y`。平移米、四元数 xyzw。Park/Tsai 的输入均为绝对 G、C，不对机器人位姿默认求逆。

会话保存活动相机内参与畸变、原图、原始角点、手工排列、G/C、查询 ID 和时间。数据文件采用临时写入加原子替换；已提交索引才是有效样本清单。角点修正和重检测保存在 `analyses/`，索引历史在 `revisions/`，原始 `sample.json` 与 `image.png` 不变。启用、角色、分析更新都会增加 `dataset_revision`，旧结果标记为旧数据版本。

D405 彩色流的反向 Brown-Conrady 模型使用与 SDK 一致的独立映射，归一化角点后求 PnP，误差仍在原始像素下计算。核心离线使用不导入 SDK；适配已与 SDK 纯投影函数比较，详见 [相机模型说明](docs/camera_model.md)。

验证组不参与 X 或 Y 的拟合。报告分别列出 PnP 误差、训练/验证的固定棋盘一致性，以及通过训练 Y 预测的像素误差；空指标是 `null`。2 mm / 0.5° 只是提示线，不是业务验收标准。

## ROS 操作接口

`calibrate.launch.py` 的应用节点在 `/handeye/handeye_calibrator`。直接 `handeye gui` 的默认节点在 `/handeye_calibrator`。接口名为私有名称，遵循节点命名空间。

| 私有接口 | 类型 | 用途 |
| --- | --- | --- |
| `~/capture` | `CaptureSample` | 传 role/label/flip_180，返回 job_id |
| `~/compute` | `StartCompute` | 指定算法，返回 job_id |
| `~/get_job` | `GetJob` | 本地异步任务状态、请求次数、样本和结果 ID |
| `~/set_sample_enabled` | `SetSampleEnabled` | 启用/停用 |
| `~/get_result` | `GetResult` | 变换和完整 report_json |
| `~/new_session` | `std_srvs/Trigger` | 新建会话 |
| `~/status` | `DiagnosticArray` | 本地应用状态，不查询机械臂 |

默认机器人接口 `/robot/get_end_pose`，定义在 `handeye_interfaces/srv/GetEndPose.srv`。桥接器业务计数接口为 `~/query_counts`。一次请求超时后不重试，迟到结果不会交给后续拍照；前次驱动读取未结束时桥接器返回 `ROBOT_BUSY`。真实 SDK 仍需有有限超时，Python 无法安全中断持有驱动锁的底层函数。

已有状态 topic 或 TF 时，可将 `robot.pose_source` 改为 `topic_snapshot` 或 `tf_snapshot`。订阅仅存在于拍照事务中；消息需带本轮可比较的新 ROS 时间戳，两机需对时。PoseStamped 的末端由配置绑定，TransformStamped 则检查父子帧。自定义消息需显式适配。机器人 SDK 调用数在兼容路径中标记为未知，不能从订阅次数推断。

## 发布 TF

求解和导出不会自动发布。先根据现场 TF 树选好连接点，保证目标子帧没有其他发布者；工具不靠短时间监听推断全局 TF 树是否永远无冲突。

```bash
ros2 launch handeye_ros publish_calibration.launch.py result_path:=/absolute/path/calibration.yaml
```

默认发布 `r_end_link <- 实际标定光学帧`。已有相机子树时优先接 `camera_link`：额外文件写入 `parent_frame: camera_link`、`child_frame: 标定光学帧`、`T_parent_child: 可信4x4矩阵`，再传 `camera_link_frame:=camera_link camera_link_to_optical_path:=/absolute/path/extrinsic.yaml`。程序使用 `X @ inverse(T_camera_link_optical)`，不会猜固定轴旋转。

数学测试运行 `python -m pytest -q`；依赖测试模块无需 ROS、Qt 或相机。附带的 Qt 工作流检查使用 `tools/verify_desktop.py`，只验证原生界面与核心流程，没有 60 秒空闲或 10 分钟耐久验收门槛。

