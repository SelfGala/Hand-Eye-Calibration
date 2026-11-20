#!/usr/bin/env bash
# D405 眼在手上标定。相机是本机 USB 上的 D405，末端位姿读 /tf。
# 右臂：base_link -> r_end_link。左臂把 config/d405_eye_in_hand.yaml 的 end 改成 l_end_link。
# /foo/camera_1 和 /foo/camera_2 是底盘相机，不参与这次标定。
cd "$(dirname "$0")"

echo "准备标定环境..."
# conda 的启停脚本在 set -e 下会让本脚本直接退出，所以这里先关掉。
set +e
set +u
# shellcheck disable=SC1091
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate biaoding
status=$?
set -u
if [ "$status" -ne 0 ]; then
  echo "无法激活 conda 环境 biaoding（退出码 $status）" >&2
  exit "$status"
fi
hash -r

CONFIG=config/d405_eye_in_hand.yaml

echo "========== 1/3 检查环境、D405 和 TF =========="
handeye doctor --config "$CONFIG" --camera --ros-graph
status=$?
if [ "$status" -ne 0 ]; then
  echo "环境检查失败（退出码 $status）" >&2
  exit "$status"
fi

echo
echo "========== 2/3 打开标定界面 =========="
echo "棋盘固定不动，相机在末端。"
echo "让完整棋盘入镜，确认绿色 0 点始终是同一个物理角。"
echo "手臂停稳后点拍照或按空格，保存完成后再换一个姿态。"
echo "先采训练组（建议约 25 组，绕不同轴），再单独采验证组。"
echo "然后点“计算 / 验证”，导出 PARK 或 TSAI。"
handeye gui --config "$CONFIG" --resume
status=$?
if [ "$status" -ne 0 ]; then
  echo "标定界面退出（退出码 $status）" >&2
  exit "$status"
fi

echo
echo "========== 3/3 结果 =========="
echo "界面里导出的 YAML / JSON / CSV 在 handeye_data 当前会话的 results 目录。"
echo "需要离线重算时："
echo "  handeye offline --session <会话目录> --method PARK --method TSAI"
