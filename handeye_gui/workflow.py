def guidance(samples, recommended=25):
    train = sum(s["enabled"] and s["role"] == "train" for s in samples)
    validation = sum(s["enabled"] and s["role"] == "validation" for s in samples)
    return (f"训练 {train} 组 / 建议 {recommended} 组，验证 {validation} 组\n"
            "1. 棋盘固定，相机安装在末端。\n"
            "2. 让完整棋盘入镜，核对绿色 0 点对应同一物理角。\n"
            "3. 手动保持机械臂不动，点击拍照；完成后换一个朝向。\n"
            "4. 绕不同轴倾斜采样；单独采集验证姿态。\n"
            "5. 计算并查看结果，导出所选算法。\n\n"
            "拍照后请保持不动，直到提示保存完成。")

