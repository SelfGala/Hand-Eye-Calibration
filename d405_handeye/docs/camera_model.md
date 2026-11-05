# D405 彩色流的畸变适配

核对 [RealSense 的 D405 内参实现](https://github.com/realsenseai/librealsense/blob/master/src/ds/d400/d400-private.cpp) 后发现，`get_d405_color_stream_intrinsic` 返回的模型可以是 `RS2_DISTORTION_INVERSE_BROWN_CONRADY`。因此，只支持标准 OpenCV 的 Brown-Conrady 会让默认 D405 彩色路径无法使用；直接把它的系数当作 OpenCV D 也不正确。

当前实现的三条路径：

1. `none`：零畸变。
2. `brown_conrady` / 明确的 `opencv`：OpenCV K/D。
3. `inverse_brown_conrady`：按 SDK 的像素到光线映射归一化角点，用单位 K、零 D 做 PnP；随后用对应的 SDK 投影映射回原始像素计算误差、验证和绘制坐标轴，避免重复校正。

具体映射以 [SDK rs2_project_point_to_pixel / rs2_deproject_pixel_to_point 的实现](https://github.com/realsenseai/librealsense/blob/master/src/rs.cpp) 为依据，向量化实现在 `handeye_core/camera_geometry.py`。没有将模型名称按教科书的另一种“反向”约定自行猜测。算法只使用数值数组，离线复算不需要导入 SDK 或连接相机；保存完整模型、系数和活动 profile 的 SDK 版本供追溯。

测试覆盖非零径向/切向畸变、光线与像素一致性、PnP 恢复以及与 pyrealsense2 的数值比较。SDK 比较只调用纯投影函数，不查询或打开相机。实际设备的内参精度与 profile 可用性仍需现场检查。

