# MuJoCo Setup and Smoke Test / MuJoCo 环境与验收

> [Environment hub](README.md) · [Scene-building tutorial](../tutorials/mujoco-scene-building.md) · [中文指南](#中文指南)

## English guide

### Install the maintained Python bindings

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install mujoco
python -c "import mujoco; print(mujoco.__version__)"
```

The PyPI package includes the MuJoCo library; a separate manual MuJoCo download is not required for this route. Do not begin a new project with the legacy `mujoco-py` wrapper.

### Run a valid physics smoke

```python
import mujoco

xml = """
<mujoco>
  <worldbody>
    <geom type="plane" size="1 1 0.1"/>
    <body pos="0 0 0.5">
      <freejoint/>
      <geom type="sphere" size="0.05" mass="0.1"/>
    </body>
  </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)
for _ in range(100):
    mujoco.mj_step(model, data)
print(data.time, data.qpos[:3])
```

Use `MjModel.from_xml_string` for XML text and `MjModel.from_xml_path` for a file path; they are not interchangeable.

### Viewer and headless rendering

```bash
python -m mujoco.viewer --mjcf=path/to/scene.xml
```

According to the [official Python guide](https://mujoco.readthedocs.io/en/stable/python.html), rendering needs a current OpenGL context. On Linux, the documented choices include GLX for an X11 window, EGL for hardware-accelerated headless rendering, and OSMesa for software headless rendering. On macOS, passive viewing requires the installed `mjpython` launcher. Select and record the backend; do not describe a viewer initialization call as physics GPU acceleration.

### Promotion gates

1. XML compiles with no asset-path warnings.
2. Simulation time and state advance under `mj_step`.
3. Renderer produces a frame in the intended interactive or headless mode.
4. The task test records seed, contacts, termination reason, and metric.
5. A real robot remains a separate hardware gate.

## 中文指南

使用 `python -m pip install mujoco` 安装官方维护的 Python 绑定；该软件包已包含 MuJoCo 库，本路径不需要手动下载旧版二进制，也不建议新项目继续使用 `mujoco-py`。

XML 字符串必须传给 `MjModel.from_xml_string`，文件路径才传给 `MjModel.from_xml_path`。最小验收应至少证明模型编译、`mj_step` 推进时间与状态、渲染后端产生画面，以及任务指标有固定种子和终止原因。

Linux 窗口渲染、硬件离屏渲染和软件离屏渲染分别涉及 GLX、EGL 和 OSMesa；macOS 的被动 Viewer 需要 `mjpython`。这些是渲染上下文选择，不等于 MuJoCo 物理计算被 GPU 加速。完整的场景建模、接触力、传感器与 XML 导出见[双语场景搭建教程](../tutorials/mujoco-scene-building.md)。

权威来源：[MuJoCo Python](https://mujoco.readthedocs.io/en/stable/python.html) · [Visualization](https://mujoco.readthedocs.io/en/stable/programming/visualization.html)
