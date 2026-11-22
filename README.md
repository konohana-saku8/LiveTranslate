# LiveTranslate（个人维护分支）

Windows 实时音频翻译：捕获系统声音（可选混入麦克风），VAD 切句后在本地跑 ASR，
再交给任意 OpenAI 兼容接口流式翻译，结果显示在透明置顶悬浮窗。

本仓库是 [TheDeathDragon/LiveTranslate](https://github.com/TheDeathDragon/LiveTranslate)
的个人维护分支。安装方式、模型说明、翻译 API 配置、主题等完整文档以原仓库为准：
<https://github.com/TheDeathDragon/LiveTranslate#readme>

## 这个分支新增的功能

| 功能 | 说明 |
| --- | --- |
| 手机远程听译 | 电脑端提供 WebSocket 服务，手机把其他应用的声音推过来，复用电脑的 ASR 与翻译，结果同时回到手机悬浮窗。客户端见 [LiveTranslateClient](https://github.com/hibane-hinome/LiveTranslateClient) |
| 逐句落库 | 每句的原文、译文、目标语言，以及开口时刻、时长、链路延迟写进 SQLite，供事后整理成中日对照逐字稿（[live-archive](https://github.com/hibane-hinome/live-archive)） |
| ReazonSpeech | 新增日语离线 ASR 引擎 |
| Qwen3-ASR 1.7B | 新增多语言 ASR 选项 |
| 调试库隔离 | 面板勾选 Debug 后写入独立调试数据库，不动正式历史 |
| 失败也入库 | 翻译失败时保留 ASR 原文与失败标记，便于事后校准 |
| 手机连接状态 | 悬浮窗标题栏显示远程服务状态：未启动 / 等待连接 / 手机已连接 |

## 快速开始

1. 双击 `install.bat`：检测 Python、创建虚拟环境、按显卡安装 CUDA 或 CPU 版 PyTorch。
2. 双击 `start.bat` 启动。
3. 首次向导选择模型下载源（ModelScope / HuggingFace）与缓存目录，等待 VAD 与 ASR 模型就绪。
4. 在控制面板「翻译」页填写 API Base、API Key 与模型。
5. 点「运行」，悬浮窗开始显示原文与译文。

手机端：

1. 安装 [LiveTranslateClient](https://github.com/hibane-hinome/LiveTranslateClient)。
2. 在电脑悬浮窗标题栏点「启动服务」，记下端口（默认 `8766`）。
3. 手机设置页填电脑的局域网 IP、端口，令牌可留空；点「连接」，再点「开始」选择要采集声音的应用。

同一时间只接受一台手机；电脑与手机需在同一局域网。

## 上游功能

系统音频捕获、麦克风混音、多 ASR 引擎、远程 Whisper（见 [REMOTE_ASR.md](REMOTE_ASR.md)）、
流式翻译、模型独立配置、内置基准测试、14 种悬浮窗主题等均来自上游。
完整文档、预编译包与更新日志请访问原仓库。

## 许可

[MIT License](LICENSE)，与原项目一致。
