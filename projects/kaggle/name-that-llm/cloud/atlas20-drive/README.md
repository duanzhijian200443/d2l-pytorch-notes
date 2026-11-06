# ATLAS20 的 Drive 输入适配

入口是 [notebooks/name-that-llm-atlas20-repro.ipynb](../../notebooks/name-that-llm-atlas20-repro.ipynb)。在已连接的 Colab 内核中依次运行顶部 **0.1 挂载**、**0.2 输入检查**，再运行原来的 **1. Inputs**。

默认 `FULL_RUN=False`：只检查和准备输入，不加载模型、不自动安装依赖，不生成占位提交文件。输入清单位于 `drive-inputs.json`，由 `build_manifest.py` 根据已核查的 Model-Library 目录和既有缓存清单生成。`drive_inputs.py` 在运行时的 `/content/atlas20-drive/input` 下建立文件链接，保留原算法依赖的目录名。原始 Drive 文件不移动、不覆盖，权重不经过本地电脑。

整理时移出的模型、LoRA、编码器和分类头优先使用新目录；尚未移动的数据和 Notebook 附属文件仍从 `name-that-llm/v32-colab/cli-cache` 读取。缺少文件、权重索引引用的分片缺失、三个已固定历史 CSV 哈希不一致时立即停止。并未重新计算所有大权重的内容哈希，也未证明现有 Drive 缓存与 EXP-004 正式提交的每个输入字节相同。

原始文件及用户已保存的挂载单元格备份在 `reviews/atlas20-drive-20261101/before-drive-adaptation.ipynb`。算法脚本的变动仅是输入/工作目录路径；外层启动代码增加输入检查模式，并让子进程错误直接抛出。完整 GPU 推理、环境兼容性和运行结束后的释放流程不在此次已执行验证范围内。

本地验证：`python -m unittest discover -s cloud/atlas20-drive -p test_drive_inputs.py -v`。文件符号链接测试应在 Linux/WSL 执行。`build_notebook.py` 是一次性的有保护替换脚本，只接受与备份相同的输入，不能用来覆盖用户后续修改。
