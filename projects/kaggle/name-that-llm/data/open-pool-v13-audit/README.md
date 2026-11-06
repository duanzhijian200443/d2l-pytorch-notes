# EXP050固定数据原件

本目录只保存只读审计所需的固定Kaggle版本原件，不是已批准训练池。

- `v13/`：Open Pool平台V13主parquet、prompt表和Gemini3.7隔离表；下载收据固定版本、字节、SHA256。
- `baseline-prvs-v7/`：与EXP031/046实际运行hash匹配的prvs平台V7 clean CSV。
- `baseline-prvs-v6/`：最初按附件请求版本V6下载的对照原件；与实际运行hash不符，保留作版本差异证据，未用于实际10469共享EXTRA重建。
- 既有数据优先复用`data/public-pools/`和A1固定extra原件，未覆盖旧文件。

审计报告、全部输入清单和逐行风险见`experiments/EXP-050-open-pool-v13-increment-audit/`。脚本不读取或下载模型权重，不训练、不生成、不提交；没有保存临时签名下载地址或凭据。
