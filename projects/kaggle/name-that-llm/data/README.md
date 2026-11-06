# 数据登记与审计入口

**增量判断须绑定具体输入字节。** [EXP-009 基线用数清单](../experiments/EXP-009-baseline-data-lineage/report.md)保留旧固定输入的投影：EXP-008 的15,989条是诊断候选，其中14,604条已命中当时投影的运行时训练表，4,158条prvs clean/new全部已用，未命中的1,385条仍可能被现成权重见过。Open Pool V13相对EXP031/046实际运行输入的逐行判断见[EXP-050审计](../experiments/EXP-050-open-pool-v13-increment-audit/report.md)，不能以旧投影的10325行替代实际10469行共享EXTRA。

- [当前公共池审计](public-pool-audits.json)：EXP-008 对六个固定版本来源逐行检查后的结果与使用建议。
- [公共池检查报告](../experiments/EXP-008-public-pool-audit/report.md)：风险、重叠、来源可信度和下一轮对照建议。
- [最初来源登记](source-inventory.json)：EXP-007 的历史登记快照，反映当时只登记、尚未读取公共原文；不作为最新审计结果。
- [固定验证](../validation/README.md)：prompt-group-v1、样本哈希和 fold 规则。
- [Open Pool 平台V13来源核查（SRC-065）](../research/datasets/SRC-065-open-pool-kaggle-v13/notes.md)：保留作者数据卡、固定版本和最初元数据核查；后续自己的逐行结果单独关联EXP-050。上游筛选已参考全部官方训练集，仅按折重算匹配权重不足以恢复严格OOF独立性。
- [Open Pool V13实际逐行审计（EXP-050）](../experiments/EXP-050-open-pool-v13-increment-audit/report.md)：实读190588行，与A1实际运行用数联合重合97870，未命中正文上界92718；这不是所有checkpoint都未见过的证明。C1恢复prompt增量42143条、C2进一步排除已知运行时prompt关联后29478条，均仍为候选、未批准训练。完整原件见`open-pool-v13-audit/`；逐行风险与候选索引见该EXP的`metrics/`。
- **prvs实际运行字节修正：** EXP031/046附件请求写平台V6，但运行manifest的7519791字节/hash `502653b8…`实际匹配平台V7；EXP-050按该哈希重建10469行/716组，和原loader逐行一致。平台V6原件仍保留，旧10325投影没有被覆盖；详见[版本差异回执](../experiments/EXP-050-open-pool-v13-increment-audit/download-version-discrepancy.json)。
- [Lucas 平台V12核查（SRC-066）](../research/datasets/SRC-066-lucas-free-proxy-v12/notes.md)：原始CSV保存在 `community-factcheck-20261102/lucas-v11/` 与 `community-factcheck-20261102/lucas-v12/`；V12较V11新增192行。已核对规模和版本差异，训练准入及与其他池的净增量尚未确认。
- [固定候选的四类外部检查](../experiments/EXP-026-independent-blend-validation/report.md)：SRC-062 V3经known-ancestry/exact隔离，保留Dolly-only 5200条/1300组；仅支持四类来源上的结论。该选集已在EXP-026评分，后续调参不得再将它称为未触碰验证集，见[使用登记](../experiments/EXP-026-independent-blend-validation/metrics/data-use.json)。

原始公共文件保存在 `public-pools/owner/dataset/vN/`，不提交 Git。清洗或筛查产物另外保存，不覆盖原文件。原始版、clean 版、汇总版是重叠视图，不能相加作为独立新增样本数。
