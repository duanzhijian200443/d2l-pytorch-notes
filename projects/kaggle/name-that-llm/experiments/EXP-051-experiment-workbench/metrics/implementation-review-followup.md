# EXP-051 修复复审

**PASS_SCOPED_CODE_REVIEW：在已审查范围内可交接，没有未解决代码发现。** 原始 implementation-review.json/md 保留。

- R1已关闭：独立 confirmed-score 指针在失败或缺分后保留；list/report复用同一绑定校验逻辑。
- R2已关闭：本地数值必须有成功verify及当前summary哈希绑定，失败或改动时不显示未经核验数字。
- R3已关闭：新只读adapter调用冻结候选验证，不调用旧收据写入函数。我独立执行adapter后，原package receipt的SHA256和mtime_ns均未改变。
- 追加检查通过：EXP049/050 registry所有声明依赖哈希匹配；启动前依赖核验；每EXP操作锁；EXP048新增数据PASS/120行/60组/holdout=0/hash门槛。

## 本地验证

21项单元测试通过，其中9项为独立新增的src/test_workbench_review.py：确认分数在list显示、同EXP重入锁拒绝并正常释放、依赖被改时child和observation均未创建、add120必要字段逐项失败关闭，以及本地子进程超时/非零退出均不产生PASS。仅使用临时目录或模拟调用。未改产品代码、台账、历史结果，未联网、开GPU或提交。

## 交接边界

本复审不替代根窗口正在进行的真实只读status/collect/check验收；真实远端结果须另附证据。没有验证hidden输出、模型训练、OOF或提分。HTTP重定向/分页/超时尚未做完整故障注入，现有测试覆盖不安全host与超限/截断流。固定本地脚本白名单和subprocess限额不是OS隔离沙箱。SDK会先解析列表与日志响应，再执行本地日志限额，限制的是后续处理与保存，不是SDK上游内存分配。

精确审查文件哈希见implementation-review-followup.json。

## 最后展示改动复查

新增正式分数差、EXP049 macro recall差与EXP050重合计数展示均通过。只有双方已有确认分数才显示正式差值，PENDING不冒算。EXP049显示+0.22238个百分点并保留缓存/非OOF/区间跨0限制。EXP050显示190588总量、97870重合、92718未命中，并保留非训练准入限制。两项本地数值仍受成功verify和当前summary哈希门槛保护；新增3项临时测试覆盖这些分支及篡改摘要抑制。

本地读取根窗口真实观察收据并重查下载文件哈希：EXP046正式COMPLETE/0.663；EXP048正式PENDING、commit COMPLETE；两者均PASS_COMMIT_SAMPLE_ONLY。这是已保存证据复核，我没有发网络请求；仍不代表完整hidden验收。产品及测试文件最终哈希已更新。
