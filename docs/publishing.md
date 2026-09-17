# GitHub / Hugging Face 发布与双盲审查

核对日期：2026-09-07。当前状态：**仅做本地准备，未创建远程仓库、未上传、未公开。** 自动扫描不能担保匿名性；最终提交材料、账号和访问方式仍须由作者复核。

## 官方要求与本项目的保守选择

ICLR 2027 要求正文和补充材料匿名，暴露作者身份可能导致 desk rejection。官方鼓励补充代码，接受匿名 ZIP 或匿名仓库链接；未要求投稿时必须把全部模型权重上传到 Hugging Face。论文和补充材料的截止时间相同。允许 arXiv 预印本并不意味着可以在匿名提交材料中放入有身份线索的资源链接。[ICLR 2027 Author Guidelines](https://iclr.cc/Conferences/2027/AuthorGuidelines)

**本项目采用的方案：双盲期间以匿名代码 ZIP 为审稿入口；模型先保留本地，或仅作私有备份。待会议实际去匿名后，再从审查过的发布副本公开 GitHub 和 Hugging Face。** 这是降低风险的工作方案，不是声称 ICLR 一律禁止提前公开模型。若审稿期必须提供外部模型下载，先向 Program Chairs 确认具体匿名访问方案；不要让审稿人实名申请访问。[ICLR 联系入口](https://iclr.cc/Conferences/2027/CallForPapers)

| 使用场景 | 处理方式 |
|---|---|
| 审稿代码 | 优先使用本地匿名导出的 ZIP；也可使用经检查的匿名仓库 |
| 审稿期模型备份 | 本地或 private；不把不可访问的私有链接当成审稿资源 |
| 审稿期必须下载权重 | 单独审查账号、链接、历史和访问隐私；必要时先取得会议确认 |
| 正式去匿名后 | 经授权和安全审查后，发布正式代码、模型卡、许可证和复现说明 |

## GitHub：不能直接推送当前工作目录

只从 `scripts/export_anonymous.py` 的导出物建立新的发布仓库，不带当前 `.git`、旧远程、分支、标签和提交历史；不要直接 `git add .` 上传整个工作区。`local_artifacts/remote-source/` 是带内部配置的原始快照，**不是匿名发布源**。

匿名仓库名称不能替代匿名账号。需要核查账号主页、用户名、组织成员、头像、邮箱、commit author/committer、签名、贡献者、PR、Issues、Actions 日志和外部链接。个人账号的 noreply 邮箱也不是匿名身份。[GitHub commit email 文档](https://docs.github.com/en/account-and-profile/how-tos/email-preferences/setting-your-commit-email-address)

先公开再删除不可靠：旧内容可能留在 forks、clones 或缓存中，因此不要把公开上传当成匿名性测试。[GitHub 敏感数据清理说明](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository)

## Hugging Face：private、gated 和匿名是不同概念

- `private` 是权限控制。未获权限的人无法查看或克隆，适合作为备份，但不是可复现的匿名审稿入口。[HF repository settings](https://huggingface.co/docs/hub/repositories-settings)
- `gated` 会向模型作者提供申请者的用户名、邮箱等信息，包括自动批准模式。不要让审稿人通过这种机制获取模型。[HF gated models](https://huggingface.co/docs/hub/models-gated)
- 匿名账号名本身不能保证匿名：还要查模型卡、上传者、提交历史、关联论文、组织、讨论、跨站链接与访问统计。不要收集审稿人的身份信息或要求发邮件索取权重。

对于 attacker，如果安全策略要求人工审核访问，不应为了“双盲”而直接取消安全限制；改为保留私有，或使用会议认可的替代审查方式。

## 两个模型的文件级发布检查

发布副本与原始 checkpoint 分开保存。只保留 safetensors、必需配置、分词器、模板、模型卡及许可证。不上传优化器、随机数状态、训练参数 pickle、训练日志、服务地址和原始轨迹。

检查 JSON 中包括嵌套对象在内的 `_name_or_path`、缓存路径、内部域名、IP、账号和关联仓库。当前 attacker 配置曾含内部训练路径，不能直接上传原目录。清理仅发生在发布副本；权重、特殊 token、词表和生成参数不得为了消除字符串匹配而被修改。公开词表中的人名或组织词汇不等于作者信息。

检查 safetensors header 的 metadata；保留逐文件哈希和权重索引验证。文件级去标识化不能证明训练数据不存在隐私问题或模型不会记忆数据，还需要数据授权、隐私和安全评估。

模型卡应说明 Base 模型、训练阶段、数据来源和筛选方式、能力边界、评测口径、风险、加载依赖及真实验证范围。不要将论文转录结果标为重新复现的结果，也不要把混合攻击者的 OOD 结果称为纯 GPT 设置。[HF model cards](https://huggingface.co/docs/hub/model-cards)

## 许可证、授权和论文声明

Qwen3.5-9B 的公开上游许可证为 Apache-2.0；必须保留适用的许可证与通知，并说明修改。它不自动解决本项目新增权重、训练数据、teacher 输出或组织资产的发布权限。作者需确认全部必要授权，并批准新增成果采用的许可证。[Qwen3.5-9B LICENSE](https://huggingface.co/Qwen/Qwen3.5-9B/blob/main/LICENSE)

同样，不应删除 verl、AgentDojo 等第三方必须保留的版权信息来“匿名”。上游权利人名称不是本论文作者署名；见 `THIRD_PARTY.md`。如果某项必须保留的项目自有权属信息会直接暴露作者，应先解决授权/提交方式，不能伪造权利人。

ICLR 2027 还要求在论文专门章节及提交表单披露 AI 使用；本次代码整理和文档辅助应纳入真实披露，由作者核查并负责。[ICLR AI Policy for Authors](https://iclr.cc/Conferences/2027/AIPolicyForAuthors)

## 上传前必须确认

### 2026-09-17 上传准备状态

论文与 README 已更新为 **CoER**；保留 `corl` 代码路径和命令别名以兼容既有脚本。新的发布包应从当前源码重新导出，不能上传先前包含旧版指标的 ZIP。

用户计划将代码放到 GitHub，并在 Hugging Face 分别托管 attacker、defender、Attacker SFT 数据、Defender SFT 数据和 RL 数据。目标账号/组织尚待提供；先按私有仓库准备，不因本次请求而取消此前的双盲限制。后续公开仍需单独确认。

两个 checkpoint 已在本地。三份原始训练数据尚未取回核验；论文给出的数量是预期验收条件，不应据此生成、补齐或推断缺失记录。RL 数据应保持 12,705 条 train 与 3,186 条 validation 的独立分割。数据需逐项检查来源/授权、个人信息、内部路径、凭据和样本元数据，保留必要的合成环境内容及可复现标识，不上传未经审查的原始轨迹或内部采集日志。

### 最终检查

- [ ] 指定 GitHub / Hugging Face 目标账号、仓库名称及可见性；不要在聊天或仓库写入 token。
- [ ] 确认代码、两个 checkpoint、训练数据与 teacher 输出的必要发布权限。
- [ ] 确认项目新增代码/权重的许可证，并保留全部适用上游条款。
- [ ] 复核最终导出文件、两张图、模型配置、词表处理方式及所有外部链接。
- [ ] 检查账号和历史不会暴露作者，审稿访问不需要实名申请、不收集审稿人身份。
- [ ] 确认论文 AI use statement 与 Reproducibility Statement 准确说明已提供和未提供的资源。
- [ ] 用干净环境完成实际加载/推理与关键训练、评测 smoke tests；当前只完成离线文件校验和 CPU 测试。
- [ ] 上线前再次检查当届官方政策；如匿名模型分发方式不明确，先咨询会议，不先公开再补救。

在这些检查未完成前，发布副本不应被称作已批准公开的资源。私有备份也需要先确认目标账号，不能默认使用当前登录的个人或组织账号。
