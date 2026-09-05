# 贡献指南

> 感谢你对 Embodied AI Zero-to-Hero 项目的关注！本指南帮助你高效地参与贡献。

## 项目定位

本项目是一个**以文档和教程为核心的具身智能知识库**，目标受众从大一新生到前沿研究者。因此，贡献内容需要兼顾**准确性**、**可读性**和**可运行性**。

---

## 如何贡献

### 1. 报告问题（Issue）

发现内容错误、链接失效或代码 bug？请提交 Issue 并包含：

- **问题类型**：`[bug]` / `[doc]` / `[link]` / `[suggestion]`
- **位置**：文件路径 + 行号/章节
- **当前内容**：复制错误片段
- **预期内容**：你认为正确的版本
- **依据**：论文链接、官方文档或实验结果

示例：
```
[link] README.md 第 157 行 DIAMOND 仓库链接失效

当前：https://github.com/ethz-rl/diamond（404）
预期：https://github.com/eloialonso/diamond
依据：论文作者 Vincent Micheli 的个人仓库
```

### 2. 提交代码/文档改进（Pull Request）

#### 前置检查

```bash
# 1. 确保示例代码可运行
cd examples
python freshman_zero_to_one.py --gesture open --model shadow

# 2. 检查新增依赖是否已记录
# 修改 setup/environment.yml 和 requirements.txt

# 3. 运行基础导入测试
python -m pytest tests/ -v
```

#### PR 规范

| 项目 | 要求 |
|------|------|
| **分支命名** | `fix/xxx`（修复）、`feat/xxx`（新功能）、`doc/xxx`（文档） |
| **Commit 消息** | 遵循 Conventional Commits：`docs:`、`fix:`、`feat:`、`refactor:` |
| **变更范围** | 一个 PR 只解决一个问题，避免大规模混合变更 |
| **文档同步** | 修改代码后，同步更新对应文档中的描述、参数说明和运行示例 |

#### Commit 消息示例

```
docs: 修正 IRIS 官方仓库链接（janner/iris -> eloialonso/iris）

IRIS 论文原始代码由 Vincent Micheli 维护，
仓库位于 eloialonso/iris，而非 janner/iris。

修复文件：
- README.md
- docs/07-world-models-for-vla.md
- docs/03-learning-path.md
```

---

## 内容质量标准

### 论文引用规范

- **必须提供 arXiv 或官方会议链接**
- **作者和机构信息需与论文一致**
- **开源代码链接需手动验证**（点击确认可访问）
- **不引用未公开或无法验证的内容**

### 代码示例规范

- **自包含优先**：示例应尽量不依赖外部文件，或提供自动下载脚本
- **依赖检查**：在 `__main__` 或函数开头检查关键依赖，给出友好提示
- **错误处理**：文件 I/O、网络请求、模型加载必须有 try/except
- **类型提示**：新代码建议添加 Python 类型注解
- **文档字符串**：每个公共函数/类必须有 docstring（Args/Returns）

### 文档撰写规范

- **术语统一**：首次出现缩写需给出全称（如 VLA (Vision-Language-Action)）
- **公式可复现**：关键公式需注明符号含义，尽量提供对应代码链接
- **公式兼容 GitHub**：行内公式使用 `$...$`，块公式使用独立行的 `$$...$$`；不要使用 `\\(...\\)` 或 `\\[...\\]`；中文标点、全角括号或汉字后开始行内公式时，在 `$` 前保留一个半角空格；单行块公式前后保留空行
- **分层表达**：同一概念提供"一句话直觉 + 技术细节 + 代码示例"三层描述
- **引用锚定**：引用外部资源时给出具体章节/页码，而非仅链接

提交前运行 `python scripts/check_markdown_format.py`。该检查会忽略代码块与行内代码，并阻止无法在 GitHub 正常渲染的公式分隔符或裸露 TeX 命令进入仓库。

美元价格使用 `USD 200–300` 或转义美元符号，避免价格范围被当成数学表达式。文档站采用静态公式缓存：修改公式后先运行 `npm ci` 和 `python scripts/generate_math_cache.py`，把更新后的 `generated/math-cache.json` 一并提交，再运行 `python -m mkdocs build --strict --clean` 和 `python scripts/check_site_math.py`。普通文档构建与读者浏览不需要 Node.js 或数学 CDN；GitHub 源码预览仍使用原始 Markdown 数学语法。

标题中使用普通文字或 `λ` 等 Unicode 符号，不嵌入 TeX 公式：目录会剥离公式外层标记，把代码直接显示出来。推导公式放在正文中。

---

## 特定领域贡献建议

### 补充前沿论文

2026 年具身智能发展迅速，欢迎补充新论文：

1. 在 `docs/16-arxiv-retargeting-scan.md` 添加扫描记录
2. 在 `docs/18-frontier-papers-online.md` 补充在线链接
3. 在 `docs/17-research-trends-and-positioning.md` 分析研究转向意义
4. 更新 README.md 的"2026 前沿研究亮点"区块

### 新增教程阶段

`tutorials/` 目录的每个子目录应包含：

```
tutorials/XX-topic-name/
├── README.md              # 概念讲解 + 步骤说明
├── code_demo.py           # 可独立运行的示例代码（可选但推荐）
└── assets/                # 示意图或数据文件（可选）
```

### 新增示例代码

`examples/` 目录的新代码应满足：

- 文件头包含模块 docstring（功能、依赖、运行命令）
- 支持命令行参数解析（`argparse`）
- 提供 `--help` 输出
- 在 README.md 对应支柱表格中注册

### 修改科研路线或 Pipeline

- 知识节点、Pipeline 和科研路线的唯一数据源依次是 `knowledge/manifest.json`、`pipelines/manifest.json` 和 `learning_paths/manifest.json`
- 新增知识节点必须声明前置节点、所在阶段、双语标题/摘要/产出/考核、对应文档、学习证据类型与关联 Pipeline
- 前置关系必须是无环图；后续节点不能反向依赖更高阶段，文档必须指向仓库内已存在的可审查内容
- 新增 Pipeline 时，必须把它定位到至少一条科研路线，并同步中英文路线文档
- 路线必须声明交付物、指标、晋级门槛和证据边界；不能因文档接入而提高 Pipeline 的证据等级
- 提交前运行 `python scripts/run_knowledge_map.py --validate`、`python scripts/run_pipeline.py --validate`、`python scripts/run_learning_path.py --validate` 和 `python scripts/audit_repository.py`

### 修改从小白到专家的课程合同

- `curriculum/manifest.json` 是 L0–L5、M00–M11、目标路线和 Capstone 的机器可读事实源；每个知识节点必须且只能映射到一个课程模块
- 新模块或门禁必须声明双语标题、学习产物、评审条件、预计工作量和真实存在的主文档
- `curriculum/quality_rubric.json` 的分数必须同时给出缺口、实现证据和边界；不能只修改数字来维持 100 分
- 评估与 Capstone 变更必须同步中英文页面，并保留关键失败、独立评审和真机授权边界
- 学习者示例应使用 [`learner/templates/`](learner/templates/) 中的实验卡、失败报告或评审表，不提交密钥、私有数据或专有机器人日志
- 提交前运行 `python scripts/run_curriculum.py --validate` 和 `python -m pytest tests/test_curriculum_journey.py -q`

---

## 审查清单（Reviewer Checklist）

维护者在合并 PR 前将检查：

- [ ] 外部链接已手动验证可访问
- [ ] 新增/修改的代码可在干净环境中运行
- [ ] 文档与代码描述一致
- [ ] 知识节点的前置关系、双语字段、产出和考核已同步
- [ ] 术语和作者信息准确
- [ ] 不引入未声明的新依赖
- [ ] Commit 历史清晰、可回滚

---

## 社区行为准则

- 尊重不同背景的学习者，避免假设对方已掌握特定前置知识
- 技术讨论聚焦问题本身，不评价个人
- 承认知识边界：不确定的内容明确标注"待验证"或"社区补充"

---

## 联系方式

- **Issue 讨论**：GitHub Issues（推荐，便于归档检索）
- **紧急链接修复**：可直接提 PR，标题前缀 `[urgent]`

> 本项目采用 MIT 许可证。贡献即表示你同意将提交的内容按 MIT 许可证授权。
