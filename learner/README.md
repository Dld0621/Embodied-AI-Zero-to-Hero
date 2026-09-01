# Learner workspace / 学习者工作区

Copy `progress.example.json` to a personal, normally uncommitted progress file and keep evidence outside the repository or in a dedicated portfolio repository. Do not commit secrets, private datasets, personal information, or proprietary robot logs.

将 `progress.example.json` 复制为个人进度文件。证据建议保存在独立作品集仓库或本仓库之外。不要提交密钥、私有数据、个人信息或专有机器人日志。

## Templates / 模板

- [`experiment-card.md`](templates/experiment-card.md): freeze the question, protocol, and evidence identity before running.
- [`failure-report.md`](templates/failure-report.md): retain a failed result and isolate the next discriminating test.
- [`capstone-review.md`](templates/capstone-review.md): score a capstone and record independent review.

Validate a progress record with:

```bash
python scripts/run_curriculum.py --report-progress learner/progress.json
```

The validator checks record structure and local evidence paths. It does not assess scientific correctness or grant hardware authorization.
