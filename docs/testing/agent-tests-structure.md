# 单元测试结构说明

`agents/tests` 目录采用分层组织：

- `unit/`：核心单元测试文件（`test_*.py`）
- `utils/`：测试辅助工具（路径引导、公共 fixture 等）

当前主要测试文件：

- `test_agent_route_metrics.py`
- `test_bot_robustness.py`
- `test_eval_core.py`
- `test_eval_dataset_loader.py`
- `test_tools_structured_output.py`

运行全部单元测试：

```bash
pytest agents/tests -v
```

按文件快速回归：

```bash
pytest agents/tests/unit/test_bot_robustness.py -v
pytest agents/tests/unit/test_eval_core.py -v
```
