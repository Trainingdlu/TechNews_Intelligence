# 单元测试结构说明

`tests/` 目录采用分层组织：

- `tests/unit/`：核心单元测试（`test_*.py`）
- `tests/utils/`：测试辅助工具（路径引导、stub、公共 fixture）

当前重点测试文件：

- `tests/unit/test_agent_route_metrics.py`
- `tests/unit/test_bot_robustness.py`
- `tests/unit/test_eval_core.py`
- `tests/unit/test_eval_dataset_loader.py`
- `tests/unit/test_tools_structured_output.py`

常用命令：

```bash
# 运行全部单元测试
pytest tests -v

# 快速回归关键模块
pytest tests/unit/test_agent_route_metrics.py -v
pytest tests/unit/test_bot_robustness.py -v
pytest tests/unit/test_eval_core.py -v
```
