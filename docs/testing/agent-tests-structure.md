# 单元测试结构说明

`agents/tests` 目录采用标准分层：

- `unit/`：可执行单元测试用例（`test_*.py`）
- `utils/`：测试共用工具（路径引导、导入桩等）

当前主要单测文件：

- `test_agent_route_metrics.py`
- `test_bot_robustness.py`
- `test_eval_core.py`
- `test_eval_dataset_loader.py`
- `test_tools_structured_output.py`

运行全部单元测试：

```bash
python -m unittest discover -s agents/tests -p "test_*.py" -v
```
