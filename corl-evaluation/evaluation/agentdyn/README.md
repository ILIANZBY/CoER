# Main Evaluation Runtime

本包实现 Official1514 主实验：157 个 clean case、1,189 个 adaptive case，以及从 attacked case 中固定选出的 42 个 case × 4 个静态模板。

- `build_official_overlap.py`：生成固定 parquet 与两个 panel。
- `defender_eval.py`：执行 clean、fixed 和 adaptive 评测。
- `configs/main_clean_fixed.yaml`：clean utility 与 fixed attack。
- `configs/main_adaptive.yaml`：使用冻结的已配置 attacker 执行 adaptive attack。
- `report.py`：单次运行汇总；跨两种设置的表格由 `../../scripts/report_main.py` 生成。

运行入口与模型环境变量见项目根目录的 `corl-evaluation/README.md`。评测会记录数据、panel、代码和 endpoint 元数据的 fingerprint，以保证恢复运行时不会混合不同协议或 checkpoint。
