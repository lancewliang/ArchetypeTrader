# Phase I fixture 数据

集成测试运行时通过 `tests/fixtures/phase1/build_fixtures.py` 现场生成小型 Feather:

- `market_train.feather`
- `market_val.feather`
- `market_test.feather`

**为什么不签入二进制 fixture**: feather 不便代码审查；用脚本即时生成可保留时间序列、close、五档盘口、衍生因子等结构，且生成逻辑本身可以单元测试。

如需手工生成，运行:

```bash
python tests/fixtures/phase1/build_fixtures.py --out tests/fixtures/phase1
```

集成测试会在 `tmp_path` 下重新生成；本目录的文件仅供本地调试。
