# FreqTrade project Workflow Guide

## Structure

```
/a0/usr/projects/freqtrade/
├── freqtrade/           # FreqTrade core source code
├── user_data/           # Your custom data
│   ├── strategies/      # Strategy files (*.py)
│   ├── data/{exchange_name}/futures    # Historical OHLCV data
│   ├── hyperopts/       # Hyperopt configurations
│   ├── notebooks/       # Jupyter notebooks
│   └── backtest_results/ # Backtest output
├── config_examples/     # Config templates
├── docs/                # Documentation
└── tests/               # Test suite
```
