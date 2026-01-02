# Start dry-run
freqtrade trade --logfile ./logs --strategy Aroon2Pctv0 --config user_data/config.json 

# Run backtest
freqtrade backtesting --logfile user_data/logs/backtest.log --strategy-list  Aroon2Pct --config user_data/config.json  --timerange 20250801-20251221 --fee 0.00055  -i 15m

# Run backtest for FReqAI enabled strategy
freqtrade backtesting --strategy qav3 --config user_data/QuickAdapterV3.json --freqaimodel XGBoostRegressorMultiTarget --timerange 20230925-20230930  -i 5m --fee 0.001

# Run hyperopt
freqtrade hyperopt --hyperopt-loss SharpeHyperOptLossDaily --spaces buy --strategy Aroon2Pct --config user_data/backtest.json -e 200 --timerange 20250801-20251119 --fee 0.00055 -j 6