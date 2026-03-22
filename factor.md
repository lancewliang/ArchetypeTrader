
'ask1_price',
'ask1_size',
'bid1_price', 
'bid1_size',
'ask2_price',
'ask2_size',
'bid2_price',
'bid2_size', 
'ask3_price',
'ask3_size', 
'bid3_price', 
'bid3_size', 
'ask4_price', 
'ask4_size',
'bid4_price', 
'bid4_size', 
'ask5_price', 
'ask5_size', 
'bid5_price', 
'bid5_size', 

'bid1_size_n', 
'bid2_size_n', 
'bid3_size_n',
'bid4_size_n', 
'bid5_size_n', 
'ask1_size_n', 
'ask2_size_n',
'ask3_size_n', 
'ask4_size_n', 
'ask5_size_n', 
'wap_1',   wap_1 = (bid1_price × bid1_size + ask1_price × ask1_size) / (bid1_size + ask1_size)
'wap_2',   wap_2 = (∑_{i=1}^{n} bid_i_price × bid_i_size + ∑_{i=1}^{n} ask_i_price × ask_i_size) 
        / (∑_{i=1}^{n} bid_i_size + ∑_{i=1}^{n} ask_i_size)
'wap_balance', 

'buy_spread', 买卖价差
'sell_spread', 买卖价差
'price_spread',  价格价差

'buy_volume',  买卖成交量
'sell_volume',买卖成交量
'volume_imbalance',    成交量不平衡度 


WAP (Weighted Average Price) - 加权平均价格 WAP = (∑(价格_i × 数量_i)) / ∑数量_i
VWAP (Volume Weighted Average Price) - 成交量加权平均价格 VWAP = (∑(价格_i × 成交量_i)) / ∑成交量_i  

'sell_vwap', 
'buy_vwap',
'log_return_wap_1',
'log_return_wap_2', 


'log_return_bid1_price', 
'log_return_bid2_price',
'log_return_ask1_price', 
'log_return_ask2_price',



'buy_vwap_trend_60', 60期趋势指标
'sell_vwap_trend_60',60期趋势指标
'buy_spread_trend_60',60期趋势指标
'sell_spread_trend_60',60期趋势指标
'ask1_price_trend_60', 60期趋势指标
'bid1_price_trend_60',60期趋势指标

'wap_1_trend_60',60期趋势指标
'wap_2_trend_60', 60期趋势指标


============================================================================================
OHLCV（时间、开盘价、最高价、最低价、收盘价、成交量）
'timestamp', 
'volume', 
'open', 
'high', 
'low', 
'close', 
'volume_trend_60', 60期趋势指标


K线形态
'kmid',  K线中间价位置 kmid = (close - open) / (high - low + ε)
'klen',K线长度（相对波动）比例 klen = (high - low) / (open + ε)
'kup',  上影线比例 kup = (high - max(open, close)) / (high - low + ε)
'klow',  下影线比例 klow = (min(open, close) - low) / (high - low + ε)
'ksft',  K线实体偏移度 ksft = (max(open, close) - (high + low)/2) / (high - low + ε)

============================================================================================

平方项特征（增强非线性关系）
'kmid2', 
'kup2', 
'klow2', 
'ksft2'
==============================================================================================
['slope_360', 'vol_360']   decomposition 通过close计算而来
============================================================================================
形态	kmid	kup	klow	ksft	市场含义
大阳线	>0.7	小	小	>0.3	强烈看涨
大阴线	<-0.7	小	小	<-0.3	强烈看跌
十字星	≈0	≈0.5	≈0.5	≈0	市场犹豫
锤子线	>0	小	>0.7	<0	底部反转
射击之星	<0	>0.7	小	>0	顶部反转
纺锤线	≈0	≈0.3	≈0.3	≈0	平衡市
===============================================================================================

single_features



['volume', 'bid1_size_n', 'bid2_size_n', 'bid3_size_n', 'bid4_size_n', 'bid5_size_n', 'ask1_size_n', 'ask2_size_n', 'ask3_size_n', 'ask4_size_n', 'ask5_size_n', 'wap_1', 'wap_2', 'wap_balance', 'buy_spread', 'sell_spread', 'buy_volume', 'sell_volume', 'volume_imbalance', 'price_spread', 'sell_vwap', 'buy_vwap', 'log_return_bid1_price', 'log_return_bid2_price', 'log_return_ask1_price', 'log_return_ask2_price', 'log_return_wap_1', 'kmid', 'klen', 'kmid2', 'kup', 'kup2', 'klow', 'klow2', 'ksft', 'ksft2']
===============================================================================================


trend_features
['ask1_price_trend_60', 'bid1_price_trend_60', 'buy_spread_trend_60', 'sell_spread_trend_60', 'wap_1_trend_60', 'wap_2_trend_60', 'buy_vwap_trend_60', 'sell_vwap_trend_60', 'volume_trend_60']