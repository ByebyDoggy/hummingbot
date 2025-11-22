import unittest
from decimal import Decimal
from typing import List

import pandas as pd
from pandas import Timestamp

from hummingbot.core.data_type.common import TradeType
from hummingbot.strategy_v2.executors.dca_executor.data_types import DCAExecutorConfig
from hummingbot.strategy_v2.backtesting.executor_simulator_base import ExecutorSimulation
from hummingbot.strategy_v2.models.executors import CloseType
from hummingbot.strategy_v2.backtesting.executors_simulator.dca_executor_simulator import DCAExecutorSimulator


class TestDCAExecutorSimulator(unittest.TestCase):
    """Unit tests for DCAExecutorSimulator."""

    def setUp(self) -> None:
        """构造 1 分钟 K 线数据，方便各种场景触发。"""
        # 构造 10 根 1 分钟 K 线，timestamp 用秒级 epoch
        base_ts = int(Timestamp("2024-01-01 00:00:00").timestamp())
        self.timestamps = [int(base_ts + i * 60) for i in range(10)]
        df = pd.DataFrame(
            {
                "timestamp": self.timestamps,
                "open": [100, 101, 102, 101, 100, 99, 98, 97, 96, 95],
                "high": [100.5, 101.5, 102.5, 101.5, 100.5, 99.5, 98.5, 97.5, 96.5, 95.5],
                "low": [99.5, 100.5, 101.5, 100.5, 99.5, 98.5, 97.5, 96.5, 95.5, 94.5],
                "close": [100, 101, 103, 101, 100, 99, 98, 97, 96, 95],
                "volume": [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000],
            }
        )
        df["timestamp"] = df["timestamp"].astype(float)
        # 设置 timestamp 为索引以便于后续查找
        df = df.set_index("timestamp")
        df["timestamp"] = df.index
        self.df = df

        self.trade_cost: float = 0.001  # 0.1 % 手续费

    @staticmethod
    def filled_timestamp(sim: ExecutorSimulation) -> float:
        """返回第一个成交的 timestamp"""
        df_pos = sim.executor_simulation[sim.executor_simulation['filled_amount_quote'] > 0]
        return df_pos.iloc[0]["timestamp"] if not df_pos.empty else None

    def _make_config(
        self,
        prices: List[float],
        amounts_quote: List[float],
        side: TradeType,
        take_profit: float = None,
        stop_loss: float = None,
        time_limit: int = None,
    ) -> DCAExecutorConfig:
        """构造 DCAExecutorConfig 用于测试"""
        return DCAExecutorConfig(
            timestamp=self.timestamps[0]-10,  # 策略启动时间早于 K 线数据
            connector_name="binance",
            trading_pair="ETH-USDT",
            side=side,
            amounts_quote=[Decimal(amount) for amount in amounts_quote],
            prices=[Decimal(price) for price in prices],
            take_profit=Decimal(take_profit) if take_profit else None,
            stop_loss=Decimal(stop_loss) if stop_loss else None,
            time_limit=time_limit
        )

    # ----------  tests  ----------
    # def test_dca_buy_levels_hit(self):
    #     """测试买入DCA，所有价位都能触发"""
    #     config = self._make_config(
    #         prices=[99.5, 98.5, 97.5],  # 三个价位都能在K线中触发
    #         amounts_quote=[100, 200, 300],  # 每个订单的金额
    #         side=TradeType.BUY
    #     )
    #     sim: ExecutorSimulation = DCAExecutorSimulator().simulate(
    #         self.df, config, self.trade_cost
    #     )
    #
    #     # 验证所有订单都已成交
    #     self.assertIsNotNone(sim.executor_simulation['filled_amount_quote_2'])
    #     # 判定没有 filled_amount_quote_3 键
    #     self.assertFalse('filled_amount_quote_3' in sim.executor_simulation)
    #
    #     # 验证成交时间正确
    #     self.assertEqual(self.filled_timestamp(sim), self.timestamps[5])  # 价格99在第5根K线触发
    #     # 默认情况下应该以时间限制结束
    #     self.assertEqual(sim.close_type, CloseType.TIME_LIMIT)

    def test_dca_sell_levels_hit(self):
        """测试卖出DCA，所有价位都能触发"""
        config = self._make_config(
            prices=[100.5, 101.5, 102.5],  # 三个价位都能在K线中触发
            amounts_quote=[100, 200, 300],  # 每个订单的金额
            side=TradeType.SELL
        )
        sim: ExecutorSimulation = DCAExecutorSimulator().simulate(
            self.df, config, self.trade_cost
        )

        #     # 验证所有订单都已成交
        self.assertIsNotNone(sim.executor_simulation['filled_amount_quote_2'])
        # 判定没有 filled_amount_quote_3 键
        self.assertFalse('filled_amount_quote_3' in sim.executor_simulation)
        # 验证成交时间正确
        self.assertEqual(self.filled_timestamp(sim), self.timestamps[1])  # 价格100.5在第1根K线触发
        # 默认情况下应该以时间限制结束
        self.assertEqual(sim.close_type, CloseType.TIME_LIMIT)

    # def test_dca_with_take_profit(self):
    #     """测试带止盈的DCA"""
    #     config = self._make_config(
    #         prices=[100],  # 两个价位
    #         amounts_quote=[100],  # 每个订单的金额
    #         side=TradeType.BUY,
    #         take_profit=0.01  # 1% 止盈
    #     )
    #     sim: ExecutorSimulation = DCAExecutorSimulator().simulate(
    #         self.df, config, self.trade_cost
    #     )
    #
    #     # 验证订单成交
    #     self.assertIsNotNone(self.filled_timestamp(sim))
    #     # 验证止盈触发
    #     self.assertEqual(sim.close_type, CloseType.TAKE_PROFIT)
    #
    # def test_dca_with_stop_loss(self):
    #     """测试带止损的DCA"""
    #     config = self._make_config(
    #         prices=[100.5, 101.5],  # 两个价位
    #         amounts_quote=[100, 200],  # 每个订单的金额
    #         side=TradeType.BUY,
    #         stop_loss=0.02  # 2% 止损
    #     )
    #     sim: ExecutorSimulation = DCAExecutorSimulator().simulate(
    #         self.df, config, self.trade_cost
    #     )
    #
    #     # 验证订单成交
    #     self.assertIsNotNone(self.filled_timestamp(sim))
    #     # 验证止损触发
    #     self.assertEqual(sim.close_type, CloseType.STOP_LOSS)
    #
    # def test_dca_no_levels_hit(self):
    #     """测试没有价位能触发的情况"""
    #     config = self._make_config(
    #         prices=[90, 85, 80],  # 这些价位都低于K线中的最低价
    #         amounts_quote=[100, 200, 300],
    #         side=TradeType.BUY
    #     )
    #     sim: ExecutorSimulation = DCAExecutorSimulator().simulate(
    #         self.df, config, self.trade_cost
    #     )
    #
    #     # 验证没有订单成交
    #     self.assertIsNone(self.filled_timestamp(sim))
    #     # 应该以时间限制结束
    #     self.assertEqual(sim.close_type, CloseType.TIME_LIMIT)
    #
    # def test_dca_partial_levels_hit(self):
    #     """测试部分价位能触发的情况"""
    #     config = self._make_config(
    #         prices=[98.5, 90, 85],  # 只有第一个价位能触发
    #         amounts_quote=[100, 200, 300],
    #         side=TradeType.BUY
    #     )
    #     sim: ExecutorSimulation = DCAExecutorSimulator().simulate(
    #         self.df, config, self.trade_cost
    #     )
    #     # 验证只有一个订单成交
    #     filled_amount = sim.executor_simulation['filled_amount_quote'].iloc[-1]
    #     self.assertEqual(filled_amount, 100.0)
    #     # 应该以时间限制结束
    #     self.assertEqual(sim.close_type, CloseType.TIME_LIMIT)


if __name__ == "__main__":
    unittest.main()
