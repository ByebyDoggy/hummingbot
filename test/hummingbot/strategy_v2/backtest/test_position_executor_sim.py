import unittest
from decimal import Decimal
from typing import Optional

import numpy as np
import pandas as pd
from pandas import Timestamp

from hummingbot.core.data_type.common import TradeType
from hummingbot.strategy_v2.executors.position_executor.data_types import (
    PositionExecutorConfig,
    TripleBarrierConfig, TrailingStop,
)
from hummingbot.strategy_v2.backtesting.executor_simulator_base import ExecutorSimulation
from hummingbot.strategy_v2.models.executors import CloseType
from hummingbot.strategy_v2.backtesting.executors_simulator.position_executor_simulator import (
    PositionExecutorSimulator,
)


class TestPositionExecutorSimulator(unittest.TestCase):
    """Unit tests for PositionExecutorSimulator."""

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
        return df_pos.iloc[0]["timestamp"]

    def _make_config(self,
                     entry_price: float,
                     amount: float,
                     side: TradeType,
                     order_type_str: str = 'LIMIT',
                     tp: Optional[float] = None,
                     sl: Optional[float] = None,
                     tl: Optional[float] = None,
                     trailing_sl_trigger_pct: Optional[float] = None,
                     trailing_sl_delta_pct: Optional[float] = None):
        """
        辅助方法：快速构造 PositionExecutorConfig
        order_type_str: 'MARKET' | 'LIMIT' | 'LIMIT_MAKER' | 'STOP_LIMIT'
        """
        order_type_map = {
            'MARKET': 1,
            'LIMIT': 2,
            'LIMIT_MAKER': 3,
            'STOP_LIMIT': 4,
        }
        open_order_type = order_type_map[order_type_str]

        return PositionExecutorConfig(
            timestamp=self.df['timestamp'].iloc[0],
            trading_pair="BTC-USDT",  # ← 新增
            connector_name="binance",  # ← 新增
            side=side,
            entry_price=Decimal(entry_price),
            amount=Decimal(amount),
            triple_barrier_config=TripleBarrierConfig(
                open_order_type=open_order_type,
                take_profit=tp,
                stop_loss=sl,
                time_limit=tl,
                trailing_stop=TrailingStop(
                    activation_price=trailing_sl_trigger_pct,
                    trailing_delta=trailing_sl_delta_pct,
                ) if trailing_sl_trigger_pct else None,
            ),
        )

    # ----------  tests  ----------
    def test_limit_buy_entry_hit(self):
        """限价 BUY，entry_price=100，第一根 K 线即可成交。"""
        config = self._make_config(entry_price=100, amount=1, side=TradeType.BUY)
        sim: ExecutorSimulation = PositionExecutorSimulator().simulate(
            self.df, config, self.trade_cost
        )
        # 第一根 K 线成交
        self.assertEqual(self.filled_timestamp(sim), self.timestamps[0])
        self.assertEqual(sim.close_type, CloseType.TIME_LIMIT)  # 无止盈止损，默认 TIME_LIMIT

    def test_limit_sell_entry_hit(self):
        """限价 SELL，entry_price=102，第三根 K 线成交。"""
        config = self._make_config(entry_price=102, amount=1, side=TradeType.SELL)
        sim: ExecutorSimulation = PositionExecutorSimulator().simulate(
            self.df, config, self.trade_cost
        )
        # 第三根 K 线 close=102 触发
        self.assertEqual(self.filled_timestamp(sim), self.timestamps[2])
        self.assertEqual(sim.close_type, CloseType.TIME_LIMIT)

    def test_market_buy_immediate_entry(self):
        """市价单立即成交场景。"""
        config = self._make_config(
            entry_price=100, amount=1, side=TradeType.BUY, order_type_str="MARKET"
        )
        sim: ExecutorSimulation = PositionExecutorSimulator().simulate(
            self.df, config, self.trade_cost
        )
        # 市价单立即用第一根 K 线成交
        self.assertEqual(sim.executor_simulation.iloc[0]["timestamp"], self.timestamps[0])

    def test_take_profit_close(self):
        """止盈触发：BUY，tp=2%，价格在第 3 根 K 线达到 102，满足 2% 止盈。"""
        config = self._make_config(
            entry_price=100, amount=1, side=TradeType.BUY, tp=0.02
        )
        sim: ExecutorSimulation = PositionExecutorSimulator().simulate(
            self.df, config, self.trade_cost
        )
        self.assertEqual(sim.close_type, CloseType.TAKE_PROFIT)
        # 第 3 根 K 线 close=102 触发
        self.assertEqual(sim.executor_simulation.iloc[-1]["timestamp"], self.timestamps[2])

    def test_stop_loss_close(self):
        """止损触发：BUY，sl=1%，价格跌至 99 以下触发。"""
        config = self._make_config(
            entry_price=100, amount=1, side=TradeType.BUY, sl=0.01
        )
        sim: ExecutorSimulation = PositionExecutorSimulator().simulate(
            self.df, config, self.trade_cost
        )
        self.assertEqual(sim.close_type, CloseType.STOP_LOSS)
        # 第 6 根 K 线 low=98.5 触发
        self.assertEqual(sim.executor_simulation.iloc[-1]["timestamp"], self.timestamps[5])

    def test_trailing_stop_close(self):
        """跟踪止损触发：BUY，激活价 1%，回撤 0.5%。"""
        config = self._make_config(
            entry_price=100,
            amount=1,
            side=TradeType.BUY,
            tp=None,
            sl=None,
            trailing_sl_trigger_pct=0.01,
            trailing_sl_delta_pct=0.005
        )
        sim: ExecutorSimulation = PositionExecutorSimulator().simulate(
            self.df, config, self.trade_cost
        )
        self.assertEqual(sim.close_type, CloseType.TRAILING_STOP)
        # 价格先涨到 102（+2%），回撤 0.5% 即 101.49，第 4 根 K 线 close=101 触发
        self.assertEqual(sim.executor_simulation.iloc[-1]["timestamp"], self.timestamps[3])

    def test_time_limit_close(self):
        """时间限制触发：tl=4 分钟，刚好第 5 根 K 线收盘截止。"""
        config = self._make_config(
            entry_price=100, amount=1, side=TradeType.BUY, tl=int(4 * 60)
        )
        sim: ExecutorSimulation = PositionExecutorSimulator().simulate(
            self.df, config, self.trade_cost
        )
        self.assertEqual(sim.close_type, CloseType.TIME_LIMIT)
        self.assertEqual(sim.executor_simulation.iloc[-1]["timestamp"], self.timestamps[4])

    def test_fee_and_pnl_calculation(self):
        """验证手续费与净盈亏计算正确性：BUY，无止盈止损，持有到结束。"""
        config = self._make_config(entry_price=100, amount=2, side=TradeType.BUY)
        sim: ExecutorSimulation = PositionExecutorSimulator().simulate(
            self.df, config, self.trade_cost
        )
        # 最后价格 95，亏损 5%
        expected_pnl_pct = (95 - 100) / 100 - self.trade_cost
        np.testing.assert_allclose(
            float(sim.executor_simulation["net_pnl_pct"].iloc[-1]),
            expected_pnl_pct,
            rtol=1e-4,
        )
        # 名义价值 200，净亏损 = 名义 * pnl_pct
        notional = 2 * 100
        expected_pnl_quote = notional * expected_pnl_pct
        np.testing.assert_allclose(
            float(sim.executor_simulation["net_pnl_quote"].iloc[-1]),
            expected_pnl_quote,
            rtol=1e-4,
        )
        # 手续费 = 名义 * fee_rate
        expected_fee = notional * self.trade_cost
        np.testing.assert_allclose(
            float(sim.executor_simulation["cum_fees_quote"].iloc[-1]),
            expected_fee,
            rtol=1e-4,
        )


if __name__ == "__main__":
    unittest.main()
