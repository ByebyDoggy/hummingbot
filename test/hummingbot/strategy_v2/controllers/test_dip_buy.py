import asyncio
from decimal import Decimal
from test.isolated_asyncio_wrapper_test_case import IsolatedAsyncioWrapperTestCase
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd

from hummingbot.core.data_type.common import MarketDict, PositionMode, TradeType
from controllers.generic.dip_buy import DipBuy, DipBuyConfig
from hummingbot.data_feed.market_data_provider import MarketDataProvider
from hummingbot.strategy_v2.executors.position_executor.data_types import PositionExecutorConfig
from hummingbot.strategy_v2.models.executor_actions import CreateExecutorAction


class TestDipBuyConfig(IsolatedAsyncioWrapperTestCase):
    """Tests for DipBuyConfig."""

    def test_default_config_values(self):
        """Test default configuration values."""
        config = DipBuyConfig(
            id="test",
            controller_name="dip_buy_test",
        )
        
        self.assertEqual(config.controller_name, "dip_buy")
        self.assertEqual(config.interval, "1m")
        self.assertEqual(config.lookback_periods, 60)
        self.assertEqual(config.dip_threshold, Decimal("0.03"))
        self.assertFalse(config.enable_short)

    def test_custom_config_values(self):
        """Test custom configuration values."""
        config = DipBuyConfig(
            id="test",
            controller_name="dip_buy_test",
            connector_name="binance",
            trading_pair="ETH-USDT",
            interval="5m",
            lookback_periods=120,
            dip_threshold=Decimal("0.05"),
            enable_short=True,
            leverage=20,
        )
        
        self.assertEqual(config.connector_name, "binance")
        self.assertEqual(config.trading_pair, "ETH-USDT")
        self.assertEqual(config.interval, "5m")
        self.assertEqual(config.lookback_periods, 120)
        self.assertEqual(config.dip_threshold, Decimal("0.05"))
        self.assertTrue(config.enable_short)
        self.assertEqual(config.leverage, 20)

    def test_candles_connector_defaults_to_connector_name(self):
        """Test that candles_connector defaults to connector_name."""
        config = DipBuyConfig(
            id="test",
            controller_name="dip_buy_test",
            connector_name="binance_perpetual",
        )
        
        self.assertEqual(config.candles_connector, "binance_perpetual")

    def test_candles_trading_pair_defaults_to_trading_pair(self):
        """Test that candles_trading_pair defaults to trading_pair."""
        config = DipBuyConfig(
            id="test",
            controller_name="dip_buy_test",
            trading_pair="BTC-USDT",
        )
        
        self.assertEqual(config.candles_trading_pair, "BTC-USDT")

    def test_custom_candles_config(self):
        """Test custom candles connector and trading pair."""
        config = DipBuyConfig(
            id="test",
            controller_name="dip_buy_test",
            connector_name="binance_perpetual",
            trading_pair="BTC-USDT",
            candles_connector="binance",
            candles_trading_pair="BTC-USDT",
        )
        
        self.assertEqual(config.candles_connector, "binance")
        self.assertEqual(config.candles_trading_pair, "BTC-USDT")

    def test_validate_dip_threshold_string(self):
        """Test dip_threshold validation from string."""
        config = DipBuyConfig(
            id="test",
            controller_name="dip_buy_test",
            dip_threshold="0.05",
        )
        
        self.assertEqual(config.dip_threshold, Decimal("0.05"))

    def test_update_markets(self):
        """Test market update."""
        config = DipBuyConfig(
            id="test",
            controller_name="dip_buy_test",
            connector_name="binance",
            trading_pair="BTC-USDT",
        )
        
        markets = MarketDict()
        updated = config.update_markets(markets)
        
        self.assertIn("binance", updated)
        self.assertIn("BTC-USDT", updated["binance"])


class TestDipBuy(IsolatedAsyncioWrapperTestCase):
    """Tests for DipBuy controller."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = DipBuyConfig(
            id="test_dip_buy",
            controller_name="dip_buy_test",
            connector_name="binance_perpetual",
            trading_pair="BTC-USDT",
            interval="1m",
            lookback_periods=60,
            dip_threshold=Decimal("0.03"),
            total_amount_quote=Decimal("1000"),
            max_executors_per_side=2,
            leverage=10,
        )
        
        self.mock_market_data_provider = MagicMock(spec=MarketDataProvider)
        self.mock_market_data_provider.ready = True
        self.mock_market_data_provider.time = MagicMock(return_value=1000000.0)
        self.mock_market_data_provider.get_price_by_type = MagicMock(return_value=Decimal("50000"))
        self.mock_market_data_provider.initialize_rate_sources = MagicMock()
        
        self.mock_actions_queue = AsyncMock(spec=asyncio.Queue)
        
        self.controller = DipBuy(
            config=self.config,
            market_data_provider=self.mock_market_data_provider,
            actions_queue=self.mock_actions_queue,
        )

    def _create_mock_candles_df(self, num_candles: int = 70, base_price: float = 50000.0, dip_percent: float = 0.0):
        """Create mock candles DataFrame for testing."""
        data = []
        for i in range(num_candles):
            # Create price movement with optional dip
            if dip_percent > 0 and i > num_candles - 30:
                # Last 30 candles show a dip
                progress = (i - (num_candles - 30)) / 30
                price_factor = 1.0 - (dip_percent * progress)
            else:
                price_factor = 1.0
            
            close = base_price * price_factor
            high = close * 1.001
            low = close * 0.999
            open_price = close * 0.9995
            volume = 100.0
            
            data.append({
                "timestamp": 1000000 + i * 60,
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            })
        
        return pd.DataFrame(data)

    def test_initialization(self):
        """Test controller initialization."""
        self.assertEqual(self.controller.max_records, 70)  # lookback_periods + 10
        self.assertIsNotNone(self.controller.config.candles_config)
        self.assertEqual(len(self.controller.config.candles_config), 1)

    def test_candles_config_auto_created(self):
        """Test that candles_config is auto-created if not provided."""
        config = DipBuyConfig(
            id="test",
            controller_name="dip_buy_test",
            connector_name="binance",
            trading_pair="BTC-USDT",
            interval="5m",
            lookback_periods=100,
        )
        
        controller = DipBuy(
            config=config,
            market_data_provider=self.mock_market_data_provider,
            actions_queue=self.mock_actions_queue,
        )
        
        self.assertEqual(len(controller.config.candles_config), 1)
        self.assertEqual(controller.config.candles_config[0].connector, "binance")
        self.assertEqual(controller.config.candles_config[0].trading_pair, "BTC-USDT")
        self.assertEqual(controller.config.candles_config[0].interval, "5m")
        self.assertEqual(controller.config.candles_config[0].max_records, 110)

    async def test_update_processed_data_no_candles(self):
        """Test update_processed_data when no candles data available."""
        self.mock_market_data_provider.get_candles_df = MagicMock(return_value=None)
        
        await self.controller.update_processed_data()
        
        self.assertEqual(self.controller.processed_data["signal"], 0)

    async def test_update_processed_data_insufficient_candles(self):
        """Test update_processed_data when insufficient candles data."""
        df = self._create_mock_candles_df(num_candles=10)
        self.mock_market_data_provider.get_candles_df = MagicMock(return_value=df)
        
        await self.controller.update_processed_data()
        
        self.assertEqual(self.controller.processed_data["signal"], 0)

    async def test_update_processed_data_no_dip(self):
        """Test update_processed_data when no significant dip."""
        # Create stable price data
        df = self._create_mock_candles_df(num_candles=70, base_price=50000.0, dip_percent=0.0)
        self.mock_market_data_provider.get_candles_df = MagicMock(return_value=df)
        
        await self.controller.update_processed_data()
        
        self.assertEqual(self.controller.processed_data["signal"], 0)

    async def test_update_processed_data_with_dip(self):
        """Test update_processed_data detects dip and generates buy signal."""
        # Create data with 5% dip
        df = self._create_mock_candles_df(num_candles=70, base_price=50000.0, dip_percent=0.05)
        self.mock_market_data_provider.get_candles_df = MagicMock(return_value=df)
        
        await self.controller.update_processed_data()
        
        # Signal should be 1 (buy) since dip exceeds threshold
        self.assertEqual(self.controller.processed_data["signal"], 1)

    async def test_update_processed_data_small_dip(self):
        """Test update_processed_data with dip below threshold."""
        # Create data with only 1% dip (below default 3% threshold)
        df = self._create_mock_candles_df(num_candles=70, base_price=50000.0, dip_percent=0.01)
        self.mock_market_data_provider.get_candles_df = MagicMock(return_value=df)
        
        await self.controller.update_processed_data()
        
        # Signal should be 0 (no signal) since dip is below threshold
        self.assertEqual(self.controller.processed_data["signal"], 0)

    async def test_update_processed_data_custom_threshold(self):
        """Test update_processed_data with custom threshold."""
        config = DipBuyConfig(
            id="test",
            controller_name="dip_buy_test",
            dip_threshold=Decimal("0.01"),  # 1% threshold
            lookback_periods=60,
        )
        
        controller = DipBuy(
            config=config,
            market_data_provider=self.mock_market_data_provider,
            actions_queue=self.mock_actions_queue,
        )
        
        # Create data with 1.5% dip
        df = self._create_mock_candles_df(num_candles=70, base_price=50000.0, dip_percent=0.015)
        self.mock_market_data_provider.get_candles_df = MagicMock(return_value=df)
        
        await controller.update_processed_data()
        
        # Signal should be 1 (buy) since dip exceeds 1% threshold
        self.assertEqual(controller.processed_data["signal"], 1)

    async def test_update_processed_data_short_disabled(self):
        """Test that short signals are not generated when disabled."""
        config = DipBuyConfig(
            id="test",
            controller_name="dip_buy_test",
            dip_threshold=Decimal("0.03"),
            enable_short=False,
            lookback_periods=60,
        )
        
        controller = DipBuy(
            config=config,
            market_data_provider=self.mock_market_data_provider,
            actions_queue=self.mock_actions_queue,
        )
        
        # Create data with surge (price going up) - should NOT trigger sell
        df = self._create_mock_candles_df(num_candles=70, base_price=50000.0, dip_percent=-0.05)
        self.mock_market_data_provider.get_candles_df = MagicMock(return_value=df)
        
        await controller.update_processed_data()
        
        # Signal should be 0 (no short signals when disabled)
        self.assertEqual(controller.processed_data["signal"], 0)

    async def test_update_processed_data_short_enabled(self):
        """Test that short signals are generated when enabled."""
        config = DipBuyConfig(
            id="test",
            controller_name="dip_buy_test",
            dip_threshold=Decimal("0.03"),
            enable_short=True,
            lookback_periods=60,
        )
        
        controller = DipBuy(
            config=config,
            market_data_provider=self.mock_market_data_provider,
            actions_queue=self.mock_actions_queue,
        )
        
        # Create data with surge - price increasing significantly
        # We need to invert the logic for surge detection
        data = []
        base_price = 50000.0
        for i in range(70):
            if i < 40:
                close = base_price * 0.95  # Start low
            else:
                # Price surges up
                progress = (i - 40) / 30
                close = base_price * 0.95 * (1 + 0.06 * progress)  # 6% surge
            
            data.append({
                "timestamp": 1000000 + i * 60,
                "open": close * 0.9995,
                "high": close * 1.001,
                "low": close * 0.999,
                "close": close,
                "volume": 100.0,
            })
        
        df = pd.DataFrame(data)
        self.mock_market_data_provider.get_candles_df = MagicMock(return_value=df)
        
        await controller.update_processed_data()
        
        # Signal should be -1 (sell) since surge exceeds threshold
        self.assertEqual(controller.processed_data["signal"], -1)

    async def test_update_processed_data_stores_features(self):
        """Test that update_processed_data stores feature data."""
        df = self._create_mock_candles_df(num_candles=70)
        self.mock_market_data_provider.get_candles_df = MagicMock(return_value=df)
        
        await self.controller.update_processed_data()
        
        self.assertIn("features", self.controller.processed_data)
        self.assertIn("current_price", self.controller.processed_data)
        self.assertIn("highest_price", self.controller.processed_data)
        self.assertIn("lowest_price", self.controller.processed_data)
        self.assertIn("dip_percentage", self.controller.processed_data)

    @patch.object(DipBuy, "get_executor_config")
    async def test_determine_executor_actions_with_signal(self, get_executor_config_mock: MagicMock):
        """Test determine_executor_actions with buy signal."""
        get_executor_config_mock.return_value = PositionExecutorConfig(
            timestamp=1234,
            controller_id=self.controller.config.id,
            connector_name="binance_perpetual",
            trading_pair="BTC-USDT",
            side=TradeType.BUY,
            entry_price=Decimal("50000"),
            amount=Decimal("0.01"),
        )
        
        # Set up conditions for buy signal
        df = self._create_mock_candles_df(num_candles=70, base_price=50000.0, dip_percent=0.05)
        self.mock_market_data_provider.get_candles_df = MagicMock(return_value=df)
        
        await self.controller.update_processed_data()
        self.controller.executors_info = []
        
        actions = self.controller.determine_executor_actions()
        
        self.assertEqual(len(actions), 1)
        self.assertIsInstance(actions[0], CreateExecutorAction)
        self.assertEqual(actions[0].controller_id, "test_dip_buy")

    def test_to_format_status(self):
        """Test status formatting."""
        # Set up processed data
        self.controller.processed_data = {
            "signal": 1,
            "current_price": 48000.0,
            "highest_price": 50000.0,
            "lowest_price": 47000.0,
            "dip_percentage": 0.04,
            "surge_percentage": 0.02,
        }
        
        status = self.controller.to_format_status()
        
        self.assertIsInstance(status, list)
        self.assertTrue(any("Dip Buy Strategy" in line for line in status))
        self.assertTrue(any("BTC-USDT" in line for line in status))
        self.assertTrue(any("3.00%" in line for line in status))  # Dip threshold
        self.assertTrue(any("48000" in line for line in status))  # Current price

    def test_to_format_status_no_data(self):
        """Test status formatting with no processed data."""
        self.controller.processed_data = {}
        
        status = self.controller.to_format_status()
        
        self.assertIsInstance(status, list)
        self.assertTrue(any("Dip Buy Strategy" in line for line in status))

    def test_get_executor_config(self):
        """Test executor config creation."""
        config = self.controller.get_executor_config(
            trade_type=TradeType.BUY,
            price=Decimal("50000"),
            amount=Decimal("0.02"),
        )
        
        self.assertIsInstance(config, PositionExecutorConfig)
        self.assertEqual(config.connector_name, "binance_perpetual")
        self.assertEqual(config.trading_pair, "BTC-USDT")
        self.assertEqual(config.side, TradeType.BUY)
        self.assertEqual(config.entry_price, Decimal("50000"))
        self.assertEqual(config.amount, Decimal("0.02"))
        self.assertEqual(config.leverage, 10)


class TestDipBuyEdgeCases(IsolatedAsyncioWrapperTestCase):
    """Tests for edge cases in DipBuy controller."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = DipBuyConfig(
            id="test_dip_buy",
            controller_name="dip_buy_test",
            connector_name="binance_perpetual",
            trading_pair="BTC-USDT",
            interval="1m",
            lookback_periods=60,
            dip_threshold=Decimal("0.03"),
        )
        
        self.mock_market_data_provider = MagicMock(spec=MarketDataProvider)
        self.mock_market_data_provider.ready = True
        self.mock_market_data_provider.time = MagicMock(return_value=1000000.0)
        self.mock_market_data_provider.get_price_by_type = MagicMock(return_value=Decimal("50000"))
        self.mock_market_data_provider.initialize_rate_sources = MagicMock()
        
        self.mock_actions_queue = AsyncMock(spec=asyncio.Queue)

    async def test_exactly_at_threshold(self):
        """Test behavior when dip is exactly at threshold."""
        controller = DipBuy(
            config=self.config,
            market_data_provider=self.mock_market_data_provider,
            actions_queue=self.mock_actions_queue,
        )
        
        # Create data with exactly 3% dip
        data = []
        base_price = 100.0
        for i in range(70):
            if i < 40:
                close = base_price
            else:
                close = base_price * 0.97  # Exactly 3% dip
            
            data.append({
                "timestamp": 1000000 + i * 60,
                "open": close,
                "high": base_price,  # High remains at base
                "low": close,
                "close": close,
                "volume": 100.0,
            })
        
        df = pd.DataFrame(data)
        self.mock_market_data_provider.get_candles_df = MagicMock(return_value=df)
        
        await controller.update_processed_data()
        
        # Signal should be 1 (buy) since dip >= threshold
        self.assertEqual(controller.processed_data["signal"], 1)

    async def test_just_below_threshold(self):
        """Test behavior when dip is just below threshold."""
        controller = DipBuy(
            config=self.config,
            market_data_provider=self.mock_market_data_provider,
            actions_queue=self.mock_actions_queue,
        )
        
        # Create data with 2.9% dip (just below 3% threshold)
        data = []
        base_price = 100.0
        for i in range(70):
            if i < 40:
                close = base_price
            else:
                close = base_price * 0.971  # 2.9% dip
            
            data.append({
                "timestamp": 1000000 + i * 60,
                "open": close,
                "high": base_price,
                "low": close,
                "close": close,
                "volume": 100.0,
            })
        
        df = pd.DataFrame(data)
        self.mock_market_data_provider.get_candles_df = MagicMock(return_value=df)
        
        await controller.update_processed_data()
        
        # Signal should be 0 (no signal) since dip < threshold
        self.assertEqual(controller.processed_data["signal"], 0)

    async def test_extreme_dip(self):
        """Test behavior with extreme dip (e.g., 50%)."""
        controller = DipBuy(
            config=self.config,
            market_data_provider=self.mock_market_data_provider,
            actions_queue=self.mock_actions_queue,
        )
        
        # Create data with 50% dip
        data = []
        base_price = 100.0
        for i in range(70):
            if i < 40:
                close = base_price
            else:
                close = base_price * 0.5  # 50% dip
            
            data.append({
                "timestamp": 1000000 + i * 60,
                "open": close,
                "high": base_price,
                "low": close,
                "close": close,
                "volume": 100.0,
            })
        
        df = pd.DataFrame(data)
        self.mock_market_data_provider.get_candles_df = MagicMock(return_value=df)
        
        await controller.update_processed_data()
        
        # Signal should still be 1 (buy)
        self.assertEqual(controller.processed_data["signal"], 1)
        # Verify the dip percentage is correct
        self.assertAlmostEqual(
            float(controller.processed_data["dip_percentage"]),
            0.5,
            places=2
        )

    async def test_volatile_price_action(self):
        """Test with volatile price movements."""
        controller = DipBuy(
            config=self.config,
            market_data_provider=self.mock_market_data_provider,
            actions_queue=self.mock_actions_queue,
        )
        
        # Create volatile data that eventually dips
        import random
        random.seed(42)
        
        data = []
        base_price = 100.0
        current_price = base_price
        
        for i in range(70):
            # Random walk with downward bias near the end
            if i < 50:
                change = random.uniform(-0.02, 0.02)
            else:
                change = random.uniform(-0.03, -0.01)  # Downward bias
            
            current_price *= (1 + change)
            
            data.append({
                "timestamp": 1000000 + i * 60,
                "open": current_price * 0.999,
                "high": max(current_price, base_price),
                "low": current_price * 0.998,
                "close": current_price,
                "volume": 100.0,
            })
        
        df = pd.DataFrame(data)
        self.mock_market_data_provider.get_candles_df = MagicMock(return_value=df)
        
        await controller.update_processed_data()
        
        # Just verify it runs without error
        self.assertIn("signal", controller.processed_data)
