import asyncio
from decimal import Decimal
from test.isolated_asyncio_wrapper_test_case import IsolatedAsyncioWrapperTestCase
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

from hummingbot.core.data_type.common import MarketDict, OrderType, PositionMode, PriceType, TradeType
from hummingbot.data_feed.market_data_provider import MarketDataProvider
from hummingbot.strategy_v2.controllers.signal_controller_base import (
    SignalControllerBase,
    SignalControllerConfigBase,
    SignalType,
)
from hummingbot.strategy_v2.executors.position_executor.data_types import PositionExecutorConfig
from hummingbot.strategy_v2.models.executor_actions import CreateExecutorAction, StopExecutorAction


class TestSignalControllerConfigBase(IsolatedAsyncioWrapperTestCase):
    """Tests for SignalControllerConfigBase."""

    def test_default_config_values(self):
        """Test default configuration values."""
        config = SignalControllerConfigBase(
            id="test",
            controller_name="signal_test",
        )
        
        self.assertEqual(config.controller_type, "signal_controller")
        self.assertEqual(config.connector_name, "binance_perpetual")
        self.assertEqual(config.trading_pair, "BTC-USDT")
        self.assertEqual(config.max_executors_per_side, 3)
        self.assertEqual(config.cooldown_time, 60)
        self.assertEqual(config.leverage, 10)
        self.assertEqual(config.position_mode, PositionMode.HEDGE)
        self.assertEqual(config.stop_loss, Decimal("0.03"))
        self.assertEqual(config.take_profit, Decimal("0.02"))
        self.assertTrue(config.auto_close_on_opposite_signal)

    def test_custom_config_values(self):
        """Test custom configuration values."""
        config = SignalControllerConfigBase(
            id="test",
            controller_name="signal_test",
            connector_name="binance",
            trading_pair="ETH-USDT",
            max_executors_per_side=5,
            cooldown_time=120,
            leverage=20,
            stop_loss=Decimal("0.05"),
            take_profit=Decimal("0.03"),
            auto_close_on_opposite_signal=False,
        )
        
        self.assertEqual(config.connector_name, "binance")
        self.assertEqual(config.trading_pair, "ETH-USDT")
        self.assertEqual(config.max_executors_per_side, 5)
        self.assertEqual(config.cooldown_time, 120)
        self.assertEqual(config.leverage, 20)
        self.assertEqual(config.stop_loss, Decimal("0.05"))
        self.assertEqual(config.take_profit, Decimal("0.03"))
        self.assertFalse(config.auto_close_on_opposite_signal)

    def test_triple_barrier_config(self):
        """Test triple barrier config generation."""
        config = SignalControllerConfigBase(
            id="test",
            controller_name="signal_test",
            stop_loss=Decimal("0.05"),
            take_profit=Decimal("0.03"),
            time_limit=1800,
        )
        
        triple_barrier = config.triple_barrier_config
        
        self.assertEqual(triple_barrier.stop_loss, Decimal("0.05"))
        self.assertEqual(triple_barrier.take_profit, Decimal("0.03"))
        self.assertEqual(triple_barrier.time_limit, 1800)
        self.assertEqual(triple_barrier.open_order_type, OrderType.MARKET)

    def test_parse_trailing_stop(self):
        """Test trailing stop parsing."""
        config = SignalControllerConfigBase(
            id="test",
            controller_name="signal_test",
            trailing_stop="0.015,0.003",
        )
        
        self.assertIsNotNone(config.trailing_stop)
        self.assertEqual(config.trailing_stop.activation_price, Decimal("0.015"))
        self.assertEqual(config.trailing_stop.trailing_delta, Decimal("0.003"))

    def test_parse_trailing_stop_empty(self):
        """Test trailing stop parsing with empty string."""
        config = SignalControllerConfigBase(
            id="test",
            controller_name="signal_test",
            trailing_stop="",
        )
        
        self.assertIsNone(config.trailing_stop)

    def test_validate_order_type(self):
        """Test order type validation."""
        for order_type_name in OrderType.__members__:
            result = SignalControllerConfigBase.validate_order_type(order_type_name)
            self.assertEqual(result, OrderType[order_type_name])

    def test_validate_order_type_invalid(self):
        """Test order type validation with invalid input."""
        with self.assertRaises(ValueError):
            SignalControllerConfigBase.validate_order_type("invalid_order_type")

    def test_validate_position_mode(self):
        """Test position mode validation."""
        self.assertEqual(
            SignalControllerConfigBase.validate_position_mode("HEDGE"),
            PositionMode.HEDGE
        )
        self.assertEqual(
            SignalControllerConfigBase.validate_position_mode("ONEWAY"),
            PositionMode.ONEWAY
        )

    def test_validate_position_mode_invalid(self):
        """Test position mode validation with invalid input."""
        with self.assertRaises(ValueError):
            SignalControllerConfigBase.validate_position_mode("INVALID")

    def test_update_markets(self):
        """Test market update."""
        config = SignalControllerConfigBase(
            id="test",
            controller_name="signal_test",
            connector_name="binance",
            trading_pair="BTC-USDT",
        )
        
        markets = MarketDict()
        updated = config.update_markets(markets)
        
        self.assertIn("binance", updated)
        self.assertIn("BTC-USDT", updated["binance"])


class TestSignalControllerBase(IsolatedAsyncioWrapperTestCase):
    """Tests for SignalControllerBase."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = SignalControllerConfigBase(
            id="test_controller",
            controller_name="signal_test",
            connector_name="binance_perpetual",
            trading_pair="BTC-USDT",
            total_amount_quote=Decimal("1000"),
            max_executors_per_side=3,
            cooldown_time=60,
            leverage=10,
        )
        
        self.mock_market_data_provider = MagicMock(spec=MarketDataProvider)
        self.mock_market_data_provider.ready = True
        self.mock_market_data_provider.time = MagicMock(return_value=1000000.0)
        self.mock_market_data_provider.get_price_by_type = MagicMock(return_value=Decimal("50000"))
        self.mock_market_data_provider.initialize_rate_sources = MagicMock()
        
        self.mock_actions_queue = AsyncMock(spec=asyncio.Queue)
        
        self.controller = SignalControllerBase(
            config=self.config,
            market_data_provider=self.mock_market_data_provider,
            actions_queue=self.mock_actions_queue,
        )

    def test_initialization(self):
        """Test controller initialization."""
        self.assertIsNotNone(self.controller._pending_signals)
        self.assertIsNotNone(self.controller._signal_event)
        self.assertEqual(self.controller._last_signal_time, 0.0)
        self.assertEqual(len(self.controller._pre_trade_hooks), 0)
        self.assertEqual(len(self.controller._post_trade_hooks), 0)

    def test_on_buy_signal(self):
        """Test buy signal trigger."""
        result = self.controller.on_buy_signal()
        
        self.assertTrue(result)
        self.assertEqual(self.controller._pending_signals.qsize(), 1)

    def test_on_sell_signal(self):
        """Test sell signal trigger."""
        result = self.controller.on_sell_signal()
        
        self.assertTrue(result)
        self.assertEqual(self.controller._pending_signals.qsize(), 1)

    def test_on_signal_buy(self):
        """Test signal trigger with BUY type."""
        result = self.controller.on_signal(SignalType.BUY)
        
        self.assertTrue(result)
        self.assertEqual(self.controller._pending_signals.qsize(), 1)

    def test_on_signal_neutral(self):
        """Test signal trigger with NEUTRAL type."""
        result = self.controller.on_signal(SignalType.NEUTRAL)
        
        self.assertFalse(result)
        self.assertEqual(self.controller._pending_signals.qsize(), 0)

    def test_on_signal_with_cooldown(self):
        """Test signal is ignored during cooldown."""
        # First signal should succeed
        result1 = self.controller.on_buy_signal()
        self.assertTrue(result1)
        
        # Update last signal time to simulate recent signal
        self.controller._last_signal_time = 1000000.0
        
        # Second signal within cooldown should be ignored
        result2 = self.controller.on_buy_signal()
        self.assertFalse(result2)

    def test_on_signal_with_custom_amount(self):
        """Test signal with custom amount."""
        result = self.controller.on_buy_signal(amount=Decimal("0.5"))
        
        self.assertTrue(result)
        self.assertEqual(self.controller._pending_signals.qsize(), 1)

    def test_signal_event_is_set_on_queue(self):
        """Test that signal event is set when signal is queued."""
        self.assertFalse(self.controller._signal_event.is_set())
        
        self.controller.on_buy_signal()
        
        self.assertTrue(self.controller._signal_event.is_set())

    def test_register_pre_trade_hook(self):
        """Test pre-trade hook registration."""
        hook = MagicMock()
        self.controller.register_pre_trade_hook(hook)
        
        self.assertIn(hook, self.controller._pre_trade_hooks)

    def test_register_post_trade_hook(self):
        """Test post-trade hook registration."""
        hook = MagicMock()
        self.controller.register_post_trade_hook(hook)
        
        self.assertIn(hook, self.controller._post_trade_hooks)

    def test_unregister_pre_trade_hook(self):
        """Test pre-trade hook unregistration."""
        hook = MagicMock()
        self.controller.register_pre_trade_hook(hook)
        self.controller.unregister_pre_trade_hook(hook)
        
        self.assertNotIn(hook, self.controller._pre_trade_hooks)

    def test_unregister_post_trade_hook(self):
        """Test post-trade hook unregistration."""
        hook = MagicMock()
        self.controller.register_post_trade_hook(hook)
        self.controller.unregister_post_trade_hook(hook)
        
        self.assertNotIn(hook, self.controller._post_trade_hooks)

    async def test_process_signals_creates_action(self):
        """Test that processing signals creates executor actions."""
        # Queue a buy signal
        self.controller._pending_signals.put_nowait((SignalType.BUY, None))
        
        # Mock the executor check
        self.controller.executors_info = []
        self.controller.executors_update_event.set()
        
        actions = await self.controller._process_signals()
        
        self.assertEqual(len(actions), 1)
        self.assertIsInstance(actions[0], CreateExecutorAction)
        self.assertEqual(actions[0].controller_id, "test_controller")

    async def test_pre_trade_hook_cancels_trade(self):
        """Test that pre-trade hook can cancel a trade."""
        # Register a hook that returns False
        cancel_hook = MagicMock(return_value=False)
        self.controller.register_pre_trade_hook(cancel_hook)
        
        self.controller._pending_signals.put_nowait((SignalType.BUY, None))
        self.controller.executors_info = []
        
        actions = await self.controller._process_signals()
        
        self.assertEqual(len(actions), 0)
        cancel_hook.assert_called_once()

    async def test_pre_trade_hook_allows_trade(self):
        """Test that pre-trade hook returning True allows trade."""
        allow_hook = MagicMock(return_value=True)
        self.controller.register_pre_trade_hook(allow_hook)
        
        self.controller._pending_signals.put_nowait((SignalType.BUY, None))
        self.controller.executors_info = []
        self.controller.executors_update_event.set()
        
        actions = await self.controller._process_signals()
        
        self.assertEqual(len(actions), 1)
        allow_hook.assert_called_once()

    async def test_async_pre_trade_hook(self):
        """Test async pre-trade hook."""
        async def async_hook(trade_type, amount, price):
            return True
        
        self.controller.register_pre_trade_hook(async_hook)
        self.controller._pending_signals.put_nowait((SignalType.BUY, None))
        self.controller.executors_info = []
        self.controller.executors_update_event.set()
        
        actions = await self.controller._process_signals()
        
        self.assertEqual(len(actions), 1)

    async def test_post_trade_hook_called(self):
        """Test that post-trade hook is called after trade."""
        post_hook = MagicMock()
        self.controller.register_post_trade_hook(post_hook)
        
        self.controller._pending_signals.put_nowait((SignalType.BUY, None))
        self.controller.executors_info = []
        self.controller.executors_update_event.set()
        
        await self.controller._process_signals()
        
        post_hook.assert_called_once()

    def test_can_create_executor_no_active(self):
        """Test can_create_executor with no active executors."""
        self.controller.executors_info = []
        
        result = self.controller._can_create_executor(TradeType.BUY)
        
        self.assertTrue(result)

    def test_can_create_executor_at_limit(self):
        """Test can_create_executor when at executor limit."""
        mock_executor = MagicMock()
        mock_executor.is_active = True
        mock_executor.side = TradeType.BUY
        mock_executor.timestamp = 900000
        
        self.controller.executors_info = [mock_executor] * 3  # At limit of 3
        
        result = self.controller._can_create_executor(TradeType.BUY)
        
        self.assertFalse(result)

    def test_can_create_executor_cooldown_not_expired(self):
        """Test can_create_executor when cooldown hasn't expired."""
        mock_executor = MagicMock()
        mock_executor.is_active = True
        mock_executor.side = TradeType.BUY
        mock_executor.timestamp = 999000  # Recent timestamp
        
        self.controller.executors_info = [mock_executor]
        
        result = self.controller._can_create_executor(TradeType.BUY)
        
        self.assertFalse(result)

    def test_get_default_amount(self):
        """Test default amount calculation."""
        # total_amount_quote=1000, max_executors_per_side=3, price=50000
        # Expected: 1000 / 50000 / 3 = 0.006666...
        amount = self.controller._get_default_amount()
        
        expected = Decimal("1000") / Decimal("50000") / Decimal("3")
        self.assertEqual(amount, expected)

    def test_get_executor_config(self):
        """Test executor config creation."""
        config = self.controller._get_executor_config(
            trade_type=TradeType.BUY,
            price=Decimal("50000"),
            amount=Decimal("0.01"),
        )
        
        self.assertIsInstance(config, PositionExecutorConfig)
        self.assertEqual(config.connector_name, "binance_perpetual")
        self.assertEqual(config.trading_pair, "BTC-USDT")
        self.assertEqual(config.side, TradeType.BUY)
        self.assertEqual(config.entry_price, Decimal("50000"))
        self.assertEqual(config.amount, Decimal("0.01"))
        self.assertEqual(config.leverage, 10)

    async def test_close_opposite_executors(self):
        """Test closing opposite executors on signal."""
        # Create mock executors
        buy_executor = MagicMock()
        buy_executor.id = "buy_executor_1"
        buy_executor.is_active = True
        buy_executor.side = TradeType.BUY
        
        sell_executor = MagicMock()
        sell_executor.id = "sell_executor_1"
        sell_executor.is_active = True
        sell_executor.side = TradeType.SELL
        
        self.controller.executors_info = [buy_executor, sell_executor]
        
        # Trigger buy signal which should close sell executors
        self.controller._close_opposite_executors(TradeType.BUY)
        
        # Allow the stop action to be queued
        await asyncio.sleep(0.01)
        
        # Verify stop action was queued for sell executor
        self.mock_actions_queue.put.assert_called()

    async def test_close_all_executors(self):
        """Test closing all executors."""
        mock_executor1 = MagicMock()
        mock_executor1.id = "executor_1"
        mock_executor1.is_active = True
        mock_executor1.side = TradeType.BUY
        
        mock_executor2 = MagicMock()
        mock_executor2.id = "executor_2"
        mock_executor2.is_active = True
        mock_executor2.side = TradeType.SELL
        
        self.controller.executors_info = [mock_executor1, mock_executor2]
        
        self.controller._close_all_executors()
        
        # Allow the stop actions to be queued
        await asyncio.sleep(0.01)
        
        # Verify stop actions were queued
        self.assertEqual(self.mock_actions_queue.put.call_count, 2)

    def test_determine_executor_actions_returns_empty(self):
        """Test that determine_executor_actions returns empty list."""
        actions = self.controller.determine_executor_actions()
        
        self.assertEqual(actions, [])

    async def test_update_processed_data(self):
        """Test update_processed_data."""
        self.controller._pending_signals.put_nowait((SignalType.BUY, None))
        
        await self.controller.update_processed_data()
        
        self.assertIn("pending_signals_count", self.controller.processed_data)
        self.assertIn("last_signal_time", self.controller.processed_data)
        self.assertEqual(self.controller.processed_data["pending_signals_count"], 1)

    def test_to_format_status(self):
        """Test status formatting."""
        status = self.controller.to_format_status()
        
        self.assertIsInstance(status, list)
        self.assertTrue(any("Signal Controller" in line for line in status))
        self.assertTrue(any("binance_perpetual" in line for line in status))
        self.assertTrue(any("BTC-USDT" in line for line in status))

    async def test_signal_processor_loop_processes_signal(self):
        """Test signal processor loop processes signals immediately."""
        self.controller.executors_info = []
        self.controller.executors_update_event.set()
        
        # Start the controller
        self.controller.start()
        
        # Queue a signal
        self.controller.on_buy_signal()
        
        # Wait for signal to be processed
        await asyncio.sleep(0.1)
        
        # Verify the signal was processed
        self.assertEqual(self.controller._pending_signals.qsize(), 0)
        
        # Verify action was sent
        self.mock_actions_queue.put.assert_called_once()
        
        # Stop the controller
        self.controller.stop()

    async def test_control_task_updates_processed_data(self):
        """Test control_task updates processed data."""
        await self.controller.control_task()
        
        self.assertIn("pending_signals_count", self.controller.processed_data)

    async def test_on_stop_cancels_signal_processor(self):
        """Test that on_stop cancels the signal processor task."""
        self.controller.start()
        
        # Wait for task to start
        await asyncio.sleep(0.01)
        
        self.assertIsNotNone(self.controller._signal_processor_task)
        
        self.controller.stop()
        
        # Wait for stop to complete
        await asyncio.sleep(0.1)
        
        # Task should be done or cancelled
        self.assertTrue(
            self.controller._signal_processor_task.done() or
            self.controller._signal_processor_task.cancelled()
        )


class TestSignalType(IsolatedAsyncioWrapperTestCase):
    """Tests for SignalType enum."""

    def test_signal_values(self):
        """Test SignalType enum values."""
        self.assertEqual(SignalType.BUY.value, 1)
        self.assertEqual(SignalType.SELL.value, -1)
        self.assertEqual(SignalType.NEUTRAL.value, 0)

    def test_signal_comparison(self):
        """Test SignalType comparison."""
        self.assertNotEqual(SignalType.BUY, SignalType.SELL)
        self.assertNotEqual(SignalType.BUY, SignalType.NEUTRAL)
        self.assertEqual(SignalType.BUY, SignalType.BUY)
