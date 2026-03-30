"""
Signal Controller Base - Event-driven controller for triggering buy/sell actions via hooks.

This controller allows external events to trigger trading actions through hook functions,
making it suitable for integration with external signal sources, webhooks, or manual triggers.
"""
import asyncio
from decimal import Decimal
from enum import Enum
from typing import Callable, List, Optional

from pydantic import Field, field_validator

from hummingbot.core.data_type.common import MarketDict, OrderType, PositionMode, PriceType, TradeType
from hummingbot.core.utils.async_utils import safe_ensure_future
from hummingbot.strategy_v2.controllers.controller_base import ControllerBase, ControllerConfigBase
from hummingbot.strategy_v2.executors.data_types import ConnectorPair
from hummingbot.strategy_v2.executors.position_executor.data_types import (
    PositionExecutorConfig,
    TrailingStop,
    TripleBarrierConfig,
)
from hummingbot.strategy_v2.models.executor_actions import CreateExecutorAction, ExecutorAction, StopExecutorAction


class SignalType(Enum):
    """Enum for signal types."""
    BUY = 1
    SELL = -1
    NEUTRAL = 0


class SignalControllerConfigBase(ControllerConfigBase):
    """
    Configuration for Signal Controller.
    
    This controller is designed to be triggered by external signals/events
    rather than continuous market analysis.
    """
    controller_type: str = "signal_controller"
    connector_name: str = Field(
        default="binance_perpetual",
        json_schema_extra={
            "prompt": "Enter the connector name (e.g., binance_perpetual): ",
            "prompt_on_new": True
        }
    )
    trading_pair: str = Field(
        default="BTC-USDT",
        json_schema_extra={
            "prompt": "Enter the trading pair (e.g., BTC-USDT): ",
            "prompt_on_new": True
        }
    )
    max_executors_per_side: int = Field(
        default=3,
        json_schema_extra={
            "prompt": "Enter the maximum number of executors per side (e.g., 3): ",
            "prompt_on_new": True,
            "is_updatable": True
        }
    )
    cooldown_time: int = Field(
        default=60,
        gt=0,
        json_schema_extra={
            "prompt": "Enter the cooldown time in seconds between signals (e.g., 60): ",
            "prompt_on_new": True,
            "is_updatable": True
        }
    )
    leverage: int = Field(
        default=10,
        json_schema_extra={
            "prompt": "Enter the leverage (e.g., 10 for 10x, 1 for spot): ",
            "prompt_on_new": True
        }
    )
    position_mode: PositionMode = Field(
        default="HEDGE",
        json_schema_extra={"prompt": "Enter the position mode (HEDGE/ONEWAY): "}
    )
    # Triple Barrier Configuration
    stop_loss: Optional[Decimal] = Field(
        default=Decimal("0.03"),
        gt=0,
        json_schema_extra={
            "prompt": "Enter the stop loss (e.g., 0.03 for 3%): ",
            "prompt_on_new": True,
            "is_updatable": True
        }
    )
    take_profit: Optional[Decimal] = Field(
        default=Decimal("0.02"),
        gt=0,
        json_schema_extra={
            "prompt": "Enter the take profit (e.g., 0.02 for 2%): ",
            "prompt_on_new": True,
            "is_updatable": True
        }
    )
    time_limit: Optional[int] = Field(
        default=60 * 30,
        gt=0,
        json_schema_extra={
            "prompt": "Enter the time limit in seconds (e.g., 1800 for 30 min): ",
            "prompt_on_new": True,
            "is_updatable": True
        }
    )
    take_profit_order_type: OrderType = Field(
        default=OrderType.LIMIT,
        json_schema_extra={
            "prompt": "Enter the order type for take profit (LIMIT/MARKET): ",
            "prompt_on_new": True,
            "is_updatable": True
        }
    )
    trailing_stop: Optional[TrailingStop] = Field(
        default=None,
        json_schema_extra={
            "prompt": "Enter trailing stop as activation_price,trailing_delta (e.g., 0.015,0.003): ",
            "prompt_on_new": True,
            "is_updatable": True
        }
    )
    # Signal-specific config
    auto_close_on_opposite_signal: bool = Field(
        default=True,
        json_schema_extra={
            "prompt": "Auto close positions on opposite signal? (true/false): ",
            "prompt_on_new": True,
            "is_updatable": True
        }
    )

    @field_validator("trailing_stop", mode="before")
    @classmethod
    def parse_trailing_stop(cls, v):
        if isinstance(v, str):
            if v == "":
                return None
            activation_price, trailing_delta = v.split(",")
            return TrailingStop(
                activation_price=Decimal(activation_price),
                trailing_delta=Decimal(trailing_delta)
            )
        return v

    @field_validator("time_limit", "stop_loss", "take_profit", mode="before")
    @classmethod
    def validate_target(cls, v):
        if isinstance(v, str):
            if v == "":
                return None
            return Decimal(v)
        return v

    @field_validator("take_profit_order_type", mode="before")
    @classmethod
    def validate_order_type(cls, v) -> OrderType:
        if isinstance(v, OrderType):
            return v
        elif v is None:
            return OrderType.MARKET
        elif isinstance(v, str):
            cleaned_str = v.replace("OrderType.", "").upper()
            if cleaned_str in OrderType.__members__:
                return OrderType[cleaned_str]
        elif isinstance(v, int):
            try:
                return OrderType(v)
            except ValueError:
                pass
        raise ValueError(
            f"Invalid order type: {v}. Valid options: {', '.join(OrderType.__members__)}"
        )

    @field_validator("position_mode", mode="before")
    @classmethod
    def validate_position_mode(cls, v: str) -> PositionMode:
        if isinstance(v, str):
            if v.upper() in PositionMode.__members__:
                return PositionMode[v.upper()]
            raise ValueError(
                f"Invalid position mode: {v}. Valid options: {', '.join(PositionMode.__members__)}"
            )
        return v

    @property
    def triple_barrier_config(self) -> TripleBarrierConfig:
        return TripleBarrierConfig(
            stop_loss=self.stop_loss,
            take_profit=self.take_profit,
            time_limit=self.time_limit,
            trailing_stop=self.trailing_stop,
            open_order_type=OrderType.MARKET,
            take_profit_order_type=self.take_profit_order_type,
            stop_loss_order_type=OrderType.MARKET,
            time_limit_order_type=OrderType.MARKET,
        )

    def update_markets(self, markets: MarketDict) -> MarketDict:
        return markets.add_or_update(self.connector_name, self.trading_pair)


class SignalControllerBase(ControllerBase):
    """
    Event-driven Signal Controller.
    
    Provides hook functions for external events to trigger buy/sell actions.
    Supports both synchronous and asynchronous signal triggers.
    
    Usage:
        controller = SignalControllerBase(config, market_data_provider, actions_queue)
        
        # Hook-based triggers
        controller.on_buy_signal(amount=Decimal("0.1"))
        controller.on_sell_signal(amount=Decimal("0.1"))
        
        # Or use signal type
        controller.on_signal(SignalType.BUY)
        
        # Register custom hooks
        controller.register_pre_trade_hook(my_pre_trade_func)
        controller.register_post_trade_hook(my_post_trade_func)
    """

    def __init__(self, config: SignalControllerConfigBase, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.config = config
        self._pending_signals: asyncio.Queue = asyncio.Queue()
        self._last_signal_time: float = 0.0
        self._pre_trade_hooks: List[Callable] = []
        self._post_trade_hooks: List[Callable] = []
        self._signal_event: asyncio.Event = asyncio.Event()
        self._signal_processor_task: Optional[asyncio.Task] = None
        
        # Initialize rate sources
        self.market_data_provider.initialize_rate_sources([
            ConnectorPair(
                connector_name=config.connector_name,
                trading_pair=config.trading_pair
            )
        ])

    # ==================== Hook Registration ====================

    def register_pre_trade_hook(self, hook: Callable) -> None:
        """
        Register a pre-trade hook function.
        
        Hook signature: async def hook(trade_type: TradeType, amount: Decimal, price: Decimal) -> bool
        Returns True to proceed with trade, False to cancel.
        """
        self._pre_trade_hooks.append(hook)

    def register_post_trade_hook(self, hook: Callable) -> None:
        """
        Register a post-trade hook function.
        
        Hook signature: async def hook(trade_type: TradeType, amount: Decimal, price: Decimal, executor_id: str)
        """
        self._post_trade_hooks.append(hook)

    def unregister_pre_trade_hook(self, hook: Callable) -> None:
        """Remove a pre-trade hook."""
        if hook in self._pre_trade_hooks:
            self._pre_trade_hooks.remove(hook)

    def unregister_post_trade_hook(self, hook: Callable) -> None:
        """Remove a post-trade hook."""
        if hook in self._post_trade_hooks:
            self._post_trade_hooks.remove(hook)

    # ==================== Signal Triggers (Public Hooks) ====================

    def on_buy_signal(self, amount: Optional[Decimal] = None) -> bool:
        """
        Hook to trigger a buy signal.
        
        Args:
            amount: Optional amount to trade. If not provided, uses configured amount.
            
        Returns:
            True if signal was queued successfully, False if in cooldown.
        """
        return self._queue_signal(SignalType.BUY, amount)

    def on_sell_signal(self, amount: Optional[Decimal] = None) -> bool:
        """
        Hook to trigger a sell signal.
        
        Args:
            amount: Optional amount to trade. If not provided, uses configured amount.
            
        Returns:
            True if signal was queued successfully, False if in cooldown.
        """
        return self._queue_signal(SignalType.SELL, amount)

    def on_signal(self, signal: SignalType, amount: Optional[Decimal] = None) -> bool:
        """
        Hook to trigger a signal by type.
        
        Args:
            signal: The signal type (BUY, SELL, or NEUTRAL).
            amount: Optional amount to trade.
            
        Returns:
            True if signal was queued successfully, False if in cooldown or neutral.
        """
        return self._queue_signal(signal, amount)

    def on_close_all_positions(self) -> None:
        """Hook to close all active positions."""
        self._close_all_executors()

    # ==================== Internal Signal Processing ====================

    def _queue_signal(self, signal: SignalType, amount: Optional[Decimal] = None) -> bool:
        """Queue a signal for processing and trigger immediate processing."""
        if signal == SignalType.NEUTRAL:
            return False
            
        current_time = self.market_data_provider.time()
        if current_time - self._last_signal_time < self.config.cooldown_time:
            self.logger().debug(
                f"Signal ignored - in cooldown period. "
                f"Time since last: {current_time - self._last_signal_time}s, "
                f"Cooldown: {self.config.cooldown_time}s"
            )
            return False
            
        self._pending_signals.put_nowait((signal, amount))
        # Trigger immediate signal processing (no delay)
        self._signal_event.set()
        return True

    async def _process_signals(self) -> List[ExecutorAction]:
        """Process all pending signals and return actions."""
        actions = []
        
        while not self._pending_signals.empty():
            signal, amount = await self._pending_signals.get()
            
            # Run pre-trade hooks
            should_proceed = await self._run_pre_trade_hooks(signal, amount)
            if not should_proceed:
                self.logger().info(f"Trade cancelled by pre-trade hook for signal: {signal}")
                continue
                
            # Create the action
            action = await self._create_action_from_signal(signal, amount)
            if action:
                actions.append(action)
                self._last_signal_time = self.market_data_provider.time()
                
                # Run post-trade hooks
                await self._run_post_trade_hooks(signal, amount, action)
                
        return actions

    async def _run_pre_trade_hooks(self, signal: SignalType, amount: Optional[Decimal]) -> bool:
        """Run all pre-trade hooks. Returns False if any hook cancels the trade."""
        for hook in self._pre_trade_hooks:
            try:
                trade_type = TradeType.BUY if signal == SignalType.BUY else TradeType.SELL
                price = self.market_data_provider.get_price_by_type(
                    self.config.connector_name,
                    self.config.trading_pair,
                    PriceType.MidPrice
                )
                actual_amount = amount or self._get_default_amount()
                
                if asyncio.iscoroutinefunction(hook):
                    result = await hook(trade_type, actual_amount, price)
                else:
                    result = hook(trade_type, actual_amount, price)
                    
                if result is False:
                    return False
            except Exception as e:
                self.logger().error(f"Error in pre-trade hook: {e}")
                
        return True

    async def _run_post_trade_hooks(
        self, signal: SignalType, amount: Optional[Decimal], action: ExecutorAction
    ) -> None:
        """Run all post-trade hooks after a trade is created."""
        for hook in self._post_trade_hooks:
            try:
                trade_type = TradeType.BUY if signal == SignalType.BUY else TradeType.SELL
                price = self.market_data_provider.get_price_by_type(
                    self.config.connector_name,
                    self.config.trading_pair,
                    PriceType.MidPrice
                )
                actual_amount = amount or self._get_default_amount()
                executor_id = getattr(action, "executor_config", {}).get("id", "unknown")
                
                if asyncio.iscoroutinefunction(hook):
                    await hook(trade_type, actual_amount, price, executor_id)
                else:
                    hook(trade_type, actual_amount, price, executor_id)
            except Exception as e:
                self.logger().error(f"Error in post-trade hook: {e}")

    def _get_default_amount(self) -> Decimal:
        """Calculate the default trade amount based on config."""
        price = self.market_data_provider.get_price_by_type(
            self.config.connector_name,
            self.config.trading_pair,
            PriceType.MidPrice
        )
        return self.config.total_amount_quote / price / Decimal(self.config.max_executors_per_side)

    async def _create_action_from_signal(
        self, signal: SignalType, amount: Optional[Decimal] = None
    ) -> Optional[ExecutorAction]:
        """Create an executor action from a signal."""
        trade_type = TradeType.BUY if signal == SignalType.BUY else TradeType.SELL
        
        if not self._can_create_executor(trade_type):
            self.logger().debug(f"Cannot create executor for {trade_type}")
            return None
            
        # Close opposite positions if configured
        if self.config.auto_close_on_opposite_signal:
            self._close_opposite_executors(trade_type)
            
        price = self.market_data_provider.get_price_by_type(
            self.config.connector_name,
            self.config.trading_pair,
            PriceType.MidPrice
        )
        
        actual_amount = amount or self._get_default_amount()
        
        executor_config = self._get_executor_config(trade_type, price, actual_amount)
        
        return CreateExecutorAction(
            controller_id=self.config.id,
            executor_config=executor_config
        )

    def _can_create_executor(self, trade_type: TradeType) -> bool:
        """Check if a new executor can be created."""
        active_executors = self.filter_executors(
            executors=self.executors_info,
            filter_func=lambda x: x.is_active and x.side == trade_type
        )
        
        max_timestamp = max(
            [executor.timestamp for executor in active_executors],
            default=0
        )
        
        active_executors_ok = len(active_executors) < self.config.max_executors_per_side
        cooldown_ok = self.market_data_provider.time() - max_timestamp >= self.config.cooldown_time
        
        return active_executors_ok and cooldown_ok

    def _close_opposite_executors(self, trade_type: TradeType) -> None:
        """Close executors on the opposite side."""
        opposite_side = TradeType.SELL if trade_type == TradeType.BUY else TradeType.BUY
        opposite_executors = self.filter_executors(
            executors=self.executors_info,
            filter_func=lambda x: x.is_active and x.side == opposite_side
        )
        
        for executor in opposite_executors:
            self._stop_executor(executor.id)

    def _close_all_executors(self) -> None:
        """Close all active executors."""
        active_executors = self.filter_executors(
            executors=self.executors_info,
            filter_func=lambda x: x.is_active
        )
        
        for executor in active_executors:
            self._stop_executor(executor.id)

    def _stop_executor(self, executor_id: str) -> None:
        """Queue a stop action for an executor."""
        action = StopExecutorAction(
            controller_id=self.config.id,
            executor_id=executor_id
        )
        safe_ensure_future(self.actions_queue.put([action]))

    def _get_executor_config(
        self, trade_type: TradeType, price: Decimal, amount: Decimal
    ) -> PositionExecutorConfig:
        """Create executor configuration."""
        return PositionExecutorConfig(
            timestamp=self.market_data_provider.time(),
            connector_name=self.config.connector_name,
            trading_pair=self.config.trading_pair,
            side=trade_type,
            entry_price=price,
            amount=amount,
            triple_barrier_config=self.config.triple_barrier_config,
            leverage=self.config.leverage,
        )

    # ==================== Controller Base Overrides ====================

    def start(self):
        """Start the controller and signal processor task."""
        super().start()
        # Start the dedicated signal processor task for immediate event handling
        if self._signal_processor_task is None or self._signal_processor_task.done():
            self._signal_processor_task = safe_ensure_future(self._signal_processor_loop())

    def on_stop(self):
        """Clean up the signal processor task on stop."""
        if self._signal_processor_task and not self._signal_processor_task.done():
            self._signal_processor_task.cancel()
        self._signal_event.set()  # Wake up the processor to exit

    async def _signal_processor_loop(self):
        """
        Dedicated loop for immediate signal processing.
        This runs independently of the control_loop to avoid delays.
        """
        while not self.terminated.is_set():
            try:
                # Wait for signal event (no polling delay)
                await self._signal_event.wait()
                self._signal_event.clear()
                
                # Process all pending signals immediately
                if self.market_data_provider.ready and self.executors_update_event.is_set():
                    actions = await self._process_signals()
                    if len(actions) > 0:
                        self.logger().debug(f"Sending signal-based actions: {actions}")
                        await self.send_actions(actions)
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger().error(f"Error in signal processor loop: {e}", exc_info=True)
                await asyncio.sleep(0.1)  # Brief pause on error

    async def update_processed_data(self):
        """Update processed data for status display."""
        self.processed_data = {
            "pending_signals_count": self._pending_signals.qsize(),
            "last_signal_time": self._last_signal_time,
        }

    def determine_executor_actions(self) -> List[ExecutorAction]:
        """
        Signal-based actions are handled by _signal_processor_loop.
        This method returns empty for the control_loop.
        """
        return []

    async def control_task(self):
        """
        Lightweight control task for status updates only.
        Signal processing is handled by _signal_processor_loop for immediate response.
        """
        await self.update_processed_data()

    def to_format_status(self) -> List[str]:
        """Format controller status for display."""
        lines = [
            f"Signal Controller: {self.config.id}",
            f"  Connector: {self.config.connector_name}",
            f"  Trading Pair: {self.config.trading_pair}",
            f"  Pending Signals: {self._pending_signals.qsize()}",
            f"  Last Signal Time: {self._last_signal_time}",
            f"  Active Executors: {len([e for e in self.executors_info if e.is_active])}",
        ]
        return lines
