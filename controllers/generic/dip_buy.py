"""
Dip Buy Strategy - Buy when price drops by a specified percentage within a time period.

This strategy monitors a trading pair and triggers a buy signal when the price
drops by a configured percentage within a specified lookback period.
"""
from decimal import Decimal
from typing import List, Optional

from pydantic import Field, field_validator
from pydantic_core.core_schema import ValidationInfo

from hummingbot.core.data_type.common import MarketDict
from hummingbot.data_feed.candles_feed.data_types import CandlesConfig
from hummingbot.strategy_v2.controllers.directional_trading_controller_base import (
    DirectionalTradingControllerBase,
    DirectionalTradingControllerConfigBase,
)


class DipBuyConfig(DirectionalTradingControllerConfigBase):
    """
    Configuration for Dip Buy Strategy.
    
    Buy when price drops by a specified percentage within a lookback period.
    """
    controller_name: str = "dip_buy"
    
    # Candles configuration
    candles_config: List[CandlesConfig] = []
    candles_connector: Optional[str] = Field(
        default=None,
        json_schema_extra={
            "prompt": "Enter the connector for candles data (leave empty to use same as trading connector): ",
            "prompt_on_new": True,
        }
    )
    candles_trading_pair: Optional[str] = Field(
        default=None,
        json_schema_extra={
            "prompt": "Enter the trading pair for candles data (leave empty to use same as trading pair): ",
            "prompt_on_new": True,
        }
    )
    
    # Strategy parameters
    interval: str = Field(
        default="1m",
        json_schema_extra={
            "prompt": "Enter the candle interval (e.g., 1m, 5m, 15m, 1h): ",
            "prompt_on_new": True,
        }
    )
    lookback_periods: int = Field(
        default=60,
        gt=0,
        json_schema_extra={
            "prompt": "Enter the number of candles to look back for dip detection (e.g., 60 for 60 candles): ",
            "prompt_on_new": True,
            "is_updatable": True,
        }
    )
    dip_threshold: Decimal = Field(
        default=Decimal("0.03"),
        gt=0,
        json_schema_extra={
            "prompt": "Enter the dip threshold as decimal (e.g., 0.03 for 3% drop): ",
            "prompt_on_new": True,
            "is_updatable": True,
        }
    )
    # Only buy, no sell signals from this strategy
    enable_short: bool = Field(
        default=False,
        json_schema_extra={
            "prompt": "Enable short signals on price surge? (true/false, default: false): ",
            "prompt_on_new": True,
        }
    )

    @field_validator("candles_connector", mode="before")
    @classmethod
    def set_candles_connector(cls, v, validation_info: ValidationInfo):
        if v is None or v == "":
            return validation_info.data.get("connector_name")
        return v

    @field_validator("candles_trading_pair", mode="before")
    @classmethod
    def set_candles_trading_pair(cls, v, validation_info: ValidationInfo):
        if v is None or v == "":
            return validation_info.data.get("trading_pair")
        return v

    @field_validator("dip_threshold", mode="before")
    @classmethod
    def validate_dip_threshold(cls, v):
        if isinstance(v, str):
            return Decimal(v)
        return v

    def update_markets(self, markets: MarketDict) -> MarketDict:
        return markets.add_or_update(self.connector_name, self.trading_pair)


class DipBuy(DirectionalTradingControllerBase):
    """
    Dip Buy Strategy Controller.
    
    Monitors price and generates buy signals when the price drops by a specified
    percentage within the lookback period.
    
    Features:
    - Configurable lookback period (in candles)
    - Configurable dip threshold (percentage drop)
    - Optional short signals on price surge
    - Uses high price within lookback period as reference
    """

    def __init__(self, config: DipBuyConfig, *args, **kwargs):
        self.config = config
        self.max_records = config.lookback_periods + 10
        
        # Initialize candles config if not provided
        if len(self.config.candles_config) == 0:
            self.config.candles_config = [
                CandlesConfig(
                    connector=config.candles_connector,
                    trading_pair=config.candles_trading_pair,
                    interval=config.interval,
                    max_records=self.max_records,
                )
            ]
        super().__init__(config, *args, **kwargs)

    async def update_processed_data(self):
        """
        Update processed data and generate trading signals.
        
        Signal logic:
        - Buy signal (1): When price drops by dip_threshold from highest price in lookback period
        - Sell signal (-1): When price rises by dip_threshold from lowest price (if enable_short=True)
        """
        # Get candles data
        df = self.market_data_provider.get_candles_df(
            connector_name=self.config.candles_connector,
            trading_pair=self.config.candles_trading_pair,
            interval=self.config.interval,
            max_records=self.max_records,
        )
        
        if df is None or len(df) < self.config.lookback_periods:
            self.processed_data["signal"] = 0
            self.processed_data["features"] = df if df is not None else None
            return
        
        # Calculate lookback high and low
        lookback_df = df.tail(self.config.lookback_periods)
        highest_price = lookback_df["high"].max()
        lowest_price = lookback_df["low"].min()
        current_price = df["close"].iloc[-1]
        
        # Calculate percentage change from high (for dip) and low (for surge)
        dip_percentage = (highest_price - current_price) / highest_price
        surge_percentage = (current_price - lowest_price) / lowest_price
        
        # Generate signals
        signal = 0
        
        # Buy signal: price dropped by threshold
        if dip_percentage >= self.config.dip_threshold:
            signal = 1
        # Sell signal: price surged by threshold (only if enabled)
        elif self.config.enable_short and surge_percentage >= self.config.dip_threshold:
            signal = -1
        
        # Store processed data
        self.processed_data["signal"] = signal
        self.processed_data["features"] = df
        self.processed_data["current_price"] = current_price
        self.processed_data["highest_price"] = highest_price
        self.processed_data["lowest_price"] = lowest_price
        self.processed_data["dip_percentage"] = dip_percentage
        self.processed_data["surge_percentage"] = surge_percentage

    def to_format_status(self) -> List[str]:
        """Format status for display."""
        lines = [
            f"Dip Buy Strategy: {self.config.trading_pair}",
            f"  Interval: {self.config.interval}",
            f"  Lookback: {self.config.lookback_periods} candles",
            f"  Dip Threshold: {float(self.config.dip_threshold) * 100:.2f}%",
        ]
        
        if "current_price" in self.processed_data:
            lines.extend([
                f"  Current Price: {self.processed_data['current_price']:.6f}",
                f"  Highest (lookback): {self.processed_data['highest_price']:.6f}",
                f"  Lowest (lookback): {self.processed_data['lowest_price']:.6f}",
                f"  Dip from High: {float(self.processed_data.get('dip_percentage', 0)) * 100:.2f}%",
                f"  Surge from Low: {float(self.processed_data.get('surge_percentage', 0)) * 100:.2f}%",
                f"  Signal: {'BUY' if self.processed_data['signal'] == 1 else 'SELL' if self.processed_data['signal'] == -1 else 'NEUTRAL'}",
            ])
        
        return lines
