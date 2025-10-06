import asyncio
import logging
from decimal import Decimal, InvalidOperation
from typing import Dict

from hummingbot.connector.connector_base import ConnectorBase, Union
from hummingbot.connector.utils import split_hb_trading_pair
from hummingbot.core.data_type.common import OrderType, PriceType, TradeType
from hummingbot.core.data_type.funding_info import FundingInfo
from hummingbot.core.data_type.order_candidate import OrderCandidate, PerpetualOrderCandidate, PositionAction
from hummingbot.core.event.events import (
    BuyOrderCompletedEvent,
    BuyOrderCreatedEvent,
    MarketOrderFailureEvent,
    SellOrderCompletedEvent,
    SellOrderCreatedEvent, OrderCancelledEvent, FundingPaymentCompletedEvent,
)
from hummingbot.core.rate_oracle.rate_oracle import RateOracle
from hummingbot.logger import HummingbotLogger
from hummingbot.strategy.script_strategy_base import ScriptStrategyBase
from hummingbot.strategy_v2.executors.perpetual_executor_base import PerpetualExecutorBase
from hummingbot.strategy_v2.executors.xemm_executor.data_types import XEMMExecutorConfig
from hummingbot.strategy_v2.models.base import RunnableStatus
from hummingbot.strategy_v2.models.executors import CloseType, TrackedOrder


class XEMMExecutor(PerpetualExecutorBase):
    _logger = None

    @classmethod
    def logger(cls) -> HummingbotLogger:
        if cls._logger is None:
            cls._logger = logging.getLogger(__name__)
        return cls._logger

    @staticmethod
    def _are_tokens_interchangeable(first_token: str, second_token: str):
        interchangeable_tokens = [
            {"WETH", "ETH"},
            {"WBTC", "BTC"},
            {"WBNB", "BNB"},
            {"WPOL", "POL"},
            {"WAVAX", "AVAX"},
            {"WONE", "ONE"},
            {"USDC", "USDC.E"},
            {"WBTC", "BTC"},
            {"USOL", "SOL"},
            {"UETH", "ETH"},
            {"UBTC", "BTC"}
        ]
        same_token_condition = first_token == second_token
        tokens_interchangeable_condition = any(({first_token, second_token} <= interchangeable_pair
                                                for interchangeable_pair
                                                in interchangeable_tokens))
        # for now, we will consider all the stablecoins interchangeable
        stable_coins_condition = "USD" in first_token and "USD" in second_token
        return same_token_condition or tokens_interchangeable_condition or stable_coins_condition

    def is_arbitrage_valid(self, pair1, pair2):
        base_asset1, _ = split_hb_trading_pair(pair1)
        base_asset2, _ = split_hb_trading_pair(pair2)
        return self._are_tokens_interchangeable(base_asset1, base_asset2)

    def __init__(self, strategy: ScriptStrategyBase, config: XEMMExecutorConfig, update_interval: float = 1,
                 max_retries: int = 3):
        if not self.is_arbitrage_valid(pair1=config.buying_market.trading_pair,
                                       pair2=config.selling_market.trading_pair):
            raise Exception("XEMM is not valid since the trading pairs are not interchangeable.")
        self.config = config
        self.rate_oracle = RateOracle.get_instance()
        if config.maker_side == TradeType.BUY:
            self.maker_connector = config.buying_market.connector_name
            self.maker_trading_pair = config.buying_market.trading_pair
            self.maker_order_side = TradeType.BUY
            self.taker_connector = config.selling_market.connector_name
            self.taker_trading_pair = config.selling_market.trading_pair
            self.taker_order_side = TradeType.SELL
        else:
            self.maker_connector = config.selling_market.connector_name
            self.maker_trading_pair = config.selling_market.trading_pair
            self.maker_order_side = TradeType.SELL
            self.taker_connector = config.buying_market.connector_name
            self.taker_trading_pair = config.buying_market.trading_pair
            self.taker_order_side = TradeType.BUY

        # Set up quote conversion pair
        _, maker_quote = split_hb_trading_pair(self.maker_trading_pair)
        _, taker_quote = split_hb_trading_pair(self.taker_trading_pair)
        self.quote_conversion_pair = f"{taker_quote}-{maker_quote}"

        taker_connector = strategy.connectors[self.taker_connector]
        if not self.is_amm_connector(exchange=self.taker_connector):
            if OrderType.MARKET not in taker_connector.supported_order_types():
                raise ValueError(f"{self.taker_connector} does not support market orders.")
        self._taker_result_price = Decimal("1")
        self._maker_target_price = Decimal("1")
        self._tx_cost = Decimal("1")
        self._tx_cost_pct = Decimal("1")
        self._current_trade_profitability = Decimal("0")
        self.maker_order = None
        self.taker_order = None
        self.maker_close_order = None
        self.taker_close_order = None
        self._current_retries = 0
        self._max_retries = max_retries

        self._arbitrage_close_flag = False
        self._trade_close_pnl_pct = Decimal('0')
        self._funding_payment_profit = Decimal('0')
        self._next_funding_profit = Decimal('0')

        super().__init__(strategy=strategy,
                         connectors=[config.buying_market.connector_name, config.selling_market.connector_name],
                         config=config, update_interval=update_interval)
        self.to_canceled_orders: list[TrackedOrder] = []

    async def validate_sufficient_balance(self):
        mid_price = self.get_price(self.maker_connector, self.maker_trading_pair,
                                   price_type=PriceType.MidPrice)
        maker_order_candidate = OrderCandidate(
            trading_pair=self.maker_trading_pair,
            is_maker=True,
            order_type=OrderType.LIMIT,
            order_side=self.maker_order_side,
            amount=self.config.order_amount,
            price=mid_price, ) if not self.is_perpetual_connector(self.maker_connector) else \
            PerpetualOrderCandidate(
                trading_pair=self.maker_trading_pair,
                is_maker=True,
                order_type=OrderType.LIMIT,
                order_side=self.maker_order_side,
                amount=self.config.order_amount,
                price=mid_price,
                leverage=Decimal(self.config.maker_leverage),
            )
        taker_order_candidate = OrderCandidate(
            trading_pair=self.taker_trading_pair,
            is_maker=False,
            order_type=OrderType.MARKET,
            order_side=self.taker_order_side,
            amount=self.config.order_amount,
            price=mid_price, ) if not self.is_perpetual_connector(self.taker_connector) else \
            PerpetualOrderCandidate(
                trading_pair=self.taker_trading_pair,
                is_maker=False,
                order_type=OrderType.MARKET,
                order_side=self.taker_order_side,
                amount=self.config.order_amount,
                price=mid_price,
                leverage=Decimal(self.config.taker_leverage),
            )
        maker_adjusted_candidate = self.adjust_order_candidates(self.maker_connector, [maker_order_candidate])[0]
        taker_adjusted_candidate = self.adjust_order_candidates(self.taker_connector, [taker_order_candidate])[0]
        if maker_adjusted_candidate.amount == Decimal("0") or taker_adjusted_candidate.amount == Decimal("0"):
            self.close_type = CloseType.INSUFFICIENT_BALANCE
            self.logger().error(
                f"{self.maker_connector if maker_adjusted_candidate.amount == Decimal('0') else self.taker_connector} don't enough budget to open position.")
            self.stop()

    async def control_task(self):
        await self.update_trade_close_pnl_pct()
        await self.update_funding_rate_profit()
        if self.status == RunnableStatus.RUNNING:
            await self.update_prices_and_tx_costs()
            await self.control_maker_order()
        elif self.status == RunnableStatus.SHUTTING_DOWN:
            if self._current_retries > self._max_retries:
                self.close_type = CloseType.FAILED
                self.stop()
            await self.control_shutdown_process()

    async def update_trade_close_pnl_pct(self):
        if self.maker_order_side == TradeType.BUY:
            buy_connector = self.taker_connector
            buy_trading_pair = self.taker_trading_pair
            sell_connector = self.maker_connector
            sell_trading_pair = self.maker_trading_pair
        else:
            buy_connector = self.maker_connector
            buy_trading_pair = self.maker_trading_pair
            sell_connector = self.taker_connector
            sell_trading_pair = self.taker_trading_pair
        buy_price_task = asyncio.create_task(self.get_resulting_price_for_amount(
            connector=buy_connector,
            trading_pair=buy_trading_pair,
            is_buy=False,
            order_amount=self.config.order_amount))
        sell_price_task = asyncio.create_task(self.get_resulting_price_for_amount(
            connector=sell_connector,
            trading_pair=sell_trading_pair,
            is_buy=True,
            order_amount=self.config.order_amount))

        buy_close_price, sell_close_price = await asyncio.gather(sell_price_task, buy_price_task)
        if not buy_close_price or not sell_close_price:
            raise Exception("Could not get buy and sell close prices")
        self._trade_close_pnl_pct = (buy_close_price - sell_close_price) / sell_close_price

    def __get_current_profit__(self):
        """
        Return: 基于已成交订单及已经支付的资金费还有实时平仓价格后计算的收益率（未包含即将到来的资金费）
        """
        if self.taker_order_side == TradeType.BUY:
            sell_order = self.maker_order
            buy_order = self.taker_order
        else:
            sell_order = self.taker_order
            buy_order = self.maker_order
        sell_quote_amount = sell_order.order.executed_amount_base * sell_order.average_executed_price
        buy_quote_amount = buy_order.order.executed_amount_base * buy_order.average_executed_price
        if any(x.is_nan() for x in
               [sell_quote_amount, buy_quote_amount, self.cum_fees_quote, self._funding_payment_profit,
                self._trade_close_pnl_pct]):
            return Decimal("0")
        return sell_quote_amount - buy_quote_amount - self.cum_fees_quote + self._funding_payment_profit \
            + self.config.order_amount * self._trade_close_pnl_pct

    async def control_maker_order(self):
        if self.maker_order is None:
            await self.create_maker_order()
        else:
            await self.control_update_maker_order()

    async def update_funding_rate_profit(self):
        ret = 0
        if self.is_perpetual_connector(self.taker_connector):
            funding_rate: FundingInfo = self.connectors[self.taker_connector].get_funding_info(
                self.taker_trading_pair
            )
            ret += funding_rate.rate * (Decimal('-1') if self.taker_order_side == TradeType.BUY else Decimal('1'))
        if self.is_perpetual_connector(self.maker_connector):
            funding_rate: FundingInfo = self.connectors[self.maker_connector].get_funding_info(
                self.maker_trading_pair
            )
            ret += funding_rate.rate * (Decimal('1') if self.maker_order_side == TradeType.BUY else Decimal('-1'))
        self._next_funding_profit = ret

    async def update_prices_and_tx_costs(self):
        self._taker_result_price = await self.get_resulting_price_for_amount(
            connector=self.taker_connector,
            trading_pair=self.taker_trading_pair,
            is_buy=self.taker_order_side == TradeType.BUY,
            order_amount=self.config.order_amount)
        await self.update_tx_costs()
        if any(x.is_nan() for x in [self._taker_result_price, self._tx_cost_pct, self._next_funding_profit]):
            return
        if self.taker_order_side == TradeType.BUY:
            self._maker_target_price = self._taker_result_price * (
                    1 + self.config.target_profitability + self._tx_cost_pct - self._next_funding_profit)
        else:
            self._maker_target_price = self._taker_result_price * (
                    1 - self.config.target_profitability - self._tx_cost_pct + self._next_funding_profit)

    async def update_tx_costs(self):
        base, quote = split_hb_trading_pair(trading_pair=self.config.buying_market.trading_pair)
        base_without_wrapped = base[1:] if base.startswith("W") else base
        taker_fee_task = asyncio.create_task(self.get_tx_cost_in_asset(
            exchange=self.taker_connector,
            trading_pair=self.taker_trading_pair,
            order_type=OrderType.MARKET,
            is_buy=self.taker_order_side == TradeType.BUY,
            order_amount=self.config.order_amount,
            asset=base_without_wrapped
        ))
        maker_fee_task = asyncio.create_task(self.get_tx_cost_in_asset(
            exchange=self.maker_connector,
            trading_pair=self.maker_trading_pair,
            order_type=OrderType.LIMIT,
            is_buy=self.maker_order_side == TradeType.BUY,
            order_amount=self.config.order_amount,
            asset=base_without_wrapped
        ))
        maker_close_fee_task = asyncio.create_task(self.get_tx_cost_in_asset(
            exchange=self.maker_connector,
            trading_pair=self.maker_trading_pair,
            order_type=OrderType.LIMIT_MAKER,
            is_buy=self.maker_order_side == TradeType.SELL,
            order_amount=self.config.order_amount,
            asset=base_without_wrapped
        ))
        taker_fee, maker_fee, maker_close_fee = await asyncio.gather(taker_fee_task, maker_fee_task,
                                                                     maker_close_fee_task)
        self._tx_cost = 2 * taker_fee + maker_fee + maker_close_fee
        self._tx_cost_pct = self._tx_cost / self.config.order_amount

    async def get_tx_cost_in_asset(self, exchange: str, trading_pair: str, is_buy: bool, order_amount: Decimal,
                                   asset: str, order_type: OrderType = OrderType.MARKET):
        connector = self.connectors[exchange]
        if self.is_amm_connector(exchange=exchange):
            gas_cost = connector.network_transaction_fee
            conversion_price = RateOracle.get_instance().get_pair_rate(f"{asset}-{gas_cost.token}")
            if conversion_price is None:
                self.logger().warning(f"Could not get conversion rate for {asset}-{gas_cost.token}")
                return Decimal("0")
            return gas_cost.amount / conversion_price
        else:
            fee = connector.get_fee(
                base_currency=asset,
                quote_currency=trading_pair.split("-")[1],
                order_type=order_type,
                order_side=TradeType.BUY if is_buy else TradeType.SELL,
                amount=order_amount,
                price=self._taker_result_price,
                is_maker=order_type.is_limit_type(),
            ) if not self.is_perpetual_connector(connector.name) else \
                connector.get_fee(
                    base_currency=asset,
                    quote_currency=trading_pair.split("-")[1],
                    order_type=order_type,
                    order_side=TradeType.BUY if is_buy else TradeType.SELL,
                    amount=order_amount,
                    price=self._taker_result_price,
                    is_maker=order_type.is_limit_type(),
                    position_action=PositionAction.OPEN
                )
            return fee.fee_amount_in_token(
                trading_pair=trading_pair,
                price=self._taker_result_price,
                order_amount=order_amount,
                token=asset,
                exchange=connector,
            )

    async def get_resulting_price_for_amount(self, connector: str, trading_pair: str, is_buy: bool,
                                             order_amount: Decimal):
        return await self.connectors[connector].get_quote_price(trading_pair, is_buy, order_amount)

    async def create_maker_order(self):
        order_id = self.place_order(
            connector_name=self.maker_connector,
            trading_pair=self.maker_trading_pair,
            order_type=OrderType.LIMIT_MAKER,
            side=self.maker_order_side,
            amount=self.config.order_amount,
            price=self._maker_target_price,
            position_action=PositionAction.OPEN if self.is_perpetual_connector(
                self.maker_connector) else PositionAction.NIL)
        self.maker_order = TrackedOrder(order_id=order_id)
        self.logger().info(f"Created maker order {order_id} at price {self._maker_target_price}.")

    async def control_shutdown_process(self):
        if self.maker_order.is_done and self.taker_order.is_done:
            if self.is_perpetual_connector(self.taker_connector) and \
                    self.is_perpetual_connector(self.maker_connector):
                if not self._arbitrage_close_flag:
                    current_profit = self.__get_current_profit__()
                    if (current_profit + self._next_funding_profit) < self.config.min_profitability * Decimal(
                            '0.5') < current_profit:
                        self._arbitrage_close_flag = True
                    if current_profit > self.config.min_profitability * Decimal('0.9'):
                        self._arbitrage_close_flag = True
                    if self._arbitrage_close_flag:
                        self.place_taker_close_order()
                        self.place_maker_close_order()
                else:
                    if self.maker_close_order.order and self.maker_close_order.is_filled and \
                            self.taker_close_order and self.taker_close_order.is_filled:
                        self.close_type = CloseType.COMPLETED
                        self.stop()
            else:
                self.logger().info("Both orders are done, executor terminated.")
                self.close_type = CloseType.COMPLETED
                self.stop()

    async def control_update_maker_order(self):
        await self.update_current_trade_profitability()
        if self._current_trade_profitability - self._tx_cost_pct < self.config.min_profitability:
            self.logger().info(
                f"Trade profitability {self._current_trade_profitability - self._tx_cost_pct} is below minimum profitability. Cancelling order.")
            self._strategy.cancel(self.maker_connector, self.maker_trading_pair, self.maker_order.order_id)
            self.to_canceled_orders.append(self.maker_order)
        elif self._current_trade_profitability - self._tx_cost_pct > self.config.max_profitability:
            self.logger().info(
                f"Trade profitability {self._current_trade_profitability - self._tx_cost_pct} is above target profitability. Cancelling order.")
            self._strategy.cancel(self.maker_connector, self.maker_trading_pair, self.maker_order.order_id)
            self.to_canceled_orders.append(self.maker_order)
        if len(self.to_canceled_orders):
            self.to_canceled_orders = [order for order in self.to_canceled_orders if order.is_open]
            for order in self.to_canceled_orders:
                self._strategy.cancel(self.maker_connector, self.maker_trading_pair, order.order_id)

    async def update_current_trade_profitability(self):
        trade_profitability = Decimal("0")
        if self.maker_order and self.maker_order.order and self.maker_order.order.is_open:
            maker_price = self.maker_order.order.price
            # Get the conversion rate to normalize prices to the same quote asset
            try:
                conversion_rate = await self.get_quote_asset_conversion_rate()
                if self.maker_order_side == TradeType.BUY:
                    # If maker is buying, normalize taker (sell) price to maker quote asset
                    normalized_taker_price = self._taker_result_price * conversion_rate
                    trade_profitability = (normalized_taker_price - maker_price) / maker_price
                else:
                    # If maker is selling, normalize taker (buy) price to maker quote asset
                    normalized_taker_price = self._taker_result_price * conversion_rate
                    trade_profitability = (maker_price - normalized_taker_price) / maker_price
            except Exception as e:
                self.logger().error(f"Error calculating trade profitability: {e}")
                return Decimal("0")
        self._current_trade_profitability = trade_profitability
        return trade_profitability

    def process_order_created_event(self,
                                    event_tag: int,
                                    market: ConnectorBase,
                                    event: Union[BuyOrderCreatedEvent, SellOrderCreatedEvent]):
        if self.maker_order and event.order_id == self.maker_order.order_id:
            self.logger().info(f"Maker order {event.order_id} created.")
            self.maker_order.order = self.get_in_flight_order(self.maker_connector, event.order_id)
        elif self.taker_order and event.order_id == self.taker_order.order_id:
            self.logger().info(f"Taker order {event.order_id} created.")
            self.taker_order.order = self.get_in_flight_order(self.taker_connector, event.order_id)
        elif self.maker_close_order and event.order_id == self.maker_close_order.order_id:
            self.logger().info(f"Maker close order {event.order_id} created.")
            self.maker_close_order.order = self.get_in_flight_order(self.maker_connector, event.order_id)
        elif self.taker_close_order and event.order_id == self.taker_close_order.order_id:
            self.logger().info(f"Taker close order {event.order_id} created.")
            self.taker_close_order.order = self.get_in_flight_order(self.taker_connector, event.order_id)

    def process_order_completed_event(self,
                                      event_tag: int,
                                      market: ConnectorBase,
                                      event: Union[BuyOrderCompletedEvent, SellOrderCompletedEvent]):
        if self.maker_order and event.order_id == self.maker_order.order_id:
            self._status = RunnableStatus.SHUTTING_DOWN
            self.logger().info(f"Maker order {event.order_id} completed. Executing taker order.")
            self.place_taker_order()

    def process_order_failed_event(self, _, market, event: MarketOrderFailureEvent):
        if self.taker_order and self.taker_order.order_id == event.order_id:
            self.place_taker_order()
            self._current_retries += 1
        elif self.taker_close_order and self.taker_close_order.order_id == event.order_id:
            self.place_taker_close_order()
            self._current_retries += 1
        elif self.maker_close_order and self.maker_close_order.order_id == event.order_id:
            self.place_maker_close_order()
            self._current_retries += 1
        elif self.maker_order and self.maker_order.order_id == event.order_id:
            self.maker_order = None
            self.to_canceled_orders = []

    def process_order_canceled_event(self,
                                     event_tag: int,
                                     market: ConnectorBase,
                                     event: OrderCancelledEvent):
        if self.maker_order and event.order_id == self.maker_order.order_id:
            self.maker_order = None
            self.to_canceled_orders = []

    def place_taker_order(self):
        taker_order_id = self.place_order(
            connector_name=self.taker_connector,
            trading_pair=self.taker_trading_pair,
            order_type=OrderType.MARKET,
            side=self.taker_order_side,
            amount=self.config.order_amount,
            position_action=PositionAction.OPEN)
        self.taker_order = TrackedOrder(order_id=taker_order_id)

    def place_maker_close_order(self):
        maker_close_order_id = self.place_order(
            connector_name=self.maker_connector,
            trading_pair=self.maker_trading_pair,
            order_type=OrderType.MARKET,
            side=self.taker_order_side,
            amount=self.config.order_amount,
            position_action=PositionAction.CLOSE)
        self.maker_close_order = TrackedOrder(order_id=maker_close_order_id)

    def place_taker_close_order(self):
        taker_close_order_id = self.place_order(
            connector_name=self.taker_connector,
            trading_pair=self.taker_trading_pair,
            order_type=OrderType.MARKET,
            side=self.maker_order_side,
            amount=self.config.order_amount,
            position_action=PositionAction.CLOSE
        )
        self.taker_close_order = TrackedOrder(order_id=taker_close_order_id)

    def process_funding_payment_event(self, _, market, event: FundingPaymentCompletedEvent):
        if self.taker_order_side == TradeType.BUY:
            buy_connector = self.taker_connector
            buy_trading_pair = self.taker_trading_pair
            sell_connector = self.maker_connector
            sell_trading_pair = self.maker_trading_pair
        else:
            buy_connector = self.maker_connector
            buy_trading_pair = self.maker_trading_pair
            sell_connector = self.taker_connector
            sell_trading_pair = self.taker_trading_pair

        if buy_connector == event.market and \
                buy_trading_pair == event.trading_pair:
            self._funding_payment_profit -= event.funding_rate * self.config.order_amount
        if sell_connector == event.market and \
                sell_trading_pair == event.trading_pair:
            self._funding_payment_profit += event.funding_rate * self.config.order_amount

    def get_custom_info(self) -> Dict:
        # Since we can't make this method async, we'll skip the profitability calculation
        # The profitability will still be shown in the status message which is async
        return {
            "side": self.config.maker_side,
            "maker_connector": self.maker_connector,
            "maker_trading_pair": self.maker_trading_pair,
            "taker_connector": self.taker_connector,
            "taker_trading_pair": self.taker_trading_pair,
            "min_profitability": self.config.min_profitability,
            "target_profitability_pct": self.config.target_profitability,
            "max_profitability": self.config.max_profitability,
            "trade_profitability": self._current_trade_profitability,
            "tx_cost": self._tx_cost,
            "tx_cost_pct": self._tx_cost_pct,
            "taker_price": self._taker_result_price,
            "maker_target_price": self._maker_target_price,
            "net_profitability": self._current_trade_profitability - self._tx_cost_pct,
            "order_amount": self.config.order_amount,
        }

    def early_stop(self, keep_position: bool = False):
        if self.maker_order and self.maker_order.order and self.maker_order.order.is_open:
            self.logger().info(f"Cancelling maker order {self.maker_order.order_id} for early stop.")
            self._strategy.cancel(self.maker_connector, self.maker_trading_pair, self.maker_order.order_id)
            self.to_canceled_orders.append(self.maker_order)
        if not self.maker_close_order.order:
            self.place_maker_close_order()
        if not self.taker_close_order.order:
            self.place_taker_close_order()
        self.close_type = CloseType.EARLY_STOP
        self.stop()

    def get_cum_fees_quote(self) -> Decimal:
        if self.close_type == CloseType.COMPLETED:
            return self.maker_order.cum_fees_quote + self.taker_order.cum_fees_quote + \
                self.maker_close_order.cum_fees_quote + self.taker_close_order.cum_fees_quote
        else:
            return Decimal("0")

    def get_net_pnl_quote(self) -> Decimal:
        if self.close_type == CloseType.COMPLETED:
            is_maker_buy = self.maker_order_side == TradeType.BUY
            maker_pnl = self.maker_order.executed_amount_base * self.maker_order.average_executed_price
            taker_pnl = self.taker_order.executed_amount_base * self.taker_order.average_executed_price
            maker_close_pnl = self.maker_close_order.executed_amount_base * self.maker_close_order.average_executed_price
            taker_close_pnl = self.taker_close_order.executed_amount_base * self.taker_close_order.average_executed_price
            if is_maker_buy:
                return maker_close_pnl - taker_close_pnl + taker_pnl - maker_pnl - self.get_cum_fees_quote()
            else:
                return taker_close_pnl - maker_close_pnl + maker_pnl - taker_pnl - self.get_cum_fees_quote()
        else:
            return Decimal("0")

    def get_net_pnl_pct(self) -> Decimal:
        pnl_quote = self.get_net_pnl_quote()
        return pnl_quote / self.maker_order.executed_amount_quote

    async def get_quote_asset_conversion_rate(self) -> Decimal:
        """
        Fetch the conversion rate between the quote assets of the buying and selling markets.
        Example: For M3M3/USDT and M3M3/USDC, fetch the USDC/USDT rate.
        """
        try:
            taker_quote, maker_quote = split_hb_trading_pair(self.quote_conversion_pair)
            if self._are_tokens_interchangeable(taker_quote, maker_quote):
                return Decimal('1')
            conversion_rate = self.rate_oracle.get_pair_rate(self.quote_conversion_pair)
            if conversion_rate is None:
                self.logger().error(f"Could not fetch conversion rate for {self.quote_conversion_pair}")
                raise ValueError(f"Could not fetch conversion rate for {self.quote_conversion_pair}")
            return conversion_rate
        except Exception as e:
            self.logger().error(f"Error fetching conversion rate for {self.quote_conversion_pair}: {e}")
            raise

    def to_format_status(self):
        try:
            # --- 确保数值安全 ---
            def safe_decimal(val, default=Decimal("0")):
                try:
                    if val is None:
                        return default
                    if isinstance(val, (int, float)):
                        return Decimal(str(val))
                    if isinstance(val, Decimal):
                        if val.is_nan() or not val.is_finite():
                            return default
                        return val
                    return default
                except (InvalidOperation, TypeError):
                    return default

            maker_side = getattr(self, "maker_order_side", "UNKNOWN")

            maker_connector = getattr(self, "maker_connector", "N/A")
            taker_connector = getattr(self, "taker_connector", "N/A")
            maker_pair = getattr(self, "maker_trading_pair", "UNKNOWN-PAIR")
            taker_pair = getattr(self, "taker_trading_pair", "UNKNOWN-PAIR")

            # --- 安全处理 Decimal 字段 ---
            min_profit = safe_decimal(getattr(self.config, "min_profitability", 0))
            target_profit = safe_decimal(getattr(self.config, "target_profitability", 0))
            max_profit = safe_decimal(getattr(self.config, "max_profitability", 0))

            trade_profit = safe_decimal(getattr(self, "_current_trade_profitability", 0))
            tx_cost_pct = safe_decimal(getattr(self, "_tx_cost_pct", 0))
            tx_cost = safe_decimal(getattr(self, "_tx_cost", 0))
            order_amount = safe_decimal(getattr(self.config, "order_amount", 0))
            taker_result_price = safe_decimal(getattr(self, "_taker_result_price", 0))

            # --- 计算当前盈利 ---
            current_profit = trade_profit - tx_cost_pct

            # --- 货币单位安全 ---
            try:
                quote = maker_pair.split('-')[-1]
            except Exception:
                quote = "UNKNOWN"

            # --- 格式化输出 ---
            return f"""
    Maker Side: {maker_side}
    -----------------------------------------------------------------------------------------------------------------------
        - Maker: {maker_connector} {maker_pair} | Taker: {taker_connector} {taker_pair}
        - Min profitability: {min_profit * 100:.2f}% | Target profitability: {target_profit * 100:.2f}% | Max profitability: {max_profit * 100:.2f}% | Current profitability: {current_profit * 100:.2f}%
        - Trade profitability: {trade_profit * 100:.2f}% | Tx cost: {tx_cost_pct * 100:.2f}%
        - Taker result price: {taker_result_price:.3f} | Tx cost: {tx_cost:.3f} {quote} | Order amount (Base): {order_amount:.2f}
    -----------------------------------------------------------------------------------------------------------------------
    """

        except Exception as e:
            self.logger().warning(f"[to_format_status] Failed to format status safely: {e}")
            return f"""
    Maker Side: UNKNOWN
    -----------------------------------------------------------------------------------------------------------------------
        Status formatting failed due to unexpected error: {e}
    -----------------------------------------------------------------------------------------------------------------------
    """
