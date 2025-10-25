"""
Modular Trading Bot - Supports multiple exchanges
"""

import os
import sys
import time
import asyncio
import traceback
from dataclasses import dataclass
from decimal import Decimal
from typing import Optional

from exchanges import ExchangeFactory
from helpers import TradingLogger
from helpers.lark_bot import LarkBot
from helpers.telegram_bot import TelegramBot


@dataclass
class TradingConfig:
    """Configuration class for trading parameters."""
    ticker: str
    contract_id: str
    quantity: Decimal
    take_profit: Decimal
    tick_size: Decimal
    direction: str
    max_orders: int
    wait_time: int
    exchange: str
    grid_step: Decimal
    stop_price: Decimal
    stop_loss_ratio: Decimal
    pause_price: Decimal
    boost_mode: bool

    @property
    def close_order_side(self) -> str:
        """Get the close order side based on bot direction."""
        return 'buy' if self.direction == "sell" else 'sell'


@dataclass
class OrderMonitor:
    """Thread-safe order monitoring state."""
    order_id: Optional[str] = None
    filled: bool = False
    filled_price: Optional[Decimal] = None
    filled_qty: Decimal = 0.0

    def reset(self):
        """Reset the monitor state."""
        self.order_id = None
        self.filled = False
        self.filled_price = None
        self.filled_qty = 0.0


class TradingBot:
    """Modular Trading Bot - Main trading logic supporting multiple exchanges."""

    def __init__(self, config: TradingConfig):
        self.config = config
        self.logger = TradingLogger(config.exchange, config.ticker, log_to_console=True)

        # Create exchange client
        try:
            self.exchange_client = ExchangeFactory.create_exchange(
                config.exchange,
                config
            )
        except ValueError as e:
            raise ValueError(f"Failed to create exchange client: {e}")

        # Trading state
        self.active_close_orders = []
        self.last_close_orders = 0
        self.last_open_order_time = 0
        self.last_log_time = 0
        self.current_order_status = None
        self.order_filled_event = asyncio.Event()
        self.order_canceled_event = asyncio.Event()
        self.shutdown_requested = False
        self.loop = None
        
        # Stop loss tracking
        self.position_avg_price = Decimal(0)
        self.position_size = Decimal(0)
        
        # Auto-rebalance tracking
        self._last_rebalance_time = 0

        # Register order callback
        self._setup_websocket_handlers()

    async def graceful_shutdown(self, reason: str = "Unknown", exit_code: int = 1):
        """Perform graceful shutdown of the trading bot."""
        self.logger.log(f"Starting graceful shutdown: {reason}", "INFO")
        self.shutdown_requested = True

        try:
            # Disconnect from exchange
            await self.exchange_client.disconnect()
            self.logger.log("Graceful shutdown completed", "INFO")

        except Exception as e:
            self.logger.log(f"Error during graceful shutdown: {e}", "ERROR")
        
        # Exit with appropriate exit code
        sys.exit(exit_code)

    def _setup_websocket_handlers(self):
        """Setup WebSocket handlers for order updates."""
        def order_update_handler(message):
            """Handle order updates from WebSocket."""
            try:
                # Check if this is for our contract
                if message.get('contract_id') != self.config.contract_id:
                    return

                order_id = message.get('order_id')
                status = message.get('status')
                side = message.get('side', '')
                order_type = message.get('order_type', '')
                filled_size = Decimal(message.get('filled_size'))
                if order_type == "OPEN":
                    self.current_order_status = status

                if status == 'FILLED':
                    if order_type == "OPEN":
                        self.order_filled_amount = filled_size
                        # Ensure thread-safe interaction with asyncio event loop
                        if self.loop is not None:
                            self.loop.call_soon_threadsafe(self.order_filled_event.set)
                        else:
                            # Fallback (should not happen after run() starts)
                            self.order_filled_event.set()

                    self.logger.log(f"[{order_type}] [{order_id}] {status} "
                                    f"{message.get('size')} @ {message.get('price')}", "INFO")
                    self.logger.log_transaction(order_id, side, message.get('size'), message.get('price'), status)
                elif status == "CANCELED":
                    if order_type == "OPEN":
                        self.order_filled_amount = filled_size
                        if self.loop is not None:
                            self.loop.call_soon_threadsafe(self.order_canceled_event.set)
                        else:
                            self.order_canceled_event.set()

                        if self.order_filled_amount > 0:
                            self.logger.log_transaction(order_id, side, self.order_filled_amount, message.get('price'), status)
                            
                    # PATCH
                    if self.config.exchange == "extended":
                        self.logger.log(f"[{order_type}] [{order_id}] {status} "
                                        f"{Decimal(message.get('size')) - filled_size} @ {message.get('price')}", "INFO")
                    else:
                        self.logger.log(f"[{order_type}] [{order_id}] {status} "
                                        f"{message.get('size')} @ {message.get('price')}", "INFO")
                elif status == "PARTIALLY_FILLED":
                    self.logger.log(f"[{order_type}] [{order_id}] {status} "
                                    f"{filled_size} @ {message.get('price')}", "INFO")
                else:
                    self.logger.log(f"[{order_type}] [{order_id}] {status} "
                                    f"{message.get('size')} @ {message.get('price')}", "INFO")

            except Exception as e:
                self.logger.log(f"Error handling order update: {e}", "ERROR")
                self.logger.log(f"Traceback: {traceback.format_exc()}", "ERROR")

        # Setup order update handler
        self.exchange_client.setup_order_update_handler(order_update_handler)

    def _calculate_wait_time(self) -> Decimal:
        """Calculate wait time between orders."""
        cool_down_time = self.config.wait_time

        if len(self.active_close_orders) < self.last_close_orders:
            self.last_close_orders = len(self.active_close_orders)
            return 0

        self.last_close_orders = len(self.active_close_orders)
        if len(self.active_close_orders) >= self.config.max_orders:
            return 1

        if len(self.active_close_orders) / self.config.max_orders >= 2/3:
            cool_down_time = 2 * self.config.wait_time
        elif len(self.active_close_orders) / self.config.max_orders >= 1/3:
            cool_down_time = self.config.wait_time
        elif len(self.active_close_orders) / self.config.max_orders >= 1/6:
            cool_down_time = self.config.wait_time / 2
        else:
            cool_down_time = self.config.wait_time / 4

        # if the program detects active_close_orders during startup, it is necessary to consider cooldown_time
        if self.last_open_order_time == 0 and len(self.active_close_orders) > 0:
            self.last_open_order_time = time.time()

        if time.time() - self.last_open_order_time > cool_down_time:
            return 0
        else:
            return 1

    async def _place_and_monitor_open_order(self) -> bool:
        """Place an order and monitor its execution with improved error handling."""
        try:
            # Reset state before placing order
            self.order_filled_event.clear()
            self.current_order_status = 'OPEN'
            self.order_filled_amount = 0.0

            # Place the order with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    order_result = await self.exchange_client.place_open_order(
                        self.config.contract_id,
                        self.config.quantity,
                        self.config.direction
                    )

                    if not order_result.success:
                        # Check if we should retry based on error type
                        error_msg = str(order_result.error_message).lower()
                        if any(keyword in error_msg for keyword in ['cannot connect', 'connection', 'network', 'timeout']):
                            if attempt < max_retries - 1:
                                wait_time = 2 ** attempt  # Exponential backoff
                                self.logger.log(f"Network error detected, waiting {wait_time}s before retry (attempt {attempt + 1}/{max_retries})", "WARNING")
                                await asyncio.sleep(wait_time)
                                continue
                        # For other errors, don't retry
                        self.logger.log(f"Order placement failed: {order_result.error_message}", "ERROR")
                        return False

                    # Order placement successful
                    if order_result.status == 'FILLED':
                        return await self._handle_order_result(order_result)
                    elif not self.order_filled_event.is_set():
                        try:
                            await asyncio.wait_for(self.order_filled_event.wait(), timeout=10)
                        except asyncio.TimeoutError:
                            pass

                    # Handle order result
                    return await self._handle_order_result(order_result)

                except Exception as e:
                    error_msg = str(e).lower()
                    if any(keyword in error_msg for keyword in ['cannot connect', 'connection', 'network', 'timeout']):
                        if attempt < max_retries - 1:
                            wait_time = 2 ** attempt
                            self.logger.log(f"Network exception detected, waiting {wait_time}s before retry (attempt {attempt + 1}/{max_retries})", "WARNING")
                            await asyncio.sleep(wait_time)
                            continue
                    # For other exceptions, don't retry
                    self.logger.log(f"Error placing order: {e}", "ERROR")
                    self.logger.log(f"Traceback: {traceback.format_exc()}", "ERROR")
                    return False

            # All retries failed
            self.logger.log(f"Failed to place order after {max_retries} attempts", "ERROR")
            return False

        except Exception as e:
            self.logger.log(f"Unexpected error in order placement: {e}", "ERROR")
            self.logger.log(f"Traceback: {traceback.format_exc()}", "ERROR")
            return False

    async def _handle_order_result(self, order_result) -> bool:
        """Handle the result of an order placement."""
        order_id = order_result.order_id
        filled_price = order_result.price

        if self.order_filled_event.is_set() or order_result.status == 'FILLED':
            if self.config.boost_mode:
                close_order_result = await self.exchange_client.place_market_order(
                    self.config.contract_id,
                    self.config.quantity,
                    self.config.close_order_side
                )
            else:
                self.last_open_order_time = time.time()
                # Place close order
                close_side = self.config.close_order_side
                if close_side == 'sell':
                    close_price = filled_price * (1 + self.config.take_profit/100)
                else:
                    close_price = filled_price * (1 - self.config.take_profit/100)

                close_order_result = await self.exchange_client.place_close_order(
                    self.config.contract_id,
                    self.config.quantity,
                    close_price,
                    close_side
                )
                if self.config.exchange == "lighter":
                    await asyncio.sleep(1)

                if not close_order_result.success:
                    self.logger.log(f"[CLOSE] Failed to place close order: {close_order_result.error_message}", "ERROR")
                    raise Exception(f"[CLOSE] Failed to place close order: {close_order_result.error_message}")

                return True

        else:
            new_order_price = await self.exchange_client.get_order_price(self.config.direction)

            def should_wait(direction: str, new_order_price: Decimal, order_result_price: Decimal) -> bool:
                if direction == "buy":
                    return new_order_price <= order_result_price
                elif direction == "sell":
                    return new_order_price >= order_result_price
                return False

            if self.config.exchange == "lighter":
                current_order_status = self.exchange_client.current_order.status
            else:
                order_info = await self.exchange_client.get_order_info(order_id)
                current_order_status = order_info.status

            while (
                should_wait(self.config.direction, new_order_price, order_result.price)
                and current_order_status == "OPEN"
            ):
                self.logger.log(f"[OPEN] [{order_id}] Waiting for order to be filled @ {order_result.price}", "INFO")
                await asyncio.sleep(5)
                if self.config.exchange == "lighter":
                    current_order_status = self.exchange_client.current_order.status
                else:
                    order_info = await self.exchange_client.get_order_info(order_id)
                    if order_info is not None:
                        current_order_status = order_info.status
                new_order_price = await self.exchange_client.get_order_price(self.config.direction)

            self.order_canceled_event.clear()
            # Cancel the order if it's still open
            self.logger.log(f"[OPEN] [{order_id}] Cancelling order and placing a new order", "INFO")
            if self.config.exchange == "lighter":
                cancel_result = await self.exchange_client.cancel_order(order_id)
                start_time = time.time()
                while (time.time() - start_time < 10 and self.exchange_client.current_order.status != 'CANCELED' and
                        self.exchange_client.current_order.status != 'FILLED'):
                    await asyncio.sleep(0.1)

                if self.exchange_client.current_order.status not in ['CANCELED', 'FILLED']:
                    raise Exception(f"[OPEN] Error cancelling order: {self.exchange_client.current_order.status}")
                else:
                    self.order_filled_amount = self.exchange_client.current_order.filled_size
            else:
                try:
                    cancel_result = await self.exchange_client.cancel_order(order_id)
                    if not cancel_result.success:
                        self.order_canceled_event.set()
                        self.logger.log(f"[CLOSE] Failed to cancel order {order_id}: {cancel_result.error_message}", "WARNING")
                    else:
                        self.current_order_status = "CANCELED"

                except Exception as e:
                    self.order_canceled_event.set()
                    self.logger.log(f"[CLOSE] Error canceling order {order_id}: {e}", "ERROR")

                if self.config.exchange == "backpack" or self.config.exchange == "extended":
                    self.order_filled_amount = cancel_result.filled_size
                else:
                    # Wait for cancel event or timeout
                    if not self.order_canceled_event.is_set():
                        try:
                            await asyncio.wait_for(self.order_canceled_event.wait(), timeout=5)
                        except asyncio.TimeoutError:
                            order_info = await self.exchange_client.get_order_info(order_id)
                            self.order_filled_amount = order_info.filled_size

            if self.order_filled_amount > 0:
                close_side = self.config.close_order_side
                if self.config.boost_mode:
                    close_order_result = await self.exchange_client.place_close_order(
                        self.config.contract_id,
                        self.order_filled_amount,
                        filled_price,
                        close_side
                    )
                else:
                    if close_side == 'sell':
                        close_price = filled_price * (1 + self.config.take_profit/100)
                    else:
                        close_price = filled_price * (1 - self.config.take_profit/100)

                    close_order_result = await self.exchange_client.place_close_order(
                        self.config.contract_id,
                        self.order_filled_amount,
                        close_price,
                        close_side
                    )
                    if self.config.exchange == "lighter":
                        await asyncio.sleep(1)

                self.last_open_order_time = time.time()
                if not close_order_result.success:
                    self.logger.log(f"[CLOSE] Failed to place close order: {close_order_result.error_message}", "ERROR")

            return True

        return False

    async def _log_status_periodically(self):
        """Log status information periodically, including positions."""
        if time.time() - self.last_log_time > 60 or self.last_log_time == 0:
            print("--------------------------------")
            try:
                # Get active orders
                active_orders = await self.exchange_client.get_active_orders(self.config.contract_id)

                # Filter close orders
                self.active_close_orders = []
                for order in active_orders:
                    if order.side == self.config.close_order_side:
                        self.active_close_orders.append({
                            'id': order.order_id,
                            'price': order.price,
                            'size': order.size
                        })

                # Get positions
                position_amt = await self.exchange_client.get_account_positions()

                # Calculate active closing amount
                active_close_amount = sum(
                    Decimal(order.get('size', 0))
                    for order in self.active_close_orders
                    if isinstance(order, dict)
                )

                self.logger.log(f"Current Position: {position_amt} | Active closing amount: {active_close_amount} | "
                                f"Order quantity: {len(self.active_close_orders)}")
                self.last_log_time = time.time()
                # Check for position mismatch
                if abs(position_amt - active_close_amount) > (2 * self.config.quantity):
                    error_message = f"\n\nERROR: [{self.config.exchange.upper()}_{self.config.ticker.upper()}] "
                    error_message += "Position mismatch detected\n"
                    error_message += "###### ERROR ###### ERROR ###### ERROR ###### ERROR #####\n"
                    error_message += "Please manually rebalance your position and take-profit orders\n"
                    error_message += "请手动平衡当前仓位和正在关闭的仓位\n"
                    error_message += f"current position: {position_amt} | active closing amount: {active_close_amount} | "f"Order quantity: {len(self.active_close_orders)}\n"
                    error_message += "###### ERROR ###### ERROR ###### ERROR ###### ERROR #####\n"
                    self.logger.log(error_message, "ERROR")

                    await self.send_notification(error_message.lstrip())

                    if not self.shutdown_requested:
                        self.shutdown_requested = True

                    mismatch_detected = True
                else:
                    mismatch_detected = False

                return mismatch_detected

            except Exception as e:
                self.logger.log(f"Error in periodic status check: {e}", "ERROR")
                self.logger.log(f"Traceback: {traceback.format_exc()}", "ERROR")

            print("--------------------------------")

    async def _meet_grid_step_condition(self) -> bool:
        if self.active_close_orders:
            picker = min if self.config.direction == "buy" else max
            next_close_order = picker(self.active_close_orders, key=lambda o: o["price"])
            next_close_price = next_close_order["price"]

            best_bid, best_ask = await self.exchange_client.fetch_bbo_prices(self.config.contract_id)
            if best_bid <= 0 or best_ask <= 0 or best_bid >= best_ask:
                raise ValueError("No bid/ask data available")

            if self.config.direction == "buy":
                new_order_close_price = best_ask * (1 + self.config.take_profit/100)
                if next_close_price / new_order_close_price > 1 + self.config.grid_step/100:
                    return True
                else:
                    return False
            elif self.config.direction == "sell":
                new_order_close_price = best_bid * (1 - self.config.take_profit/100)
                if new_order_close_price / next_close_price > 1 + self.config.grid_step/100:
                    return True
                else:
                    return False
            else:
                raise ValueError(f"Invalid direction: {self.config.direction}")
        else:
            return True

    async def _update_position_tracking(self, position_amt: Decimal):
        """Update position tracking for stop loss calculation."""
        if position_amt != self.position_size:
            # Position changed, update average price
            if position_amt == 0:
                # Position closed, reset tracking
                self.position_avg_price = Decimal(0)
                self.position_size = Decimal(0)
            else:
                # Get current position info to update average price
                try:
                    # For simplicity, we'll use the current market price as the new entry price
                    # In a more sophisticated implementation, you would track each fill separately
                    best_bid, best_ask = await self.exchange_client.fetch_bbo_prices(self.config.contract_id)
                    current_price = (best_bid + best_ask) / 2
                    
                    if self.position_size == 0:
                        # New position
                        self.position_avg_price = current_price
                        self.position_size = position_amt
                    else:
                        # Position changed, recalculate average price
                        # This is a simplified calculation - in practice you'd track each fill
                        self.position_avg_price = current_price
                        self.position_size = position_amt
                        
                except Exception as e:
                    self.logger.log(f"Error updating position tracking: {e}", "WARNING")

    async def _execute_stop_loss(self) -> bool:
        """Execute stop loss by closing the position."""
        try:
            self.logger.log(f"Executing stop loss: Closing position of size {self.position_size}", "WARNING")
            
            # Determine the side to close the position
            # For long positions, we need to sell to close
            # For short positions, we need to buy to close
            close_side = 'sell' if self.config.direction == 'buy' else 'buy'
            
            # Use market order to ensure quick execution
            close_result = await self.exchange_client.place_market_order(
                self.config.contract_id,
                abs(self.position_size),  # Use absolute value for size
                close_side
            )
            
            if close_result.success:
                self.logger.log(f"Stop loss executed successfully: Closed {self.position_size} at market price", "WARNING")
                return True
            else:
                self.logger.log(f"Failed to execute stop loss: {close_result.error_message}", "ERROR")
                return False
                
        except Exception as e:
            self.logger.log(f"Error executing stop loss: {e}", "ERROR")
            self.logger.log(f"Traceback: {traceback.format_exc()}", "ERROR")
            return False

    async def _check_stop_loss_condition(self) -> bool:
        """Check if stop loss condition is triggered."""
        if self.config.stop_loss_ratio == -1 or self.position_size == 0:
            return False

        try:
            best_bid, best_ask = await self.exchange_client.fetch_bbo_prices(self.config.contract_id)
            if best_bid <= 0 or best_ask <= 0 or best_bid >= best_ask:
                return False

            if self.config.direction == "buy":
                # For long positions, stop loss triggers when price drops below threshold
                stop_loss_price = self.position_avg_price * (1 - self.config.stop_loss_ratio / 100)
                price_diff = best_bid - stop_loss_price
                price_diff_percent = (price_diff / self.position_avg_price) * 100
                
                # Only log every 5th check to reduce frequency (approximately every 5 minutes)
                current_time = time.time()
                if not hasattr(self, '_last_stop_loss_log_time'):
                    self._last_stop_loss_log_time = 0
                
                if current_time - self._last_stop_loss_log_time > 300:  # 5 minutes
                    self.logger.log(f"Stop Loss Check - LONG Position:", "INFO")
                    self.logger.log(f"  Current Bid Price: {best_bid}", "INFO")
                    self.logger.log(f"  Stop Loss Price: {stop_loss_price}", "INFO")
                    self.logger.log(f"  Average Entry Price: {self.position_avg_price}", "INFO")
                    self.logger.log(f"  Stop Loss Ratio: {self.config.stop_loss_ratio}%", "INFO")
                    self.logger.log(f"  Distance to Stop Loss: {price_diff:.2f} ({price_diff_percent:.2f}%)", "INFO")
                    self.logger.log(f"  Position Size: {self.position_size}", "INFO")
                    self.logger.log(f"  Status: {'ABOVE STOP LOSS' if price_diff > 0 else 'BELOW STOP LOSS'}", "INFO")
                    self._last_stop_loss_log_time = current_time
                
                if best_bid <= stop_loss_price:
                    self.logger.log(f"STOP LOSS TRIGGERED - LONG POSITION:", "WARNING")
                    self.logger.log(f"  Current Bid Price: {best_bid}", "WARNING")
                    self.logger.log(f"  Stop Loss Price: {stop_loss_price}", "WARNING")
                    self.logger.log(f"  Average Entry Price: {self.position_avg_price}", "WARNING")
                    self.logger.log(f"  Stop Loss Ratio: {self.config.stop_loss_ratio}%", "WARNING")
                    self.logger.log(f"  Position Size: {self.position_size}", "WARNING")
                    self.logger.log(f"  Loss Percentage: {((self.position_avg_price - best_bid) / self.position_avg_price * 100):.2f}%", "WARNING")
                    return True
                    
            elif self.config.direction == "sell":
                # For short positions, stop loss triggers when price rises above threshold
                stop_loss_price = self.position_avg_price * (1 + self.config.stop_loss_ratio / 100)
                price_diff = stop_loss_price - best_ask
                price_diff_percent = (price_diff / self.position_avg_price) * 100
                
                # Only log every 5th check to reduce frequency (approximately every 5 minutes)
                current_time = time.time()
                if not hasattr(self, '_last_stop_loss_log_time'):
                    self._last_stop_loss_log_time = 0
                
                if current_time - self._last_stop_loss_log_time > 300:  # 5 minutes
                    self.logger.log(f"Stop Loss Check - SHORT Position:", "INFO")
                    self.logger.log(f"  Current Ask Price: {best_ask}", "INFO")
                    self.logger.log(f"  Stop Loss Price: {stop_loss_price}", "INFO")
                    self.logger.log(f"  Average Entry Price: {self.position_avg_price}", "INFO")
                    self.logger.log(f"  Stop Loss Ratio: {self.config.stop_loss_ratio}%", "INFO")
                    self.logger.log(f"  Distance to Stop Loss: {price_diff:.2f} ({price_diff_percent:.2f}%)", "INFO")
                    self.logger.log(f"  Position Size: {self.position_size}", "INFO")
                    self.logger.log(f"  Status: {'BELOW STOP LOSS' if price_diff > 0 else 'ABOVE STOP LOSS'}", "INFO")
                    self._last_stop_loss_log_time = current_time
                
                if best_ask >= stop_loss_price:
                    self.logger.log(f"STOP LOSS TRIGGERED - SHORT POSITION:", "WARNING")
                    self.logger.log(f"  Current Ask Price: {best_ask}", "WARNING")
                    self.logger.log(f"  Stop Loss Price: {stop_loss_price}", "WARNING")
                    self.logger.log(f"  Average Entry Price: {self.position_avg_price}", "WARNING")
                    self.logger.log(f"  Stop Loss Ratio: {self.config.stop_loss_ratio}%", "WARNING")
                    self.logger.log(f"  Position Size: {self.position_size}", "WARNING")
                    self.logger.log(f"  Loss Percentage: {((best_ask - self.position_avg_price) / self.position_avg_price * 100):.2f}%", "WARNING")
                    return True

        except Exception as e:
            self.logger.log(f"Error checking stop loss condition: {e}", "ERROR")
            
        return False

    async def _check_price_condition(self) -> bool:
        stop_trading = False
        pause_trading = False

        if self.config.pause_price == self.config.stop_price == -1:
            return stop_trading, pause_trading

        best_bid, best_ask = await self.exchange_client.fetch_bbo_prices(self.config.contract_id)
        if best_bid <= 0 or best_ask <= 0 or best_bid >= best_ask:
            raise ValueError("No bid/ask data available")

        if self.config.stop_price != -1:
            if self.config.direction == "buy":
                if best_ask >= self.config.stop_price:
                    stop_trading = True
            elif self.config.direction == "sell":
                if best_bid <= self.config.stop_price:
                    stop_trading = True

        if self.config.pause_price != -1:
            if self.config.direction == "buy":
                if best_ask >= self.config.pause_price:
                    pause_trading = True
            elif self.config.direction == "sell":
                if best_bid <= self.config.pause_price:
                    pause_trading = True

        return stop_trading, pause_trading

    async def send_notification(self, message: str):
        lark_token = os.getenv("LARK_TOKEN")
        if lark_token:
            async with LarkBot(lark_token) as lark_bot:
                await lark_bot.send_text(message)

        telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
        telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID")
        if telegram_token and telegram_chat_id:
            with TelegramBot(telegram_token, telegram_chat_id) as tg_bot:
                tg_bot.send_text(message)

    async def _retry_network_connection(self, max_retries: int = 3) -> bool:
        """Try to re-establish network connection with exponential backoff."""
        for attempt in range(max_retries):
            try:
                # Simple network check - try to fetch prices
                await self.exchange_client.fetch_bbo_prices(self.config.contract_id)
                self.logger.log(f"Network connection recovered after {attempt + 1} attempts", "INFO")
                return True
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 5 * (attempt + 1)  # 5s, 10s, 15s
                    self.logger.log(f"Network still unavailable, waiting {wait_time}s before retry...", "WARNING")
                    await asyncio.sleep(wait_time)
                else:
                    self.logger.log(f"Network connection failed after {max_retries} attempts: {e}", "ERROR")
        return False

    async def _auto_rebalance_position(self):
        """Automatically rebalance position and orders when mismatch detected."""
        try:
            # Avoid frequent rebalancing (minimum 5 minutes between rebalances)
            current_time = time.time()
            if current_time - self._last_rebalance_time < 300:  # 5 minutes
                return False
                
            # Get real-time data
            position_amt = await self.exchange_client.get_account_positions()
            active_orders = await self.exchange_client.get_active_orders(self.config.contract_id)
            
            # Calculate active close orders amount
            active_close_amount = sum(
                Decimal(order.size) 
                for order in active_orders 
                if order.side == self.config.close_order_side
            )
            
            # Calculate mismatch amount
            mismatch_amount = position_amt - active_close_amount
            
            # Check if rebalancing is needed (10% tolerance)
            tolerance = self.config.quantity * Decimal('0.1')
            if abs(mismatch_amount) <= tolerance:
                return False
                
            # Limit rebalance amount (max 3x single trade quantity)
            max_rebalance_amount = self.config.quantity * 3
            if abs(mismatch_amount) > max_rebalance_amount:
                self.logger.log(f"Mismatch too large for auto-rebalance: {mismatch_amount}", "WARNING")
                return False
            
            self.logger.log(f"Auto-rebalancing: Position={position_amt}, Close orders={active_close_amount}, Mismatch={mismatch_amount}", "WARNING")
            
            if mismatch_amount > 0:
                # Position > Close orders, need to add close orders
                success = await self._add_close_orders(mismatch_amount)
            else:
                # Position < Close orders, need to cancel excess orders
                success = await self._cancel_excess_orders(abs(mismatch_amount))
            
            if success:
                self._last_rebalance_time = current_time
                self.logger.log("Auto-rebalance completed successfully", "INFO")
                return True
            else:
                self.logger.log("Auto-rebalance failed", "ERROR")
                return False
                
        except Exception as e:
            self.logger.log(f"Auto-rebalance error: {e}", "ERROR")
            return False

    async def _add_close_orders(self, amount_to_add: Decimal):
        """Add close orders to match position."""
        try:
            self.logger.log(f"Adding close orders for {amount_to_add} to rebalance position", "INFO")
            
            # Get current market prices
            best_bid, best_ask = await self.exchange_client.fetch_bbo_prices(self.config.contract_id)
            
            # Calculate close price based on strategy
            if self.config.direction == "buy":
                # Long strategy: close price = ask price * (1 + take_profit%)
                close_price = best_ask * (1 + self.config.take_profit/100)
            else:
                # Short strategy: close price = bid price * (1 - take_profit%)
                close_price = best_bid * (1 - self.config.take_profit/100)
            
            # Place close order
            close_side = self.config.close_order_side
            order_result = await self.exchange_client.place_close_order(
                self.config.contract_id,
                amount_to_add,
                close_price,
                close_side
            )
            
            if order_result.success:
                self.logger.log(f"Successfully added close order: {amount_to_add} @ {close_price}", "INFO")
                return True
            else:
                self.logger.log(f"Failed to add close order: {order_result.error_message}", "ERROR")
                return False
                
        except Exception as e:
            self.logger.log(f"Error adding close orders: {e}", "ERROR")
            return False

    async def _cancel_excess_orders(self, amount_to_cancel: Decimal):
        """Cancel excess close orders to match position."""
        try:
            self.logger.log(f"Cancelling excess orders for {amount_to_cancel} to rebalance position", "INFO")
            
            # Get active close orders
            active_orders = await self.exchange_client.get_active_orders(self.config.contract_id)
            close_orders = [order for order in active_orders if order.side == self.config.close_order_side]
            
            # Sort orders by price (for long strategy cancel highest prices, for short strategy cancel lowest prices)
            if self.config.direction == "buy":
                close_orders.sort(key=lambda o: o.price, reverse=True)  # Highest prices first
            else:
                close_orders.sort(key=lambda o: o.price)  # Lowest prices first
            
            # Cancel orders until target amount is reached
            cancelled_amount = Decimal(0)
            for order in close_orders:
                if cancelled_amount >= amount_to_cancel:
                    break
                    
                # Cancel order
                cancel_result = await self.exchange_client.cancel_order(order.order_id)
                if cancel_result.success:
                    cancelled_amount += order.size
                    self.logger.log(f"Cancelled order {order.order_id}: {order.size} @ {order.price}", "INFO")
                else:
                    self.logger.log(f"Failed to cancel order {order.order_id}: {cancel_result.error_message}", "WARNING")
            
            self.logger.log(f"Cancelled {cancelled_amount} in excess orders", "INFO")
            return cancelled_amount >= amount_to_cancel
            
        except Exception as e:
            self.logger.log(f"Error cancelling excess orders: {e}", "ERROR")
            return False

    async def _restore_trading_state(self):
        """Restore trading state after program restart with network retry."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.logger.log("Restoring trading state after restart...", "INFO")
                
                # Get current position
                position_amt = await self.exchange_client.get_account_positions()
                self.position_size = position_amt
                
                # Get active orders
                active_orders = await self.exchange_client.get_active_orders(self.config.contract_id)
                
                # Rebuild close orders list
                self.active_close_orders = []
                for order in active_orders:
                    if order.side == self.config.close_order_side:
                        self.active_close_orders.append({
                            'id': order.order_id,
                            'price': order.price,
                            'size': order.size
                        })
                
                # Rebuild last_close_orders count
                self.last_close_orders = len(self.active_close_orders)
                
                # Rebuild position tracking (simplified - use current market price as average)
                if position_amt != 0:
                    try:
                        best_bid, best_ask = await self.exchange_client.fetch_bbo_prices(self.config.contract_id)
                        self.position_avg_price = (best_bid + best_ask) / 2
                    except Exception as e:
                        self.logger.log(f"Failed to get market price for position tracking: {e}", "WARNING")
                        # Use a safe default
                        self.position_avg_price = Decimal(0)
                
                self.logger.log(f"State restored: Position={position_amt}, Close orders={len(self.active_close_orders)}", "INFO")
                return
                
            except ConnectionError as e:
                if attempt < max_retries - 1:
                    wait_time = 5 * (attempt + 1)  # 5s, 10s, 15s
                    self.logger.log(f"Network error during state restore (attempt {attempt + 1}/{max_retries}), waiting {wait_time}s...", "WARNING")
                    await asyncio.sleep(wait_time)
                else:
                    self.logger.log(f"Failed to restore trading state after {max_retries} attempts: {e}", "ERROR")
                    raise
            except Exception as e:
                # For non-network errors, don't retry
                self.logger.log(f"Failed to restore trading state: {e}", "ERROR")
                raise

    async def run(self):
        """Main trading loop."""
        try:
            self.config.contract_id, self.config.tick_size = await self.exchange_client.get_contract_attributes()

            # Log current TradingConfig
            self.logger.log("=== Trading Configuration ===", "INFO")
            self.logger.log(f"Ticker: {self.config.ticker}", "INFO")
            self.logger.log(f"Contract ID: {self.config.contract_id}", "INFO")
            self.logger.log(f"Quantity: {self.config.quantity}", "INFO")
            self.logger.log(f"Take Profit: {self.config.take_profit}%", "INFO")
            self.logger.log(f"Direction: {self.config.direction}", "INFO")
            self.logger.log(f"Max Orders: {self.config.max_orders}", "INFO")
            self.logger.log(f"Wait Time: {self.config.wait_time}s", "INFO")
            self.logger.log(f"Exchange: {self.config.exchange}", "INFO")
            self.logger.log(f"Grid Step: {self.config.grid_step}%", "INFO")
            self.logger.log(f"Stop Price: {self.config.stop_price}", "INFO")
            self.logger.log(f"Pause Price: {self.config.pause_price}", "INFO")
            self.logger.log(f"Boost Mode: {self.config.boost_mode}", "INFO")
            self.logger.log("=============================", "INFO")

            # Capture the running event loop for thread-safe callbacks
            self.loop = asyncio.get_running_loop()
            # Connect to exchange
            await self.exchange_client.connect()

            # wait for connection to establish
            await asyncio.sleep(5)

            # Restore trading state after restart
            await self._restore_trading_state()

            # Main trading loop
            while not self.shutdown_requested:
                # Update active orders
                active_orders = await self.exchange_client.get_active_orders(self.config.contract_id)

                # Filter close orders
                self.active_close_orders = []
                for order in active_orders:
                    if order.side == self.config.close_order_side:
                        self.active_close_orders.append({
                            'id': order.order_id,
                            'price': order.price,
                            'size': order.size
                        })

                # Update position tracking for stop loss
                position_amt = await self.exchange_client.get_account_positions()
                await self._update_position_tracking(position_amt)

                # Try auto-rebalance first
                rebalanced = await self._auto_rebalance_position()
                
                # Periodic logging
                mismatch_detected = await self._log_status_periodically()

                # Check stop loss condition (only log every 5 minutes to reduce frequency)
                current_time = time.time()
                if not hasattr(self, '_last_stop_loss_check_time'):
                    self._last_stop_loss_check_time = 0
                
                # Only check stop loss condition every 30 seconds to reduce frequency
                if current_time - self._last_stop_loss_check_time > 30:
                    self._last_stop_loss_check_time = current_time
                    stop_loss_triggered = await self._check_stop_loss_condition()
                else:
                    stop_loss_triggered = False
                    
                if stop_loss_triggered:
                    msg = f"\n\nWARNING: [{self.config.exchange.upper()}_{self.config.ticker.upper()}] \n"
                    msg += f"Stop loss triggered at {self.config.stop_loss_ratio}% loss\n"
                    msg += f"止损触发，亏损达到{self.config.stop_loss_ratio}%\n"
                    msg += f"Position size: {self.position_size}, Average price: {self.position_avg_price}\n"
                    msg += "Executing stop loss by closing position...\n"
                    msg += "正在执行止损平仓...\n"
                    await self.send_notification(msg.lstrip())
                    
                    # Execute stop loss by closing the position
                    stop_loss_executed = await self._execute_stop_loss()
                    if stop_loss_executed:
                        msg = f"Stop loss executed successfully. Position closed.\n"
                        msg += f"止损执行成功，仓位已平仓。\n"
                    else:
                        msg = f"Failed to execute stop loss. Manual intervention required.\n"
                        msg += f"止损执行失败，需要手动干预。\n"
                    
                    await self.send_notification(msg)
                    # Exit with code 0 to prevent auto-restart
                    await self.graceful_shutdown("Stop loss triggered and executed", exit_code=0)
                    continue

                stop_trading, pause_trading = await self._check_price_condition()
                if stop_trading:
                    msg = f"\n\nWARNING: [{self.config.exchange.upper()}_{self.config.ticker.upper()}] \n"
                    msg += "Stopped trading due to stop price triggered\n"
                    msg += "价格已经达到停止交易价格，脚本将停止交易\n"
                    await self.send_notification(msg.lstrip())
                    # Exit with code 0 to prevent auto-restart
                    await self.graceful_shutdown(msg, exit_code=0)
                    continue

                if pause_trading:
                    await asyncio.sleep(5)
                    continue

                if not mismatch_detected:
                    wait_time = self._calculate_wait_time()

                    if wait_time > 0:
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        meet_grid_step_condition = await self._meet_grid_step_condition()
                        if not meet_grid_step_condition:
                            await asyncio.sleep(1)
                            continue

                        await self._place_and_monitor_open_order()
                        self.last_close_orders += 1

        except KeyboardInterrupt:
            self.logger.log("Bot stopped by user")
            await self.graceful_shutdown("User interruption (Ctrl+C)")
        except ConnectionError as e:
            # Network error: try to reconnect before shutting down
            self.logger.log(f"Network connection error: {e}", "WARNING")
            self.logger.log("Attempting to re-establish network connection...", "WARNING")
            
            if await self._retry_network_connection():
                self.logger.log("Network connection recovered, continuing trading", "INFO")
                # Continue running instead of shutting down
                return await self.run()
            else:
                self.logger.log("Failed to re-establish network connection after retries", "ERROR")
                await self.graceful_shutdown(f"Network error: {e}", exit_code=1)
        except Exception as e:
            self.logger.log(f"Critical error: {e}", "ERROR")
            self.logger.log(f"Traceback: {traceback.format_exc()}", "ERROR")
            await self.graceful_shutdown(f"Critical error: {e}")
            raise
        finally:
            # Ensure all connections are closed even if graceful shutdown fails
            try:
                await self.exchange_client.disconnect()
            except Exception as e:
                self.logger.log(f"Error disconnecting from exchange: {e}", "ERROR")
