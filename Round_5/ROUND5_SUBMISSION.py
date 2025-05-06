import json
import numpy as np
import math
from statistics import NormalDist
from datamodel import *
from typing import Any
import math
from math import log, sqrt, exp
INF = 1e9
class Status:

    _position_limit = {
        "RAINFOREST_RESIN": 50,
        "KELP": 50,
        "SQUID_INK": 50,
        "PICNIC_BASKET1": 60,
        "PICNIC_BASKET2": 100,
        "JAMS": 350,
        "CROISSANTS": 250,
        "DJEMBES": 60,
        "VOLCANIC_ROCK": 400,
        "VOLCANIC_ROCK_VOUCHER_9500": 200,
        "VOLCANIC_ROCK_VOUCHER_9750": 200,
        "VOLCANIC_ROCK_VOUCHER_10000": 200,
        "VOLCANIC_ROCK_VOUCHER_10250": 200,
        "VOLCANIC_ROCK_VOUCHER_10500": 200,
        "MAGNIFICENT_MACARONS": 75
    }
    _state = None
    _realtime_position = {key:0 for key in _position_limit.keys()}
    _hist_order_depths = {
        product:{
            'bidprc1': [],
            'bidamt1': [],
            'bidprc2': [],
            'bidamt2': [],
            'bidprc3': [],
            'bidamt3': [],
            'askprc1': [],
            'askamt1': [],
            'askprc2': [],
            'askamt2': [],
            'askprc3': [],
            'askamt3': [],
        } for product in _position_limit.keys()
    }
    _hist_observation = {
        'sunlight': [],
        'humidity': [],
        'transportFees': [],
        'exportTariff': [],
        'importTariff': [],
        'bidPrice': [],
        'askPrice': [],        
    }    
    _num_data = 0
    def __init__(self, product: str) -> None:
        self.product = product
        self.cdf_buy_at = None
        self.cdf_sell_at = None    
        self.ema = None
        self.deviation = 0.0
        self.deviation_zscore_ema = 0.0
    @classmethod
    def cls_update(cls, state: TradingState) -> None:
        # Update TradingState
        cls._state = state
        # Update realtime position
        for product, posit in state.position.items():
            cls._realtime_position[product] = posit
        # Update historical order_depths
        for product, orderdepth in state.order_depths.items():
            cnt = 1
            for prc, amt in sorted(orderdepth.sell_orders.items(), reverse=False):
                cls._hist_order_depths[product][f'askamt{cnt}'].append(amt)
                cls._hist_order_depths[product][f'askprc{cnt}'].append(prc)
                cnt += 1
                if cnt == 4:
                    break
            while cnt < 4:
                cls._hist_order_depths[product][f'askprc{cnt}'].append(np.nan)
                cls._hist_order_depths[product][f'askamt{cnt}'].append(np.nan)
                cnt += 1
            cnt = 1
            for prc, amt in sorted(orderdepth.buy_orders.items(), reverse=True):
                cls._hist_order_depths[product][f'bidprc{cnt}'].append(prc)
                cls._hist_order_depths[product][f'bidamt{cnt}'].append(amt)
                cnt += 1
                if cnt == 4:
                    break
            while cnt < 4:
                cls._hist_order_depths[product][f'bidprc{cnt}'].append(np.nan)
                cls._hist_order_depths[product][f'bidamt{cnt}'].append(np.nan)
                cnt += 1
        cls._num_data += 1
    def hist_order_depth(self, type: str, depth: int, size) -> np.ndarray:
        return np.array(self._hist_order_depths[self.product][f'{type}{depth}'][-size:], dtype=np.float32)
    @property
    def timestep(self) -> int:
        return self._state.timestamp / 100
    @property
    def position_limit(self) -> int:
        return self._position_limit[self.product]
    @property
    def position(self) -> int:
        if self.product in self._state.position:
            return int(self._state.position[self.product])
        else:
            return 0
    @property
    def rt_position(self) -> int:
        return self._realtime_position[self.product]
    def _cls_rt_position_update(cls, product, new_position):
        if abs(new_position) <= cls._position_limit[product]:
            cls._realtime_position[product] = new_position
        else:
            raise ValueError("New position exceeds position limit")
    def rt_position_update(self, new_position: int) -> None:
        self._cls_rt_position_update(self.product, new_position)
    @property
    def bids(self) -> list[tuple[int, int]]:
        return list(self._state.order_depths[self.product].buy_orders.items())
    @property
    def asks(self) -> list[tuple[int, int]]:
        return list(self._state.order_depths[self.product].sell_orders.items())
    @property
    def second_bid(self) -> None:
        bids_list = sorted(self.bids, key=lambda x: x[0], reverse=True)
        if len(bids_list) > 1:
            return bids_list[1][0]
        else:
            return None
    @property
    def second_ask(self) -> None:
        asks_list = sorted(self.asks, key=lambda x: x[0], reverse=False)
        if len(asks_list) > 1:
            return asks_list[1][0]
        else:
            return None
    @classmethod
    def _cls_update_bids(cls, product, prc, new_amt):
        if new_amt > 0:
            cls._state.order_depths[product].buy_orders[prc] = new_amt
        elif new_amt == 0:
            cls._state.order_depths[product].buy_orders[prc] = 0
    @classmethod
    def _cls_update_asks(cls, product, prc, new_amt):
        if new_amt < 0:
            cls._state.order_depths[product].sell_orders[prc] = new_amt
        elif new_amt == 0:
            cls._state.order_depths[product].sell_orders[prc] = 0
    def update_bids(self, prc: int, new_amt: int) -> None:
        self._cls_update_bids(self.product, prc, new_amt)
    def update_asks(self, prc: int, new_amt: int) -> None:
        self._cls_update_asks(self.product, prc, new_amt)
    @property
    def possible_buy_amt(self) -> int:
        possible_buy_amount1 = self._position_limit[self.product] - self.rt_position
        possible_buy_amount2 = self._position_limit[self.product] - self.position
        return min(possible_buy_amount1, possible_buy_amount2)
    @property
    def possible_sell_amt(self) -> int:
        possible_sell_amount1 = self._position_limit[self.product] + self.rt_position
        possible_sell_amount2 = self._position_limit[self.product] + self.position
        return min(abs(possible_sell_amount1), abs(possible_sell_amount2))
    @property
    def log_return(self):
        prev_mid = self.last_mid_price
        curr_mid = self.mid

        if prev_mid is not None and prev_mid > 0 and curr_mid > 0:
            return 100 * math.log(curr_mid / prev_mid)
        else:
            return 0.0
    @property
    def last_mid_price(self) -> int | None:
        try:
            trader_object = jsonpickle.decode(self._state.traderData)
            return trader_object.get(f"{self.product}_last_mid", None)
        except:
            return None
    @property
    def import_ema(self):
        try:
            trader_object = jsonpickle.decode(self._state.traderData)
            return trader_object.get(f"{self.product}_ema", None)
        except:
            return None
    @property  
    def import_deviation(self):
        try:
            trader_object = jsonpickle.decode(self._state.traderData)
            return trader_object.get(f"{self.product}_deviation", None)
        except:
            return None
   ## SQUID STUFF     
    def precompute_distributions(self):
        offset_sell = 10
        offset_sell_b = 5
        offset_buy_b = 5
        offset_buy = 10
        
        alphaL_buy, alphaR_buy = 1.0, 4.0
        buy_a =  self.ema - offset_buy 
        buy_b = self.ema - offset_buy - offset_buy_b  # a > b

        _, _, _, cdf_at_buy = self.build_mirrored_beta_distribution(
            buy_a, buy_b, alphaL_buy, alphaR_buy, steps=100
        )
        self.cdf_buy_at = cdf_at_buy

        alphaL_sell, alphaR_sell = 1.0, 4.0
        sell_a =  self.ema + offset_sell
        sell_b =  self.ema + offset_sell + offset_sell_b

        _, _, _, cdf_at_sell = self.build_ascending_beta_distribution(
            sell_a, sell_b, alphaL_sell, alphaR_sell, steps=100
        )
        self.cdf_sell_at = cdf_at_sell   

    def build_mirrored_beta_distribution(self, a, b, alphaL, alphaR, steps=100):
        if not (a > b):
            raise ValueError("For mirrored distribution, expect a > b.")

        x_vals_desc = np.linspace(a, b, steps)

        def x_to_u(x):
            return (a - x)/(a - b)

        pdf_u_desc = np.array([self.beta_pdf(x_to_u(xx), alphaL, alphaR) for xx in x_vals_desc])
        pdf_x_desc = pdf_u_desc / abs(b - a)

        cdf_desc = np.zeros_like(pdf_x_desc)
        for i in range(1, steps):
            dx = x_vals_desc[i] - x_vals_desc[i - 1]
            area = 0.5*(pdf_x_desc[i] + pdf_x_desc[i - 1])*abs(dx)
            cdf_desc[i] = cdf_desc[i - 1] + area

        def cdf_at(x):
            if x >= a:
                return 0.0
            if x <= b:
                return 1.0
            x_asc = x_vals_desc[::-1]
            cdf_asc = cdf_desc[::-1]
            return np.interp(x, x_asc, cdf_asc)
        return x_vals_desc, pdf_x_desc, cdf_desc, cdf_at
    def beta_pdf(self, u, alpha, beta):
        if u < 0.0 or u > 1.0:
            return 0.0
        from math import gamma
        beta_func = gamma(alpha)*gamma(beta)/gamma(alpha+beta)
        return (u**(alpha-1) * (1.0 - u)**(beta-1)) / beta_func
    
    def build_ascending_beta_distribution(self, a, b, alphaL, alphaR, steps=100):
        if not (a < b):
            raise ValueError("For ascending distribution, expect a < b.")

        x_vals = np.linspace(a, b, steps)

        def x_to_u(x):
            return (x - a)/(b - a)

        pdf_u = np.array([self.beta_pdf(x_to_u(xx), alphaL, alphaR) for xx in x_vals])
        pdf_x = pdf_u / (b - a)

        cdf_x = np.zeros_like(pdf_x)
        for i in range(1, steps):
            dx = x_vals[i] - x_vals[i-1]
            area = 0.5*(pdf_x[i] + pdf_x[i-1])*dx
            cdf_x[i] = cdf_x[i-1] + area

        def cdf_at(x):
            if x <= a:
                return 0.0
            if x >= b:
                return 1.0
            return np.interp(x, x_vals, cdf_x)

        return x_vals, pdf_x, cdf_x, cdf_at
    
    @property
    def standardized_log_return(self):
        return (self.log_return + 0.00012)/0.0385
    
    def set_log_return_and_mid(self, trader_object: dict) -> str:
        try:
            trader_object[f"{self.product}_log_return"] = self.log_return
            trader_object[f"{self.product}_last_mid"] = self.mid
            return trader_object
        except:
            return trader_object  # fallback

    def refresh_ema_deviation(self, trader_memory: dict) -> dict:
        # ----- 1) Ensure we have memory keys -----
        price_ema_key       = f"{self.product}_ema"          # float
        dev_sq_ema_key      = f"{self.product}_devsq_ema"    # float
        dev_sq_diff_listkey = f"{self.product}_devsq_diff"   # rolling list of dev^2 - dev_sq_ema
        devz_ema_key        = f"{self.product}_devsq_z_ema"  # final z-score EMA

        # Make sure the rolling list is initialized
        if dev_sq_diff_listkey not in trader_memory:
            trader_memory[dev_sq_diff_listkey] = []

        # ----- 2) Price EMA update (alpha=0.004) -----
        alpha_ema = 0.005
        old_price_ema = trader_memory.get(price_ema_key, None)
        curr_mid = self.volume_weighted_mid_price

        if old_price_ema is None:
            new_price_ema = curr_mid
        else:
            new_price_ema = (1 - alpha_ema) * old_price_ema + alpha_ema * curr_mid

        self.ema = new_price_ema
        trader_memory[price_ema_key] = new_price_ema

        # ----- 3) deviation = mid - ema -----
        new_dev = curr_mid - new_price_ema
        self.deviation = new_dev

        # ----- 4) dev^2 EMA (alpha=0.004) -----
        dev_sq = new_dev * new_dev
        old_dev_sq_ema = trader_memory.get(dev_sq_ema_key, None)

        if old_dev_sq_ema is None:
            new_dev_sq_ema = dev_sq
        else:
            new_dev_sq_ema = (1 - alpha_ema) * old_dev_sq_ema + alpha_ema * dev_sq

        # Store it in object + memory
        self.dev_sq_ema = new_dev_sq_ema
        trader_memory[dev_sq_ema_key] = new_dev_sq_ema

        # ----- 5) Build rolling array of length 40 for (dev^2 - dev_sq_ema) -----
        diff = dev_sq - new_dev_sq_ema
        trader_memory[dev_sq_diff_listkey].append(diff)
        if len(trader_memory[dev_sq_diff_listkey]) > 40:
            trader_memory[dev_sq_diff_listkey].pop(0)

        # ----- 6) Compute the Z-score of [ dev^2 - dev_sq_ema ] once we have 40 data points
        if len(trader_memory[dev_sq_diff_listkey]) == 40:
            arr = np.array(trader_memory[dev_sq_diff_listkey], dtype=float)
            mean_ = np.mean(arr)
            std_ = np.std(arr)
            if std_ < 1e-12:
                z_score = 0.0
            else:
                z_score = (diff - mean_) / std_
        else:
            z_score = 0.0

        # ----- 7) Keep an EMA of that z-score with alpha=0.01 -----
        alpha_z = 0.1
        old_devz_ema = trader_memory.get(devz_ema_key, 0.0)
        new_devz_ema = (1 - alpha_z) * old_devz_ema + alpha_z * z_score

        # Store final result
        self.deviation_zscore_ema = new_devz_ema
        trader_memory[devz_ema_key] = new_devz_ema

        return trader_memory

    
    def hist_mid_prc(self, size:int) -> np.ndarray:
        return (self.hist_order_depth('bidprc', 1, size) + self.hist_order_depth('askprc', 1, size)) / 2
    def hist_maxamt_askprc(self, size:int) -> np.ndarray:
        res_array = np.empty(size)
        prc_array = np.array([self.hist_order_depth('askprc', 1, size), self.hist_order_depth('askprc', 2, size), self.hist_order_depth('askprc', 3, size)]).T
        amt_array = np.array([self.hist_order_depth('askamt', 1, size), self.hist_order_depth('askamt', 2, size), self.hist_order_depth('askamt', 3, size)]).T

        for i, amt_arr in enumerate(amt_array):
            res_array[i] = prc_array[i,np.nanargmax(amt_arr)]
        return res_array

    def hist_maxamt_bidprc(self, size:int) -> np.ndarray:
        res_array = np.empty(size)
        prc_array = np.array([self.hist_order_depth('bidprc', 1, size), self.hist_order_depth('bidprc', 2, size), self.hist_order_depth('bidprc', 3, size)]).T
        amt_array = np.array([self.hist_order_depth('bidamt', 1, size), self.hist_order_depth('bidamt', 2, size), self.hist_order_depth('bidamt', 3, size)]).T

        for i, amt_arr in enumerate(amt_array):
            res_array[i] = prc_array[i,np.nanargmax(amt_arr)]
        return res_array

    def hist_vwap_all(self, size:int) -> np.ndarray:
        res_array = np.zeros(size)
        volsum_array = np.zeros(size)
        for i in range(1,4):
            tmp_bid_vol = self.hist_order_depth(f'bidamt', i, size)
            tmp_ask_vol = self.hist_order_depth(f'askamt', i, size)
            tmp_bid_prc = self.hist_order_depth(f'bidprc', i, size)
            tmp_ask_prc = self.hist_order_depth(f'askprc', i, size)
            if i == 0:
                res_array = res_array + (tmp_bid_prc*tmp_bid_vol) + (-tmp_ask_prc*tmp_ask_vol)
                volsum_array = volsum_array + tmp_bid_vol - tmp_ask_vol
            else:
                bid_nan_idx = np.isnan(tmp_bid_prc)
                ask_nan_idx = np.isnan(tmp_ask_prc)
                res_array = res_array + np.where(bid_nan_idx, 0, tmp_bid_prc*tmp_bid_vol) + np.where(ask_nan_idx, 0, -tmp_ask_prc*tmp_ask_vol)
                volsum_array = volsum_array + np.where(bid_nan_idx, 0, tmp_bid_vol) - np.where(ask_nan_idx, 0, tmp_ask_vol)
        return res_array / volsum_array

    def hist_obs_humidity(self, size:int) -> np.ndarray:
        return np.array(self._hist_observation['humidity'][-size:], dtype=np.float32)
    
    def hist_obs_sunlight(self, size:int) -> np.ndarray:
        return np.array(self._hist_observation['sunlight'][-size:], dtype=np.float32)
    
    def hist_obs_transportFees(self, size:int) -> np.ndarray:
        return np.array(self._hist_observation['transportFees'][-size:], dtype=np.float32)
    
    def hist_obs_exportTariff(self, size:int) -> np.ndarray:
        return np.array(self._hist_observation['exportTariff'][-size:], dtype=np.float32)
    
    def hist_obs_importTariff(self, size:int) -> np.ndarray:
        return np.array(self._hist_observation['importTariff'][-size:], dtype=np.float32)
    
    def hist_obs_bidPrice(self, size:int) -> np.ndarray:
        return np.array(self._hist_observation['bidPrice'][-size:], dtype=np.float32)
    
    def hist_obs_askPrice(self, size:int) -> np.ndarray:
        return np.array(self._hist_observation['askPrice'][-size:], dtype=np.float32)

    @property
    def best_bid(self) -> int:
        buy_orders = self._state.order_depths[self.product].buy_orders
        if len(buy_orders) > 0:
            return max(buy_orders.keys())
        else:
            return self.best_ask - 1

    @property
    def best_ask(self) -> int:
        sell_orders = self._state.order_depths[self.product].sell_orders
        if len(sell_orders) > 0:
            return min(sell_orders.keys())
        else:
            return self.best_bid + 1
    @property
    def mid(self) -> float:
        return (self.best_bid + self.best_ask) / 2
    
    @property
    def bid_ask_spread(self) -> int:
        return self.best_ask - self.best_bid

    @property
    def best_bid_amount(self) -> int:
        best_prc = max(self._state.order_depths[self.product].buy_orders.keys())
        best_amt = self._state.order_depths[self.product].buy_orders[best_prc]
        return best_amt
        
    @property
    def best_ask_amount(self) -> int:
        best_prc = min(self._state.order_depths[self.product].sell_orders.keys())
        best_amt = self._state.order_depths[self.product].sell_orders[best_prc]
        return -best_amt
    
    @property
    def volume_weighted_mid_price(self) -> float:    
        return (self.best_bid * abs(self.best_ask_amount) + self.best_ask * self.best_bid_amount) / (abs(self.best_ask_amount) + self.best_bid_amount)
    @property
    def worst_bid(self) -> int:
        buy_orders = self._state.order_depths[self.product].buy_orders
        if len(buy_orders) > 0:
            return min(buy_orders.keys())
        else:
            return self.best_ask - 1

    @property
    def worst_ask(self) -> int:
        sell_orders = self._state.order_depths[self.product].sell_orders
        if len(sell_orders) > 0:
            return max(sell_orders.keys())
        else:
            return self.best_bid + 1

    @property
    def vwap(self) -> float:
        vwap = 0
        total_amt = 0

        for prc, amt in self._state.order_depths[self.product].buy_orders.items():
            vwap += (prc * amt)
            total_amt += amt

        for prc, amt in self._state.order_depths[self.product].sell_orders.items():
            vwap += (prc * abs(amt))
            total_amt += abs(amt)

        vwap /= total_amt
        return vwap

    @property
    def vwap_bidprc(self) -> float:
        vwap = 0
        for prc, amt in self._state.order_depths[self.product].buy_orders.items():
            vwap += (prc * amt)
        vwap /= sum(self._state.order_depths[self.product].buy_orders.values())
        return vwap
    
    @property
    def vwap_askprc(self) -> float:
        vwap = 0
        for prc, amt in self._state.order_depths[self.product].sell_orders.items():
            vwap += (prc * -amt)
        vwap /= -sum(self._state.order_depths[self.product].sell_orders.values())
        return vwap
    
    @property
    def maxamt_bidprc(self) -> int:
        prc_max_mat, max_amt = 0,0
        for prc, amt in self._state.order_depths[self.product].buy_orders.items():
            if amt > max_amt:
                max_amt = amt
                prc_max_mat = prc

        return prc_max_mat
    @property
    def maxamt_askprc(self) -> int:

        prc_max_mat, max_amt = 0,0
        for prc, amt in self._state.order_depths[self.product].sell_orders.items():
            if amt < max_amt:
                max_amt = amt
                prc_max_mat = prc

        return prc_max_mat
    @property
    def maxamt_midprc(self) -> float:
        return (self.maxamt_bidprc + self.maxamt_askprc) / 2

    def bidamt(self, price) -> int:
        order_depth = self._state.order_depths[self.product].buy_orders
        if price in order_depth.keys():
            return order_depth[price]
        else:
            return 0
        
    def askamt(self, price) -> int:
        order_depth = self._state.order_depths[self.product].sell_orders
        if price in order_depth.keys():
            return order_depth[price]
        else:
            return 0
        
    @property
    def top_lyr_skew(self) -> float:
        if abs(self.best_ask_amount) != 0:
            return np.log(self.best_bid_amount/self.best_ask_amount)
        else:
            return 0
    
    @property
    def bid_gap(self) -> int:
        return self.best_bid - self.maxamt_bidprc
    
    @property
    def ask_gap(self) -> int:
        return self.maxamt_askprc - self.best_ask
    
    @property
    def my_avg_execution(self) -> float:
        my_trades = self._state.own_trades.get(self.product, [])
        if my_trades:
            return sum(t.price for t in my_trades) / len(my_trades)
        else:
            return self.volume_weighted_mid_price

    @property
    def total_bidamt(self) -> int:
        return sum(self._state.order_depths[self.product].buy_orders.values())

    @property
    def total_askamt(self) -> int:
        return -sum(self._state.order_depths[self.product].sell_orders.values())

    
    @property
    def stoarageFees(self) -> float:
        return 0.1
    
    @property
    def transportFees(self) -> float:
        return self._state.observations.conversionObservations[self.product].transportFees
    
    @property
    def exportTariff(self) -> float:
        return self._state.observations.conversionObservations[self.product].exportTariff
    
    @property
    def importTariff(self) -> float:
        return self._state.observations.conversionObservations[self.product].importTariff
    
    @property
    def sunlight(self) -> float:
        return self._state.observations.conversionObservations[self.product].sunlight
    
    @property
    def humidity(self) -> float:
        return self._state.observations.conversionObservations[self.product].humidity

    @property
    def market_trades(self) -> list:
        return self._state.market_trades.get(self.product, [])
    
    @property
    def my_trades(self) -> Dict:
        return self._state.own_trades.get(self.product, [])
class Strategy:
    @staticmethod
    def resin_fair_value_arb(state: Status, fair_price):
        orders = []

        # arbitrage opportunity on the bids so sell here
        if state.best_bid > 9999:
            if state.second_bid is not None and state.second_bid > 9999 and state.rt_position > 0:
                orders.append(Order(state.product, state.second_bid, -state.possible_sell_amt))
                if state.maxamt_bidprc + 1 < state.second_bid:
                    orders.append(Order(state.product, state.maxamt_bidprc + 1, state.possible_buy_amt))
                else:
                    orders.append(Order(state.product, state.second_bid-1, state.possible_buy_amt))
            else:
                orders.append(Order(state.product, state.best_bid, -state.best_bid_amount))
                remaining_order = state.possible_sell_amt - state.best_bid_amount
                if remaining_order > 0:
                    orders.append(Order(state.product, state.best_ask-1, - remaining_order))
                
                if state.second_bid is not None and state.second_bid + 1 < state.best_bid:
                    orders.append(Order(state.product, state.second_bid+1, state.possible_buy_amt))
                elif state.maxamt_bidprc + 1 < state.best_bid:
                    orders.append(Order(state.product, state.maxamt_bidprc+1, state.possible_buy_amt))
        
        # arbitrage opportunity on the asks so buy here
        elif state.best_ask < 10001:
            if state.second_ask is not None and state.second_ask < 10001 and state.rt_position < 0:
                orders.append(Order(state.product, state.second_ask, state.possible_buy_amt))
                if state.maxamt_askprc - 1 > state.second_ask:
                    orders.append(Order(state.product, state.maxamt_askprc - 1, -state.possible_sell_amt))
                else:
                    orders.append(Order(state.product, state.second_ask + 1, -state.possible_sell_amt))
            else:
                orders.append(Order(state.product, state.best_ask, abs(state.best_ask_amount)))
                remaining_order = state.possible_buy_amt - abs(state.best_ask_amount)
                if remaining_order > 0:
                    orders.append(Order(state.product, state.best_bid + 1, remaining_order))
                if state.second_ask is not None and state.second_ask - 1 > state.best_ask:
                    orders.append(Order(state.product, state.second_ask -1, -state.possible_sell_amt))
                elif state.maxamt_askprc - 1 > state.best_ask:
                    orders.append(Order(state.product, state.maxamt_askprc - 1, -state.possible_sell_amt))

        # No arbitrage opportunity, simple market making
        elif state.position > 49:
            orders.append(Order(state.product, 9999, -state.possible_sell_amt))
        elif state.position < -49:
            orders.append(Order(state.product, 10001, state.possible_sell_amt))

        else:
            if state.bid_ask_spread > 1:
                orders.append(Order(state.product, state.best_bid+1, state.possible_buy_amt))
                orders.append(Order(state.product, state.best_ask-1, -state.possible_sell_amt))
            else:
                orders.append(Order(state.product, state.best_bid, state.possible_buy_amt))
                orders.append(Order(state.product, state.best_ask, -state.possible_sell_amt))

        return orders             

    @staticmethod
    def kelp_orderbook_gap_arb(state: Status):
        orders = []

        if state.bid_gap > 2:
            orders.append(Order(state.product, state.best_bid, -state.possible_sell_amt))
            orders.append(Order(state.product, state.maxamt_bidprc + 1, state.possible_buy_amt))
        
        elif state.ask_gap > 2:
            orders.append(Order(state.product, state.best_ask, state.possible_buy_amt))
            orders.append(Order(state.product, state.maxamt_askprc - 1, -state.possible_sell_amt))
        
        return orders
    
    @staticmethod
    def kelp_clear_inventory_at_fair_price(state: Status, threshold_position = 20, fair_price_threshold = 0):
        orders = []

        if state.rt_position > threshold_position and state.maxamt_midprc - state.best_bid <= fair_price_threshold:
            orders.append(Order(state.product, state.best_bid, -state.possible_sell_amt))
            orders.append(Order(state.product, state.best_bid - 1, state.possible_buy_amt))
        
        elif state.rt_position < -threshold_position and state.best_ask - state.maxamt_midprc <= fair_price_threshold:
            orders.append(Order(state.product, state.best_ask, state.possible_buy_amt))
            orders.append(Order(state.product, state.best_ask + 1, -state.possible_sell_amt))
        
        return orders
    
    @staticmethod
    def kelp_second_orderbook_gap_arb(state: Status):
        orders = []
        
        if state.bid_gap > 1 and state.best_ask==state.maxamt_askprc:
            orders.append(Order(state.product, state.best_bid, -state.possible_sell_amt))
            orders.append(Order(state.product, state.maxamt_bidprc + 1, state.possible_buy_amt))
        
        elif state.ask_gap > 1 and state.best_bid==state.best_bid_amount:
            orders.append(Order(state.product, state.best_ask, state.possible_buy_amt))
            orders.append(Order(state.product, state.maxamt_askprc - 1, -state.possible_sell_amt))

        return orders
    
    @staticmethod
    def kelp_toplyr_skew_arbitrage(state: Status,skew_thresh = 1, spread_thresh = 2, threshold_position = 40):
        orders = []

        if state.top_lyr_skew > skew_thresh and state.bid_ask_spread < spread_thresh and state.rt_position < threshold_position:
            if state.top_lyr_skew > 1:
                orders.append(Order(state.product, state.best_ask, state.possible_buy_amt))
                orders.append(Order(state.product, state.best_ask + 1, -state.possible_sell_amt))
            else:
                orders.append(Order(state.product, state.best_ask, state.possible_buy_amt))
        
        elif state.top_lyr_skew < -skew_thresh and state.bid_ask_spread < spread_thresh and state.rt_position > -threshold_position:
            if state.top_lyr_skew < -1:
                orders.append(Order(state.product, state.best_bid, -state.possible_sell_amt))
                orders.append(Order(state.product, state.best_bid - 1, state.possible_buy_amt))
            else:
                orders.append(Order(state.product, state.best_bid, -state.possible_sell_amt))
        
        return orders
    
    @staticmethod
    def kelp_volatility_trading(state: Status, standardized_return_threshold=3):
        orders = []

        if state.standardized_log_return > standardized_return_threshold and state.best_bid > state.my_avg_execution:
            if state.rt_position > 0:
                orders.append(Order(state.product, state.best_bid, -state.rt_position))
            else:
                orders.append(Order(state.product, state.best_bid, -abs(state.best_bid_amount)))

        elif state.standardized_log_return < -standardized_return_threshold and state.best_ask < state.my_avg_execution:
            if state.rt_position < 0:
                orders.append(Order(state.product, state.best_ask, abs(state.rt_position)))
            else:
                orders.append(Order(state.product, state.best_ask, abs(state.best_ask_amount)))

        return orders
    
    @staticmethod
    def kelp_market_maker_analysis(state: Status, market_maker_skew_thresh = 0.4):

        orders = []

        if state.best_ask == state.maxamt_askprc and state.best_bid == state.maxamt_bidprc:
            
            if state.top_lyr_skew > market_maker_skew_thresh:
                orders.append(Order(state.product, state.maxamt_askprc + 1, -state.possible_sell_amt))
                orders.append(Order(state.product, state.maxamt_bidprc + 1, state.possible_buy_amt))
            
            elif state.top_lyr_skew < -market_maker_skew_thresh:
                orders.append(Order(state.product, state.maxamt_askprc - 1, -state.possible_sell_amt))
                orders.append(Order(state.product, state.maxamt_bidprc - 1, state.possible_buy_amt))
            
            else:
                orders.append(Order(state.product, state.maxamt_askprc - 1, -state.possible_sell_amt))
                orders.append(Order(state.product, state.maxamt_bidprc + 1, state.possible_buy_amt))
        
        return orders
    
    @staticmethod
    def kelp_second_toplyr_skew(state: Status, skew_thresh = 3, spread_thresh = 2, threshold_position = 40):

        orders = []

        if state.top_lyr_skew > skew_thresh and state.bid_ask_spread > spread_thresh and state.rt_position < threshold_position:
                orders.append(Order(state.product, state.best_ask, -state.possible_sell_amt))
                orders.append(Order(state.product, state.best_bid - 1, state.possible_buy_amt))
        
        elif state.top_lyr_skew < -skew_thresh and state.bid_ask_spread > spread_thresh and state.rt_position > -threshold_position:
                orders.append(Order(state.product, state.best_bid, state.possible_buy_amt))
                orders.append(Order(state.product, state.best_ask + 1, -state.possible_sell_amt))
        
        return orders

    @staticmethod
    def kelp_final(state: Status, spread_thresh = 2):

        orders = []

        if state.bid_ask_spread > spread_thresh:
            if state.rt_position == 0:
                pos_fact = 0
            else:
                pos_fact = state.rt_position*(int(round(np.log(abs(state.rt_position)))))/abs(state.rt_position) 

        
            orders.append(Order(state.product, int(round(state.maxamt_askprc - 1)), -state.possible_sell_amt))
            orders.append(Order(state.product, int(round(state.maxamt_bidprc + 1)), state.possible_buy_amt))
        else:
            orders.append(Order(state.product, state.best_ask, -state.possible_sell_amt))
            orders.append(Order(state.product, state.best_bid, state.possible_buy_amt))

        return orders
class BlackScholes:
    @staticmethod
    def implied_volatility(call_price, spot, strike, time_to_expiry, max_iterations=200, tolerance=1e-10):
        low_vol = 0.01
        high_vol = 1.0
        volatility = (low_vol + high_vol) / 2.0
        for _ in range(max_iterations):
            estimated_price = BlackScholes.black_scholes_call(spot, strike, time_to_expiry, volatility)
            diff = estimated_price - call_price
            if abs(diff) < tolerance:
                break
            elif diff > 0:
                high_vol = volatility
            else:
                low_vol = volatility
            volatility = (low_vol + high_vol) / 2.0
        return volatility
    
    @staticmethod
    def black_scholes_call(spot, strike, time_to_expiry, volatility):
        d1 = (log(spot) - log(strike) + (0.5 * volatility * volatility) * time_to_expiry) / (volatility * sqrt(time_to_expiry))
        d2 = d1 - volatility * sqrt(time_to_expiry)
        call_price = spot * NormalDist().cdf(d1) - strike * NormalDist().cdf(d2)
        return call_price
class DistributionCalculator:
    def beta_pdf(self, u, alpha, beta):
        if u < 0.0 or u > 1.0:
            return 0.0
        beta_func = math.gamma(alpha) * math.gamma(beta) / math.gamma(alpha + beta)
        return (u**(alpha - 1) * (1.0 - u)**(beta - 1)) / beta_func

    def build_mirrored_beta_distribution(self, a, b, alphaL, alphaR, steps=100):
        if not (a > b):
            raise ValueError("For mirrored distribution, expect a > b.")
        x_vals = np.linspace(a, b, steps)
        # Map x to u in [0,1] in descending order:
        def x_to_u(x): return (a - x) / (a - b)
        pdf_u = np.array([self.beta_pdf(x_to_u(x), alphaL, alphaR) for x in x_vals])
        pdf_x = pdf_u / abs(b - a)
        cdf = np.zeros_like(pdf_x)
        for i in range(1, steps):
            dx = x_vals[i] - x_vals[i - 1]
            cdf[i] = cdf[i - 1] + 0.5 * (pdf_x[i] + pdf_x[i - 1]) * abs(dx)
        def cdf_at(x):
            if x >= a:
                return 0.0
            if x <= b:
                return 1.0
            # Reverse arrays for ascending order:
            return np.interp(x, x_vals[::-1], cdf[::-1])
        return x_vals, pdf_x, cdf, cdf_at

    def build_ascending_beta_distribution(self, a, b, alphaL, alphaR, steps=100):
        if not (a < b):
            raise ValueError("For ascending distribution, expect a < b.")
        x_vals = np.linspace(a, b, steps)
        def x_to_u(x): return (x - a) / (b - a)
        pdf_u = np.array([self.beta_pdf(x_to_u(x), alphaL, alphaR) for x in x_vals])
        pdf_x = pdf_u / (b - a)
        cdf = np.zeros_like(pdf_x)
        for i in range(1, steps):
            dx = x_vals[i] - x_vals[i - 1]
            cdf[i] = cdf[i - 1] + 0.5 * (pdf_x[i] + pdf_x[i - 1]) * dx
        def cdf_at(x):
            if x <= a:
                return 0.0
            if x >= b:
                return 1.0
            return np.interp(x, x_vals, cdf)
        return x_vals, pdf_x, cdf, cdf_at

    def precompute_distributions(self):
        # For buy positions (mirrored beta)
        alphaL_buy, alphaR_buy = 2.0, 4.0
        buy_a, buy_b = 320.0, 200.0
        _, _, _, cdf_buy_at = self.build_mirrored_beta_distribution(buy_a, buy_b, alphaL_buy, alphaR_buy, steps=100)
        self.cdf_buy_at = cdf_buy_at

        # For sell positions (ascending beta)
        alphaL_sell, alphaR_sell = 2.0, 4.0
        sell_a, sell_b = 390.0, 500.0
        _, _, _, cdf_sell_at = self.build_ascending_beta_distribution(sell_a, sell_b, alphaL_sell, alphaR_sell, steps=100)
        self.cdf_sell_at = cdf_sell_at

    def precompute_iv_distributions(self):
        # For IV buy positions
        alphaL_buy, alphaR_buy = 4,2.0
        buy_a, buy_b = 100, 97
        _, _, _, cdf_buy_at_iv = self.build_mirrored_beta_distribution(buy_a, buy_b, alphaL_buy, alphaR_buy, steps=3)
        self.cdf_buy_at_iv = cdf_buy_at_iv

        # For IV sell positions
        alphaL_sell, alphaR_sell = 4.0, 2.0
        sell_a, sell_b = 100, 103
        _, _, _, cdf_sell_at_iv = self.build_ascending_beta_distribution(sell_a, sell_b, alphaL_sell, alphaR_sell, steps=3)
        self.cdf_sell_at_iv = cdf_sell_at_iv
class Trade:
    @staticmethod
    def resin(state: Status) -> list[Order]:
        fair = 10000
        orders = []
        orders.extend(Strategy.resin_fair_value_arb(state=state, fair_price=fair))
        return orders
    
    @staticmethod
    def kelp(state: Status) -> list[Order]:
        kelp_orders = []
        if not kelp_orders:
            kelp_orders = Strategy.kelp_orderbook_gap_arb(state=state)
        if not kelp_orders:
            kelp_orders = Strategy.kelp_second_orderbook_gap_arb(state=state)
        if not kelp_orders:
            kelp_orders = Strategy.kelp_clear_inventory_at_fair_price(state=state)
        if not kelp_orders:
            kelp_orders = Strategy.kelp_toplyr_skew_arbitrage(state=state)
        if not kelp_orders:
            kelp_orders = Strategy.kelp_market_maker_analysis(state=state)
        if not kelp_orders:
            kelp_orders = Strategy.kelp_volatility_trading(
                state=state)              
        if not kelp_orders:
            kelp_orders = Strategy.kelp_second_toplyr_skew(state=state)
        if not kelp_orders:
            kelp_orders = Strategy.kelp_final(state=state)
            
        return kelp_orders
    
    @staticmethod
    def volcano(trader, state: TradingState, trader_memory: dict) -> tuple[dict[str, list[Order]], dict]:
        result = {}
        
        # 1) restore EMAs from memory
        if "ema_iv" in trader_memory:
            trader.ema_iv = trader_memory["ema_iv"]
        if "ema_iv_9750" in trader_memory:
            trader.ema_iv_9750 = trader_memory["ema_iv_9750"]

        # 2) compute underlying rock mid, voucher mid, vol, regime, update EMAs...
        # (copy your monolith here, but replace self→trader, and store
        # orders into `result[voucher_symbol] = [Order(...), …]`)
        if "ema_iv" in trader_memory:
            trader.ema_iv = trader_memory["ema_iv"]
            # print(f"Yay for 10000, self.ema_iv is {self.ema_iv}")
        if "ema_iv_9750" in trader_memory:
            trader.ema_iv_9750 = trader_memory["ema_iv_9750"]
            # print(f"Yay for 9750,self.ema_iv is {self.ema_iv_9750}")

        conversions = 0
        
        if "VOLCANIC_ROCK_VOUCHER_10000" in state.order_depths:
            voucher_pos = state.position.get("VOLCANIC_ROCK_VOUCHER_10000", 0)
            rock_pos = state.position.get("VOLCANIC_ROCK", 0)
            max_pos = trader.status_voucher_10000._position_limit[trader.status_voucher_10000.product]
 

            # Get the underlying rock's order depth to compute the rock mid price.
            rock_depth = state.order_depths["VOLCANIC_ROCK"]
            rock_bid = max(rock_depth.buy_orders.keys())
            rock_ask = min(rock_depth.sell_orders.keys())
            rock_mid = (min(rock_depth.buy_orders.keys()) + max(rock_depth.sell_orders.keys())) / 2

            # Get the voucher’s order depth.
            voucher_depth = state.order_depths["VOLCANIC_ROCK_VOUCHER_10000"]
            if not voucher_depth.buy_orders or not voucher_depth.sell_orders:
                result["VOLCANIC_ROCK_VOUCHER_10000"] = []
            else:
                voucher_bid = max(voucher_depth.buy_orders.keys())
                voucher_ask = min(voucher_depth.sell_orders.keys())
                voucher_bid_amt = voucher_depth.buy_orders[voucher_bid]
                voucher_ask_amt = abs(voucher_depth.sell_orders[voucher_ask])
                voucher_skew = 0 if voucher_bid_amt == 0 or voucher_ask_amt == 0 else np.log(voucher_bid_amt / voucher_ask_amt)

                # Compute the mid price for the voucher.
                voucher_mid = trader.get_voucher_mid_price(voucher_depth, state.traderData if state.traderData else {})

                # Time-to-expiry (assuming 7 days with rounds as days)
                # tte = 6/247  # adjust if needed

                tte = (
                (2/247)
            )
                # Compute the implied volatility using the voucher’s mid price and the rock mid price.
# For the voucher you are processing (e.g., VOLCANIC_ROCK_VOUCHER_10500)

                volatility = BlackScholes.implied_volatility(voucher_mid, rock_mid, 10000, tte)

                if rock_mid >= 10250:
                    regime = "10000"
                else:
                    if "VOLCANIC_ROCK_VOUCHER_9750" in state.order_depths:
                        voucher_depth_9750 = state.order_depths["VOLCANIC_ROCK_VOUCHER_9750"]
                        if voucher_depth_9750.buy_orders and voucher_depth_9750.sell_orders:
                            voucher_mid_9750 = trader.get_voucher_mid_price(voucher_depth_9750, state.traderData if state.traderData else {})
                            vol_9750 = BlackScholes.implied_volatility(voucher_mid_9750, rock_mid, 9750, tte)

                            if vol_9750 < 1 and vol_9750 > 0:
                                regime = "9750"
                                if trader.ema_iv_9750 is None:
                                    trader.ema_iv_9750 = vol_9750
                                else:
                                    trader.ema_iv_9750 = 0.0055 * vol_9750 + (1 - 0.0055) * trader.ema_iv_9750
                            else:
                                regime = "10000"
                        else:
                            regime = "10000"

                
                # Update our slow EMA for IV.
                if trader.last_vol is None:
                    trader.last_vol = volatility

                if trader.ema_iv is None:
                    trader.ema_iv = volatility

                else:
                    trader.ema_iv = trader.alpha_iv * volatility + (1 - trader.alpha_iv) * trader.ema_iv

                if regime == "10000":
                    # Use default EMA (or volatility, depending on your window check)
                    if state.timestamp < trader.window:
                        trader.update_iv_distribution(volatility, volatility)
                    else:
                        trader.update_iv_distribution(trader.ema_iv, volatility)

                else:
                    # regime is "9750"
                    # For voucher 10000: use the default distribution.
                    if state.timestamp < trader.window:
                        trader.update_iv_distribution(volatility, volatility)
                    else:
                        trader.update_iv_distribution(trader.ema_iv, volatility)
                        
                    # For other vouchers, update alternate Beta distributions with different offsets.
                    alt_dynamic_buy_a = trader.ema_iv_9750 - 0.001# alternative offset
                    alt_dynamic_buy_b = alt_dynamic_buy_a - 0.004
                    alt_dynamic_sell_a = trader.ema_iv_9750 + 0.001
                    alt_dynamic_sell_b = alt_dynamic_sell_a + 0.004

                    alphaL_buy_alt, alphaR_buy_alt = 5.0, 1.0
                    _, _, _, cdf_buy_at_iv_alt = trader.distribution_calc.build_mirrored_beta_distribution(
                        alt_dynamic_buy_a, alt_dynamic_buy_b, alphaL_buy_alt, alphaR_buy_alt, steps=100
                    )

                    alphaL_sell_alt, alphaR_sell_alt = 5.0, 1.0
                    _, _, _, cdf_sell_at_iv_alt = trader.distribution_calc.build_ascending_beta_distribution(
                        alt_dynamic_sell_a, alt_dynamic_sell_b, alphaL_sell_alt, alphaR_sell_alt, steps=100
                    )

                    trader.distribution_calc.cdf_buy_at_iv_alt = cdf_buy_at_iv_alt
                    trader.distribution_calc.cdf_sell_at_iv_alt = cdf_sell_at_iv_alt

                    # --- Double up: Second alternative distribution with different parameters ---
                    # These new parameters can be tuned as desired.
                    alt2_dynamic_buy_a = trader.ema_iv - 0.001  # different offset
                    alt2_dynamic_buy_b = alt2_dynamic_buy_a - 0.002
                    alt2_dynamic_sell_a = trader.ema_iv + 0.001
                    alt2_dynamic_sell_b = alt2_dynamic_sell_a + 0.002

                    alphaL_buy_alt2, alphaR_buy_alt2 = 5, 1 # for example, slightly adjusted shape parameters
                    _, _, _, cdf_buy_at_iv_alt2 = trader.distribution_calc.build_mirrored_beta_distribution(
                        alt2_dynamic_buy_a, alt2_dynamic_buy_b, alphaL_buy_alt2, alphaR_buy_alt2, steps=100
                    )

                    alphaL_sell_alt2, alphaR_sell_alt2 = 5, 1
                    _, _, _, cdf_sell_at_iv_alt2 = trader.distribution_calc.build_ascending_beta_distribution(
                        alt2_dynamic_sell_a, alt2_dynamic_sell_b, alphaL_sell_alt2, alphaR_sell_alt2, steps=100
                    )

                    trader.distribution_calc.cdf_buy_at_iv_alt2 = cdf_buy_at_iv_alt2
                    trader.distribution_calc.cdf_sell_at_iv_alt2 = cdf_sell_at_iv_alt2
                
                voucher_list = [
                    "VOLCANIC_ROCK_VOUCHER_9500",
                    "VOLCANIC_ROCK_VOUCHER_9750",
                    "VOLCANIC_ROCK_VOUCHER_10000",
                    "VOLCANIC_ROCK_VOUCHER_10250",
                    "VOLCANIC_ROCK_VOUCHER_10500",
                ]
                for voucher in voucher_list:
                    if voucher in state.order_depths:
                        depth = state.order_depths[voucher]
                        # If the voucher order depth is not liquid, skip it.
                        if not depth.buy_orders or not depth.sell_orders:
                            result[voucher] = []
                            continue
                        my_trades = state.own_trades.get(voucher, [])
                        avg_exec_price  = (sum(t.price for t in my_trades) / len(my_trades)) if my_trades else 0.0 
                        
                        # if regime == "9750" and voucher in ["VOLCANIC_ROCK_VOUCHER_9500","VOLCANIC_ROCK_VOUCHER_10250", "VOLCANIC_ROCK_VOUCHER_10500"]:
                        if voucher in ["VOLCANIC_ROCK_VOUCHER_9500","VOLCANIC_ROCK_VOUCHER_10250", "VOLCANIC_ROCK_VOUCHER_10500"]:
            
                            current_position = state.position.get(voucher, 0)
                            best_ask_v = min(depth.sell_orders.keys())
                            best_bid_v = max(depth.buy_orders.keys())
                            if current_position > 0 and best_bid_v > avg_exec_price:
                                result[voucher] = [Order(voucher, best_bid_v, -current_position)]
                            elif current_position < 0 and best_ask_v < avg_exec_price:
                                result[voucher] = [Order(voucher, best_ask_v, -current_position)]  # Do not trade these instruments.
                            continue

                        # Compute the voucher's current mid price.
                        vb = max(depth.buy_orders.keys())
                        va = min(depth.sell_orders.keys())
                        current_mid = (vb + va) / 2
                        # Use the benchmark scaled_iv (from the 10000 voucher) for all vouchers.
                        # Compute the Beta–CDF fractions.
                        pos = state.position.get(voucher, 0)
                        orders_limit_buy = max(0, 200 - pos)
                        orders_limit_sell = max(0, 200 + pos)             
                        
                        if regime == "9750":
                            if voucher == "VOLCANIC_ROCK_VOUCHER_9750" :
                                fraction_buy = trader.distribution_calc.cdf_buy_at_iv_alt(vol_9750)
                                fraction_sell = trader.distribution_calc.cdf_sell_at_iv_alt(vol_9750)
                            
                            if voucher == "VOLCANIC_ROCK_VOUCHER_10000":
                                fraction_buy = trader.distribution_calc.cdf_buy_at_iv_alt2(volatility)
                                fraction_sell = trader.distribution_calc.cdf_sell_at_iv_alt2(volatility)                               
                        
                        else:
 
                            fraction_buy = trader.distribution_calc.cdf_buy_at_iv(volatility)
                            fraction_sell = trader.distribution_calc.cdf_sell_at_iv(volatility)
                        # Determine a target position; here using a notional of 600. You can adjust.
                        
                        target_position = int(round(fraction_buy * orders_limit_buy)) - int(round(fraction_sell * orders_limit_sell))

                        my_trades = state.own_trades.get(voucher, [])
                        avg_exec_price  = (sum(t.price for t in my_trades) / len(my_trades)) if my_trades else 0.0 
                        
                        current_position = state.position.get(voucher, 0)
                        pos_diff = target_position - current_position
                        # print(f"For {voucher}: EMA: {trader.ema_iv:.2f}, Scaled IV: {volatility:.2f}, "
                        #       f"target_position: {target_position}, current: {current_position}")
                        if target_position > 0:
                            # If target is higher, buy the difference.
                            best_ask_v = min(depth.sell_orders.keys())
                            best_bid_v = max(depth.buy_orders.keys())
                            buy_qty = min(pos_diff, 200 - current_position)
                            if buy_qty > 0:
                                result[voucher] = [Order(voucher, best_ask_v, buy_qty)]
                                # print(f"buying at the price {current_mid}, vol {volatility}, time {state.timestamp}")
                        elif target_position < 0:
                            # If target is lower, sell the difference.
                            best_bid_v = max(depth.buy_orders.keys())
                            best_ask_v = min(depth.sell_orders.keys())
                            sell_qty = min(abs(pos_diff), 200 + current_position)
                            if sell_qty > 0:
                                result[voucher] = [Order(voucher, best_bid_v, -sell_qty)]
                                # print(f"SELLING at the price {current_mid}, vol {volatility}, time {state.timestamp}")
                        else:
                            result[voucher] = []

        else:
            for voucher in ["VOLCANIC_ROCK_VOUCHER_9500",
                            "VOLCANIC_ROCK_VOUCHER_9750",
                            "VOLCANIC_ROCK_VOUCHER_10000",
                            "VOLCANIC_ROCK_VOUCHER_10250",
                            "VOLCANIC_ROCK_VOUCHER_10500"]:
                result[voucher] = []

        trader_memory["ema_iv"] = trader.ema_iv
        trader_memory["ema_iv_9750"] = trader.ema_iv_9750
        return result, trader_memory 
class CrossStrategy:
    def __init__(self, params=None)->None:

        self._params = params or {}
        self.long_arb_z_val = 0.0
        self.short_arb_z_val = 0.0
        self.short_arb_ema = None
        self.long_arb_ema = None

        self.long_profit_thresh = 0.0
        self.short_profit_thresh = 0.0

        # self.mean = self._params.get("mean", 0)

        self.offset_1 = self._params.get("offset_1", 30)
        self.offset_2 = self._params.get("offset_2", 60)

        self.alphaL = self._params.get("alphaL", 2)
        self.alphaR = self._params.get("alphaR", 2)
        
        self.offset_liq = self._params.get("offset_liq", 10)
    


        # Precompute the Beta distributions right at the beginning.
        self.precompute_distributions(ema_premium1=0.0, ema_premium2=0.0)

    
    def get_long_arb_profit(self, status: dict[str, Status]) -> float:
        b1_bid = status["PICNIC_BASKET1"].best_bid
        b2_ask = status["PICNIC_BASKET2"].best_ask
        d_ask  = status["DJEMBES"].best_ask
        return (2 * b1_bid) - (3 * b2_ask + 2 * d_ask)

    def get_short_arb_profit(self, status: dict[str, Status]) -> float:
        b1_ask = status["PICNIC_BASKET1"].best_ask
        b2_bid = status["PICNIC_BASKET2"].best_bid
        d_bid  = status["DJEMBES"].best_bid
        return (3 * b2_bid + 2 * d_bid) - (2 * b1_ask)
    def beta_pdf(self, u, alpha, beta):
        if u < 0.0 or u > 1.0:
            return 0.0
        beta_func = math.gamma(alpha)*math.gamma(beta)/math.gamma(alpha + beta)
        return (u**(alpha - 1) * (1.0 - u)**(beta - 1)) / beta_func

    def build_mirrored_beta_distribution(self, a, b, alphaL, alphaR, steps=100):
        """
        Exactly the same as your logic, building a descending distribution 
        on [a..b] (a> b).
        """
        if not (a > b):
            raise ValueError("For mirrored distribution, expect a > b.")

        x_vals_desc = np.linspace(a, b, steps)

        def x_to_u(x):
            return (a - x)/(a - b)

        pdf_u_desc = np.array([self.beta_pdf(x_to_u(xx), alphaL, alphaR) for xx in x_vals_desc])
        pdf_x_desc = pdf_u_desc / abs(b - a)

        cdf_desc = np.zeros_like(pdf_x_desc)
        for i in range(1, steps):
            dx = x_vals_desc[i] - x_vals_desc[i - 1]
            area = 0.5*(pdf_x_desc[i] + pdf_x_desc[i - 1])*abs(dx)
            cdf_desc[i] = cdf_desc[i - 1] + area

        def cdf_at(x):
            if x >= a:
                return 0.0
            if x <= b:
                return 1.0
            x_asc = x_vals_desc[::-1]
            cdf_asc = cdf_desc[::-1]
            return np.interp(x, x_asc, cdf_asc)

        return x_vals_desc, pdf_x_desc, cdf_desc, cdf_at

    def build_ascending_beta_distribution(self, a, b, alphaL, alphaR, steps=100):
        if not (a < b):
            raise ValueError("For ascending distribution, expect a < b.")

        x_vals = np.linspace(a, b, steps)

        def x_to_u(x):
            return (x - a)/(b - a)

        pdf_u = np.array([self.beta_pdf(x_to_u(xx), alphaL, alphaR) for xx in x_vals])
        pdf_x = pdf_u / (b - a)

        cdf_x = np.zeros_like(pdf_x)
        for i in range(1, steps):
            dx = x_vals[i] - x_vals[i-1]
            area = 0.5*(pdf_x[i] + pdf_x[i-1])*dx
            cdf_x[i] = cdf_x[i-1] + area

        def cdf_at(x):
            if x <= a:
                return 0.0
            if x >= b:
                return 1.0
            return np.interp(x, x_vals, cdf_x)

        return x_vals, pdf_x, cdf_x, cdf_at

    def precompute_distributions(self, ema_premium1:float, ema_premium2:float):
        """
        Precompute two different sets of Beta-based CDFs:
        - One pair for Basket 1.
        - One pair for Basket 2.
        """

        # --- For Basket 1:
        # Example parameters for Basket 1
        alphaL_buy_b1, alphaR_buy_b1 = 4, 3
        buy_a_b1, buy_b_b1 = ema_premium1 - self.offset_1, ema_premium1 - self.offset_1 - 80 # Note: a > b for mirrored distribution
        
        _, _, _, cdf_at_buy_b1 = self.build_mirrored_beta_distribution(
            buy_a_b1, buy_b_b1, alphaL_buy_b1, alphaR_buy_b1, steps=100
        )
        self.cdf_buy_at_basket1 = cdf_at_buy_b1

        alphaL_sell_b1, alphaR_sell_b1 = 4, 3
        sell_a_b1, sell_b_b1 = ema_premium1 + self.offset_1, ema_premium1 + self.offset_1 + 80  # a < b for ascending distribution

        _, _, _, cdf_at_sell_b1 = self.build_ascending_beta_distribution(
            sell_a_b1, sell_b_b1, alphaL_sell_b1, alphaR_sell_b1, steps=100
        )
        self.cdf_sell_at_basket1 = cdf_at_sell_b1

        # --- For Basket 2:
        # Use different parameters for Basket 2 (adjust as needed)
        alphaL_buy_b2, alphaR_buy_b2 = self.alphaL, self.alphaR 
        buy_a_b2, buy_b_b2 = ema_premium2 - self.offset_1, ema_premium2 - self.offset_1 - self.offset_2 # ensure a > b
        _, _, _, cdf_at_buy_b2 = self.build_mirrored_beta_distribution(
            buy_a_b2, buy_b_b2, alphaL_buy_b2, alphaR_buy_b2, steps=100
        )
        self.cdf_buy_at_basket2 = cdf_at_buy_b2

        alphaL_sell_b2, alphaR_sell_b2 = self.alphaL, self.alphaR
        sell_a_b2, sell_b_b2 = ema_premium2 + self.offset_1, ema_premium2 + self.offset_1 + self.offset_2 # ensure a < b
        _, _, _, cdf_at_sell_b2 = self.build_ascending_beta_distribution(
            sell_a_b2, sell_b_b2, alphaL_sell_b2, alphaR_sell_b2, steps=100
        )
        self.cdf_sell_at_basket2 = cdf_at_sell_b2


    def compute_premium_spread_basket1(self, status: dict[str, Status]) -> float:
        djembe_best_ask = status['DJEMBES'].best_ask
        basket1_ask = status['PICNIC_BASKET1'].best_ask
        basket2_ask = status['PICNIC_BASKET2'].best_ask
        croissant_ask = status['CROISSANTS'].best_ask
        jam_ask = status['JAMS'].best_ask

        djembe_best_bid = status['DJEMBES'].best_bid
        basket1_bid = status['PICNIC_BASKET1'].best_bid
        basket2_bid = status['PICNIC_BASKET2'].best_bid
        croissant_bid = status['CROISSANTS'].best_bid
        jam_bid = status['JAMS'].best_bid
       
        premium_bid_basket_1 = basket1_bid - 6*croissant_ask - 3*jam_ask - djembe_best_ask
        premium_ask_basket_1 = basket1_ask - 6*croissant_bid - 3*jam_bid - djembe_best_bid    

        return (premium_ask_basket_1+premium_bid_basket_1)/2


    def compute_premium_spread_basket2(self, status: dict[str, Status]) -> float:
        djembe_best_ask = status['DJEMBES'].best_ask
        basket1_ask = status['PICNIC_BASKET1'].best_ask
        basket2_ask = status['PICNIC_BASKET2'].best_ask
        croissant_ask = status['CROISSANTS'].best_ask
        jam_ask = status['JAMS'].best_ask

        djembe_best_bid = status['DJEMBES'].best_bid
        basket1_bid = status['PICNIC_BASKET1'].best_bid
        basket2_bid = status['PICNIC_BASKET2'].best_bid
        croissant_bid = status['CROISSANTS'].best_bid
        jam_bid = status['JAMS'].best_bid
       
        premium_bid_basket_2 = basket2_bid - 4*croissant_ask - 2*jam_ask
        premium_ask_basket_2 = basket2_ask - 4*croissant_bid - 2*jam_bid
  

        return (premium_bid_basket_2+premium_ask_basket_2)/2


    def inter_basket_arbitrage(self, status: dict[str, Status], ema_premium1, ema_premium2) -> list[Order]:
        """
        Now uses Beta distributions to decide how large a trade to do
        for 'long' vs. 'short' side, based on the current profit.
        """
        orders = []
        djembe_best_ask = status['DJEMBES'].best_ask
        basket1_ask = status['PICNIC_BASKET1'].best_ask
        basket2_ask = status['PICNIC_BASKET2'].best_ask
        croissant_ask = status['CROISSANTS'].best_ask
        jam_ask = status['JAMS'].best_ask

        djembe_best_bid = status['DJEMBES'].best_bid
        basket1_bid = status['PICNIC_BASKET1'].best_bid
        basket2_bid = status['PICNIC_BASKET2'].best_bid
        croissant_bid = status['CROISSANTS'].best_bid
        jam_bid = status['JAMS'].best_bid


        djembe_best_ask_amt = abs(status['DJEMBES'].best_ask_amount)
        basket1_ask_amt = abs(status['PICNIC_BASKET1'].best_ask_amount)
        basket2_ask_amt = abs(status['PICNIC_BASKET2'].best_ask_amount)
        croissant_ask_amt = abs(status['CROISSANTS'].best_ask_amount)
        jam_ask_amt = abs(status['JAMS'].best_ask_amount)

        djembe_best_bid_amt = status['DJEMBES'].best_bid_amount
        basket1_bid_amt = status['PICNIC_BASKET1'].best_bid_amount
        basket2_bid_amt = status['PICNIC_BASKET2'].best_bid_amount
        croissant_bid_amt = status['CROISSANTS'].best_bid_amount
        jam_bid_amt = status['JAMS'].best_bid_amount

        djembe_possible_sell = abs(status['DJEMBES'].possible_sell_amt)
        basket1_possible_sell = abs(status['PICNIC_BASKET1'].possible_sell_amt)
        basket2_possible_sell = abs(status['PICNIC_BASKET2'].possible_sell_amt)
        croissant_possible_sell = abs(status['CROISSANTS'].possible_sell_amt)
        jam_possible_sell = abs(status['JAMS'].possible_sell_amt)

        djembe_possible_buy = status['DJEMBES'].possible_buy_amt
        basket1_possible_buy = status['PICNIC_BASKET1'].possible_buy_amt
        basket2_possible_buy = status['PICNIC_BASKET2'].possible_buy_amt
        croissant_possible_buy = status['CROISSANTS'].possible_buy_amt
        jam_possible_buy = status['JAMS'].possible_buy_amt

        djembe_position = status['DJEMBES'].position
        basket1_position = status['PICNIC_BASKET1'].position
        basket2_position= status['PICNIC_BASKET2'].position
        croissant_position = status['CROISSANTS'].position
        jam_position = status['JAMS'].position

        poss_long_arb_qty = min(djembe_best_ask_amt//2, basket2_ask_amt//3, basket1_bid_amt//2)
        real_poss_long_qty = min(poss_long_arb_qty, djembe_possible_buy//2, basket1_possible_sell//2, basket2_possible_buy//3)
        send_ord_long_arb = max(real_poss_long_qty, 0)

        poss_short_arb_qty = min(djembe_best_bid_amt//2, basket1_ask_amt//2, basket2_bid_amt//3)
        real_poss_short_qty = min(poss_short_arb_qty, basket1_possible_buy//2, djembe_possible_sell//2, basket2_possible_sell//3)
        send_ord_short_arb = max(real_poss_short_qty, 0)

        long_arb_profit = (2*basket1_bid - 3*basket2_ask - 2*djembe_best_ask)
        short_arb_profit = (3*basket2_bid + 2*djembe_best_bid - 2*basket1_ask)
        max_prof = max(long_arb_profit, short_arb_profit)


        def buy_position_basket1(ask_price, scale=60):
            val = self.cdf_buy_at_basket1(ask_price)
            return scale * val

        def sell_position_basket1(bid_price, scale=60):
            val = self.cdf_sell_at_basket1(bid_price)
            return -scale * val

        # For Basket 2
        def buy_position_basket2(ask_price, scale=100):
            val = self.cdf_buy_at_basket2(ask_price)
            return scale * val

        def sell_position_basket2(bid_price, scale=100):
            val = self.cdf_sell_at_basket2(bid_price)
            return -scale * val




        premium_bid_basket_1 = basket1_bid - 6*croissant_ask - 3*jam_ask - djembe_best_ask
        premium_ask_basket_1 = basket1_ask - 6*croissant_bid - 3*jam_bid - djembe_best_bid
        
        premium_bid_basket_2 = basket2_bid - 4*croissant_ask - 2*jam_ask
        premium_ask_basket_2 = basket2_ask - 4*croissant_bid - 2*jam_bid


        premium_bid_vol_1 = min(abs(basket1_bid_amt), abs(djembe_best_ask_amt))
        premium_ask_vol_1 = min(abs(basket1_ask_amt), abs(djembe_best_bid_amt))


        premium_bid_vol_2 = basket2_bid_amt
        premium_ask_vol_2 = basket2_ask_amt



        inv_buy_side_basket_1 = buy_position_basket1(premium_ask_basket_1, scale=60)
        inv_sell_side_basket_1 = sell_position_basket1(premium_bid_basket_1, scale=60)
        inv_total_basket_1 = inv_buy_side_basket_1 + inv_sell_side_basket_1

        inv_buy_side_basket_2 = buy_position_basket2(premium_ask_basket_2, scale=100)
        inv_sell_side_basket_2 = sell_position_basket2(premium_bid_basket_2, scale=100)
        inv_total_basket_2 = inv_buy_side_basket_2 + inv_sell_side_basket_2


        
        if premium_bid_basket_1 > ema_premium1 + self.offset_1:
            if basket1_position > int(round(inv_total_basket_1)):
                ord_pos = int(round(inv_total_basket_1)) - basket1_position
                send_ord = min(abs(ord_pos), premium_bid_vol_1)
                orders.append(Order(status['PICNIC_BASKET1'].product, basket1_bid, -abs(send_ord)))
                orders.append(Order(status['DJEMBES'].product, djembe_best_ask, abs(send_ord)))

        elif premium_ask_basket_1 < ema_premium1 - self.offset_1:
            if basket1_position < int(round(inv_total_basket_1)):
                ord_pos = int(round(inv_total_basket_1)) - basket1_position
                send_ord = min(premium_ask_vol_1, ord_pos)
                orders.append(Order(status['PICNIC_BASKET1'].product, basket1_ask, abs(send_ord)))
                orders.append(Order(status['DJEMBES'].product, djembe_best_bid, -abs(send_ord)))
        
        else:
            if premium_ask_basket_1 < ema_premium1+self.offset_liq and basket1_position < 0:
                if djembe_position > 0:
                    send_ord = min(abs(basket1_position), premium_ask_vol_1)
                    orders.append(Order(status['PICNIC_BASKET1'].product, basket1_ask, send_ord))
                    orders.append(Order(status['DJEMBES'].product, djembe_best_bid, -djembe_position))
                else:
                    send_ord = min(abs(basket1_position), premium_ask_vol_1)
                    orders.append(Order(status['PICNIC_BASKET1'].product, basket1_ask, send_ord))


            elif premium_bid_basket_1 > ema_premium1-self.offset_liq and basket1_position > 0:
                if djembe_position < 0:
                    send_ord = min(basket1_position, premium_bid_vol_1)
                    dj_pos = min(send_ord, abs(djembe_position))
                    orders.append(Order(status['PICNIC_BASKET1'].product, basket1_bid, -send_ord))
                    orders.append(Order(status['DJEMBES'].product, djembe_best_ask, dj_pos))

                else:
                    send_ord = min(basket1_position, premium_bid_vol_1)
                    orders.append(Order(status['PICNIC_BASKET1'].product, basket1_bid, -send_ord))




        if premium_bid_basket_2 > ema_premium2 + self.offset_1:
            if basket2_position > int(round(inv_total_basket_2)):           
                ord_pos = int(round(inv_total_basket_2)) - basket2_position
                orders.append(Order(status['PICNIC_BASKET2'].product, basket2_bid, -abs(ord_pos)))

        elif premium_ask_basket_2 < ema_premium2 - self.offset_1:
            if basket2_position < int(round(inv_total_basket_2)):       
                ord_pos = int(round(inv_total_basket_2)) - basket2_position
                send_ord = min(premium_ask_vol_2, ord_pos)
                orders.append(Order(status['PICNIC_BASKET2'].product, basket2_ask, abs(send_ord)))
                # orders.append(Order(Product.CROISSANTS, best_ask_crss, 4*ord_pos))
                # orders.append(Order(Product.JAMS, best_ask_jams, 2*ord_pos))

        else:

            if premium_ask_basket_2 < ema_premium2+self.offset_liq and basket2_position < 0:
                send_ord = min(abs(basket2_position), premium_ask_vol_2)
                orders.append(Order(status['PICNIC_BASKET2'].product, basket2_ask, basket2_position))
                # orders.append(Order(Product.CROISSANTS, best_ask_crss, 4*send_ord))
                # orders.append(Order(Product.JAMS, best_ask_jams, 2*send_ord))

            elif premium_bid_basket_2 > ema_premium2-self.offset_liq and basket2_position > 0:
                send_ord = min(basket2_position, premium_bid_vol_2)
                orders.append(Order(status['PICNIC_BASKET2'].product, basket2_bid, -basket2_position))      
        

        # 8) Return the set of orders
        return orders
    
class Trader:
    def __init__(
        self,
        buyer1: str | None = None,
        seller1: str | None = None,
        buyer2: str | None = None,
        seller2: str | None = None,
    ):
        # the four “bug” aliases you want to track for inter‑bug signals
        self.buyer1 = buyer1
        self.seller1 = seller1
        self.buyer2 = buyer2
        self.seller2 = seller2

        # instantiate all your Status objects exactly as before
        self.state_resin       = Status('RAINFOREST_RESIN')
        self.state_kelp        = Status('KELP')
        self.state_squid       = Status('SQUID_INK')
        self.status_croissant  = Status('CROISSANTS')
        self.status_jam        = Status('JAMS')
        self.status_djembe     = Status('DJEMBES')
        self.status_basket1    = Status('PICNIC_BASKET1')
        self.status_basket2    = Status('PICNIC_BASKET2')
        self.status_volcanic_rock     = Status('VOLCANIC_ROCK')
        self.status_voucher_9500      = Status('VOLCANIC_ROCK_VOUCHER_9500')
        self.status_voucher_9750      = Status('VOLCANIC_ROCK_VOUCHER_9750')
        self.status_voucher_10000     = Status('VOLCANIC_ROCK_VOUCHER_10000')
        self.status_voucher_10250     = Status('VOLCANIC_ROCK_VOUCHER_10250')
        self.status_voucher_10500     = Status('VOLCANIC_ROCK_VOUCHER_10500')

        # bundle them into your statuses dict
        self.statuses = {
            s.product: s for s in (
                self.status_croissant,
                self.status_jam,
                self.status_djembe,
                self.status_basket1,
                self.status_basket2,
                self.status_volcanic_rock,
                self.status_voucher_9500,
                self.status_voucher_9750,
                self.status_voucher_10000,
                self.status_voucher_10250,
                self.status_voucher_10500,
            )
        }

        # bring in your pre‑computed Distributions
        self.distribution_calc = DistributionCalculator()
        self.distribution_calc.precompute_distributions()
        self.distribution_calc.precompute_iv_distributions()

        # all your IV‐management fields
        self.last_vol      = None
        self.coupon_ema    = None
        self.non_coupon_ema= None
        self.ema_iv        = None
        self.ema_iv_9750   = None
        self.alpha_iv      = 0.0055
        self.window        = 0
        self.ema_CSI       = None
        self.sunlight_direction = 0
        self.last_sunlight = None

        # and your cross‐strategy
        self.cross_strategy = CrossStrategy()

    state_resin = Status('RAINFOREST_RESIN')
    state_kelp = Status('KELP')
    state_squid = Status('SQUID_INK')
    
    status_croissant = Status('CROISSANTS')
    status_jam = Status('JAMS')
    status_djembe = Status('DJEMBES')
    status_basket1 = Status('PICNIC_BASKET1')
    status_basket2 = Status('PICNIC_BASKET2')
    status_volcanic_rock = Status('VOLCANIC_ROCK')
    status_voucher_9500 = Status('VOLCANIC_ROCK_VOUCHER_9500')
    status_voucher_9750 = Status('VOLCANIC_ROCK_VOUCHER_9750')
    status_voucher_10000 = Status('VOLCANIC_ROCK_VOUCHER_10000')
    status_voucher_10250 = Status('VOLCANIC_ROCK_VOUCHER_10250')
    status_voucher_10500 = Status('VOLCANIC_ROCK_VOUCHER_10500')
    
    distribution_calc = DistributionCalculator()
    distribution_calc.precompute_distributions()
    distribution_calc.precompute_iv_distributions()
    
    # -----------------------------
    # IV Management Variables
    # -----------------------------
    # For voucher IV management; these values should be persisted via trader_memory.
    last_vol = None
    coupon_ema = None         # EMA for the 10000 coupon only.
    non_coupon_ema = None      # EMA for all other vouchers.
    ema_iv = None              # Slow-moving EMA for the ATM voucher (VOLCANIC_ROCK_VOUCHER_10000)
    ema_iv_9750 = None         # Slow-moving EMA for the 9750 voucher (persist only when threshold is met)
    alpha_iv = 0.0055
    window = 0                 # For any time-based window logic
    ema_CSI = None
    sunlight_direction = 0
    last_sunlight = None

    statuses = {
    'CROISSANTS': status_croissant,
    'JAMS': status_jam,
    'DJEMBES': status_djembe,
    'PICNIC_BASKET1': status_basket1,
    'PICNIC_BASKET2': status_basket2,
    'VOLCANIC_ROCK': status_volcanic_rock,
    'VOLCANIC_ROCK_VOUCHER_9500': status_voucher_9500,
    'VOLCANIC_ROCK_VOUCHER_9750': status_voucher_9750,
    'VOLCANIC_ROCK_VOUCHER_10000': status_voucher_10000,
    'VOLCANIC_ROCK_VOUCHER_10250': status_voucher_10250,
    'VOLCANIC_ROCK_VOUCHER_10500': status_voucher_10500
}
    cross_strategy = CrossStrategy() 
    
    @staticmethod
    def update_iv_distribution(ema: float, current_iv: float):
        """
        Update the IV-based Beta distributions using the current slow EMA.
        (The fast IV, as computed for the ATM voucher, is passed in as current_iv.)
        """
        dynamic_buy_a = ema - 0.006
        dynamic_buy_b = dynamic_buy_a - 0.004  # You can tune this offset as needed

        dynamic_sell_a = ema + 0.006
        dynamic_sell_b = dynamic_sell_a + 0.004

        alphaL_buy, alphaR_buy = 4.0, 2.0  # Example parameters for buy-side
        # Rebuild the mirrored (buy) distribution using the current slow EMA:
        _, _, _, cdf_buy_at_iv = Trader.distribution_calc.build_mirrored_beta_distribution(
            dynamic_buy_a, dynamic_buy_b, alphaL_buy, alphaR_buy, steps=100
        )
        Trader.distribution_calc.cdf_buy_at_iv = cdf_buy_at_iv

        # Build the ascending (sell-side) distribution:
        alphaL_sell, alphaR_sell = 4.0, 2.0
        _, _, _, cdf_sell_at_iv = Trader.distribution_calc.build_ascending_beta_distribution(
            dynamic_sell_a, dynamic_sell_b, alphaL_sell, alphaR_sell, steps=100
        )
        Trader.distribution_calc.cdf_sell_at_iv = cdf_sell_at_iv
        
    def get_voucher_mid_price(self, voucher_depth: OrderDepth, traderData: Dict[str, Any]) -> float:
        if voucher_depth.buy_orders and voucher_depth.sell_orders:
            best_bid = max(voucher_depth.buy_orders.keys())
            best_ask = min(voucher_depth.sell_orders.keys())
            best_bid_vol = abs(voucher_depth.buy_orders[best_bid])
            best_ask_vol = abs(voucher_depth.sell_orders[best_ask])
            return (best_bid * best_ask_vol + best_ask * best_bid_vol) / (best_bid_vol + best_ask_vol)
        return traderData.get("prev_voucher_price", 0)

    def run(self, state: TradingState) -> tuple[dict[Symbol], list[Order], int, str]:
     
        Status.cls_update(state)

        # Deserialize traderData
        trader_memory = {}
        if state.traderData:
            try:
                trader_memory = jsonpickle.decode(state.traderData)
            except Exception as e:
                print(f"Failed to decode traderData: {e}")
                trader_memory = {}

        result: dict[Symbol, list[Order]] = {}
        macaron_orders: list[Order] = []  

        # Resin Strategy
        result['RAINFOREST_RESIN'] = Trade.resin(self.state_resin)


        trader_memory = self.state_squid.refresh_ema_deviation(trader_memory)

        trader_memory = self.state_squid.set_log_return_and_mid(trader_memory)
        # squid_orders = Trade.squid(self.state_squid, trader_memory)
        # if squid_orders:
        #     result["SQUID_INK"] = squid_orders
        # squid_trade_amt = Trade
        # # ----------- KELP STRATEGIES CHAIN -----------
 
        # # ✅ Update log return + mid price in traderMemory
        trader_memory = self.state_kelp.set_log_return_and_mid(trader_memory)

        result['KELP'] = Trade.kelp(self.state_kelp)


        alpha_cross = 0.0005
        premium_basket_1_spread = self.cross_strategy.compute_premium_spread_basket1(self.statuses)
        premium_basket_2_spread = self.cross_strategy.compute_premium_spread_basket2(self.statuses)

        ema_premium1 = trader_memory.get("ema_premium1", None)
        ema_premium2 = trader_memory.get("ema_premium2", None)
        if ema_premium1 is None:
            ema_premium1 = 0
            ema_premium2 = 0
        else:
            ema_premium1 = alpha_cross * premium_basket_1_spread + (1-alpha_cross)*ema_premium1
            ema_premium2 = alpha_cross * premium_basket_2_spread + (1-alpha_cross)*ema_premium2

        trader_memory["ema_premium1"] = ema_premium1
        trader_memory["ema_premium2"] = ema_premium2
        self.cross_strategy.precompute_distributions(ema_premium1, ema_premium2)

        # print(premium_basket_1_spread,         ema_premium1)

        cross_orders = self.cross_strategy.inter_basket_arbitrage(self.statuses, ema_premium1, ema_premium2)
        if cross_orders:
            for order in cross_orders:
                if order.symbol in result:
                    result[order.symbol].append(order)
                else:
                    result[order.symbol] = [order]
                    # print(result[order.symbol] )
        
        # -------------- ROUND 3 IMPLEMENTATION --------------
        voucher_orders, trader_memory = Trade.volcano(self, state, trader_memory)
        for sym, orders in voucher_orders.items():
            result[sym] = orders

        # -------------- ROUND 5 IMPLEMENTATION --------------

        status     = self.status_croissant
        product    = status.product
        limit      = status._position_limit[product]  # your maximum net position

        # 1) load (or init) our “where-I-want-to-be” flag from memory
        target_pos = trader_memory.get("croissant_target", 0)
        
        # print(f"[Croissant] start tick — target_pos={target_pos}")

        # 2) if we see a Caesar - Olivia cross, flip target to +-limit
        for t in state.market_trades.get(product, []):
            if t.buyer == t.seller:
                continue
            if t.seller == "Caesar" and t.buyer == "Olivia":
                target_pos = +limit   # want to be net long
                # print(f"[Croissant] saw Olivia→Caesar, switching target_pos to {target_pos}")

            elif t.seller == "Olivia" and t.buyer == "Caesar":
                target_pos = -limit   # want to be net short
                # print(f"[Croissant] saw Caesar→Olivia, switching target_pos to {target_pos}")


        # 3) persist that target
        trader_memory["croissant_target"] = target_pos

        # 4) each tick, if target ≠ 0, send your full allowed order in that direction
        #    once your fills bring current_pos up to target, possible_*_amt will go to zero
        if target_pos > 0:
            buy_qty = status.possible_buy_amt
            # print(f"[Croissant] sending BUY {buy_qty} @ {status.maxamt_askprc} to reach {target_pos}")
            if buy_qty > 0:
                result.setdefault(product, []).append(
                    Order(product, status.maxamt_askprc, buy_qty)
                )
        elif target_pos < 0:
            sell_qty = status.possible_sell_amt
            # print(f"[Croissant] sending SELL {sell_qty} @ {status.maxamt_bidprc} to reach {target_pos}")
            if sell_qty > 0:
                result.setdefault(product, []).append(
                    Order(product, status.maxamt_bidprc, -sell_qty)
                )
        # else:
        #     print(f"[Croissant] target_pos is 0, no order sent")

        # -------------- ROUND 5: PICNIC_BASKET2 (same Caesar↔Olivia flips) --------------

        # pull out the Status object for basket2  
        # basket2 = self.status_basket2  
        # product2 = basket2.product  
        # limit2   = basket2._position_limit[product2]  

        # # load (or init) our “where‑I‑want‑to‑be” for basket2  
        # target2 = trader_memory.get("basket2_target", 0)  
        # # print(f"[Basket2] start tick — target2={target2}")  

        # # scan Caesar↔Olivia crosses on basket2  
        # for t in state.market_trades.get(product, []):  
        #     if t.buyer == t.seller:  
        #         continue  

        #     # Caesar→Olivia ⇒ downtrend ⇒ go net SHORT  
        #     if t.seller == "Caesar" and t.buyer == "Olivia":
        #         if target2 < 0 :
        #             target2 = 1
        #         else:  
        #             target2 = +limit2  
        #         # print(f"[Basket2] saw Caesar→Olivia, switching target2 to {target2} @ time: {status.timestep}")  

        #     # Olivia→Caesar ⇒ uptrend ⇒ go net LONG  
        #     elif t.seller == "Olivia" and t.buyer == "Caesar": 
        #         if target2 > 0:
        #             target2 = -1
        #         else: 
        #             target2 = -limit2  
        #         # print(f"[Basket2] saw Olivia→Caesar, switching target2 to {target2} @ time: {status.timestep}")

        # # persist basket2 target  
        # trader_memory["basket2_target"] = target2  

        # # blast toward that target each tick  
        # if target2 > 0:  
        #     buy2 = basket2.possible_buy_amt  
        #     # print(f"[Basket2] sending BUY {buy2} @ {basket2.maxamt_askprc} to reach {target2}")  
        #     if buy2 > 0:  
        #         result.setdefault(product2, []).append(  
        #             Order(product2, basket2.maxamt_askprc, buy2)  
        #         )
        #         # print(result)  

        # elif target2 < 0:  
        #     sell2 = basket2.possible_sell_amt  
        #     # print(f"[Basket2] sending SELL {sell2} @ {basket2.maxamt_bidprc} to reach {target2}")  
        #     if sell2 > 0:  
        #         result.setdefault(product2, []).append(  
        #             Order(product2, basket2.maxamt_bidprc, -sell2)  
        #         )
        #         # print(result) 

        # else:  
        #     print(f"[Basket2] target2 is 0, no order sent") 

        # ------- ROUND 5 SQUID INK IMPLEMENTATION -------
        # pull out the Status object for squid ink
        status   = self.state_squid
        product  = status.product
        limit    = status._position_limit[product]       # your max net position for SQUID_INK

        # 1) load (or init) our “where-I-want-to-be” flag from memory
        target_pos = trader_memory.get("squid_target", 0)
        #print(f"[Squid] start tick — target_pos={target_pos}")

        # 2) scan this tick’s market_trades for Charlie↔Olivia crosses
        for t in state.market_trades.get(product, []):
            if t.buyer == t.seller:
                continue

            # Charlie→Olivia ⇒ price DOWN ⇒ go SHORT
            if t.seller == "Charlie" and t.buyer == "Olivia":
                target_pos = +limit
                #print(f"[Squid] saw Charlie→Olivia, switching target_pos to {target_pos}")

            # Olivia→Charlie ⇒ price UP ⇒ go LONG
            elif t.seller == "Olivia" and t.buyer == "Charlie":
                target_pos = -limit
                #print(f"[Squid] saw Olivia→Charlie, switching target_pos to {target_pos}")

        # 3) persist that target for next tick
        trader_memory["squid_target"] = target_pos

        # 4) each tick, drive your order all the way toward target_pos
        if target_pos > 0:
            buy_qty = status.possible_buy_amt
            #print(f"[Squid] sending BUY {buy_qty} @ {status.maxamt_askprc} to reach {target_pos}")
            if buy_qty > 0:
                result.setdefault(product, []).append(
                    Order(product, status.maxamt_askprc, buy_qty)
                )

        elif target_pos < 0:
            sell_qty = status.possible_sell_amt
            #print(f"[Squid] sending SELL {sell_qty} @ {status.maxamt_bidprc} to reach {target_pos}")
            if sell_qty > 0:
                result.setdefault(product, []).append(
                    Order(product, status.maxamt_bidprc, -sell_qty)
                )


        #-------------- ROUND 4 MACARON --------------
        importTariff = state.observations.conversionObservations['MAGNIFICENT_MACARONS'].importTariff
        exportTariff = state.observations.conversionObservations['MAGNIFICENT_MACARONS'].exportTariff
        bidPrice = state.observations.conversionObservations['MAGNIFICENT_MACARONS'].bidPrice
        askPrice = state.observations.conversionObservations['MAGNIFICENT_MACARONS'].askPrice
        sugarPrice = state.observations.conversionObservations['MAGNIFICENT_MACARONS'].sugarPrice
        sunlightIndex = state.observations.conversionObservations['MAGNIFICENT_MACARONS'].sunlightIndex
        transportFees = state.observations.conversionObservations['MAGNIFICENT_MACARONS'].transportFees

        macaron_pos = state.position.get("MAGNIFICENT_MACARONS", 0)

        macaron_depth = state.order_depths["MAGNIFICENT_MACARONS"]
        macaron_bid = max(macaron_depth.buy_orders.keys())
        macaron_bid_amt = macaron_depth.buy_orders[macaron_bid]
        macaron_ask = min(macaron_depth.sell_orders.keys())
        macaron_ask_amt = abs(macaron_depth.sell_orders[macaron_ask])

        max_macaron_orders_limit_buy = max(0, 75 - macaron_pos)
        max_macaron_orders_limit_sell = max(0, 75 + macaron_pos)

        conv_amnt = 0

        if "ema_CSI" in trader_memory:
            self.ema_CSI = trader_memory["ema_CSI"]
            self.sunlight_direction = trader_memory.get("sunlight_direction", 0.0)
            self.last_sunlight     = trader_memory.get("last_sunlight", None)
        else:
            self.ema_CSI            = None
            self.sunlight_direction = None
            self.last_sunlight      = None
        
        if self.last_sunlight is None:
            # start both EMA and last at the first observed value
            self.ema_CSI       = 49
            self.last_sunlight = sunlightIndex
            # direction has nowhere to go yet, stay at 0
            self.sunlight_direction = 0.0

        # 2) Subsequent updates
        else:
            # 2a) EMA update
            alpha = 0.00007
            self.ema_CSI = alpha * sunlightIndex + (1 - alpha) * self.ema_CSI

            # 2b) instantaneous “direction” as log-return
            new_dir = sunlightIndex - self.last_sunlight
            # if you really want to ignore zero‐change, you can:
            if new_dir != 0:
                self.sunlight_direction = new_dir

            # 2c) advance your “last” reading
            self.last_sunlight = sunlightIndex

        my_trades       = state.own_trades.get("MAGNIFICENT_MACARONS", [])
        avg_exec_price  = (sum(t.price for t in my_trades) / len(my_trades)) if my_trades else 0.0 

        convert_existing_short_position_prc = askPrice + importTariff + transportFees
        convert_existing_long_position_prc = bidPrice - exportTariff - transportFees - .3*min(10, macaron_pos)

        if macaron_bid_amt > 0 and macaron_ask_amt > 0:     # FIX #6
            top_lyr_skew = np.log(macaron_bid_amt / macaron_ask_amt)
        else:
            top_lyr_skew = 0.0

        # -------------- FEAR INDEX SCRIT --------------
        if sunlightIndex < self.ema_CSI and self.sunlight_direction < 0:
            print("fear phase starting")
            if macaron_pos < 0:
                
                conversion_possible_amnt = min(10, abs(macaron_pos))
                # because our current position is negative the conv amont will be a positive number
                conv_amnt = conversion_possible_amnt

                new_pos = macaron_pos + conv_amnt

                sell_capacity = max(0, 75 + new_pos)
                buy_capacity = max(0, 75 - new_pos)

                send_ord = min(buy_capacity, abs(macaron_ask_amt))
                macaron_orders.append(Order("MAGNIFICENT_MACARONS", int(macaron_ask), send_ord))
                remaining_amnt = buy_capacity - send_ord
                if remaining_amnt > 0:
                    if top_lyr_skew < -0.8:
                        macaron_orders.append(Order("MAGNIFICENT_MACARONS", int(macaron_bid)+1, remaining_amnt))
                    else:
                        macaron_orders.append(Order("MAGNIFICENT_MACARONS", int(macaron_bid)+2, remaining_amnt))

            else:
                if macaron_pos > 20:

                    if convert_existing_long_position_prc > avg_exec_price:
                        conversion_possible_amnt = min(10, abs(macaron_pos))
                        # because our current position is positive the conv amont will be a negative number
                        conv_amnt = -conversion_possible_amnt

                    else:
                        conv_amnt = 0
                    
                    new_pos = macaron_pos + conv_amnt

                    sell_capacity = max(0, 75 + new_pos)
                    buy_capacity = max(0, 75 - new_pos)

                    if top_lyr_skew > .8:
                        macaron_orders.append(Order("MAGNIFICENT_MACARONS", int(macaron_ask)-1, -new_pos))
                    else:
                        macaron_orders.append(Order("MAGNIFICENT_MACARONS", int(macaron_ask)-2, -new_pos))


                    if top_lyr_skew < -.8:
                        macaron_orders.append(Order("MAGNIFICENT_MACARONS", int(macaron_bid)+1, buy_capacity))
                    else:
                        macaron_orders.append(Order("MAGNIFICENT_MACARONS", int(macaron_bid)+2, buy_capacity))

                else:

                    conv_amnt = 0
                    
                    new_pos = macaron_pos + conv_amnt

                    sell_capacity = max(0, 75 + new_pos)
                    buy_capacity = max(0, 75 - new_pos)

                    if top_lyr_skew > .8:
                        macaron_orders.append(Order("MAGNIFICENT_MACARONS", int(macaron_ask)-1, -new_pos))
                    else:
                        macaron_orders.append(Order("MAGNIFICENT_MACARONS", int(macaron_ask)-2, -new_pos))


                    if top_lyr_skew < -.8:
                        macaron_orders.append(Order("MAGNIFICENT_MACARONS", int(macaron_bid)+1, buy_capacity))
                    else:
                        macaron_orders.append(Order("MAGNIFICENT_MACARONS", int(macaron_bid)+2, buy_capacity))
       
        elif sunlightIndex < self.ema_CSI and self.sunlight_direction > 0:

            print("fear phase ending")

            if macaron_pos > 0:

                conv_amnt = 0
                
                new_pos = macaron_pos + conv_amnt

                sell_capacity = max(0, 75 + new_pos)
                buy_capacity = max(0, 75 - new_pos)

                send_ord = min(macaron_bid_amt, sell_capacity)
                macaron_orders.append(Order("MAGNIFICENT_MACARONS", int(macaron_bid), -send_ord))

                remaining_amnt = sell_capacity - send_ord

                if top_lyr_skew > .8:
                    macaron_orders.append(Order("MAGNIFICENT_MACARONS", int(macaron_ask)-1, -remaining_amnt))
                else:
                    macaron_orders.append(Order("MAGNIFICENT_MACARONS", int(macaron_ask)-2, -remaining_amnt))

            else:

                if macaron_bid > convert_existing_short_position_prc:
                    send_ord = min(macaron_bid_amt, max_macaron_orders_limit_sell)
                    macaron_orders.append(Order("MAGNIFICENT_MACARONS", int(macaron_bid), -send_ord))

                    if macaron_pos < 0:
                        conversion_possible_amnt = min(10, abs(macaron_pos))
                        # because our current position is negative the conv amont will be a positive number
                        conv_amnt = conversion_possible_amnt
                    else:
                        conv_amnt = 0

                    new_pos = macaron_pos - send_ord + conv_amnt

                    remaining_sell_capacity = max(0, 75 + new_pos)

                    if top_lyr_skew > .8:
                        macaron_orders.append(Order("MAGNIFICENT_MACARONS", int(macaron_ask)-1, -remaining_sell_capacity))
                    else:
                        macaron_orders.append(Order("MAGNIFICENT_MACARONS", int(macaron_ask)-2, -remaining_sell_capacity))
                    
                    if new_pos < 0 :

                        if top_lyr_skew < -.8:
                            macaron_orders.append(Order("MAGNIFICENT_MACARONS", int(macaron_bid)+1, -new_pos))
                        else:
                            macaron_orders.append(Order("MAGNIFICENT_MACARONS", int(macaron_bid)+2, -new_pos)) 


                else:
                    if macaron_pos < 0 and convert_existing_short_position_prc < avg_exec_price:
                        conversion_possible_amnt = min(10, abs(macaron_pos))
                        # because our current position is negative the conv amont will be a positive number
                        conv_amnt = conversion_possible_amnt
                    else:
                        conv_amnt = 0

                    new_pos = macaron_pos + conv_amnt

                    sell_capacity = max(0, 75 + new_pos)
                    buy_capacity = max(0, 75 - new_pos)

                    if top_lyr_skew > .8:
                        macaron_orders.append(Order("MAGNIFICENT_MACARONS", int(macaron_ask)-1, -sell_capacity))
                    else:
                        macaron_orders.append(Order("MAGNIFICENT_MACARONS", int(macaron_ask)-2, -sell_capacity))
                    
                    if new_pos < 0 :

                        if top_lyr_skew < -.8:
                            macaron_orders.append(Order("MAGNIFICENT_MACARONS", int(macaron_bid)+1, -new_pos))
                        else:
                            macaron_orders.append(Order("MAGNIFICENT_MACARONS", int(macaron_bid)+2, -new_pos))


        else:
            
            if macaron_bid > convert_existing_short_position_prc:
                send_ord = min(macaron_bid_amt, max_macaron_orders_limit_sell)
                macaron_orders.append(Order("MAGNIFICENT_MACARONS", int(macaron_bid), -send_ord))

                if macaron_pos < 0:
                    conversion_possible_amnt = min(10, abs(macaron_pos))
                    # because our current position is negative the conv amont will be a positive number
                    conv_amnt = conversion_possible_amnt
                else:
                    conv_amnt = 0

                new_pos = macaron_pos - send_ord + conv_amnt

                remaining_sell_capacity = max(0, 75 + new_pos)

                if top_lyr_skew > .8:
                    macaron_orders.append(Order("MAGNIFICENT_MACARONS", int(macaron_ask)-1, -remaining_sell_capacity))
                else:
                    macaron_orders.append(Order("MAGNIFICENT_MACARONS", int(macaron_ask)-2, -remaining_sell_capacity))
                
                if new_pos < 0 :

                    if top_lyr_skew < -.8:
                        macaron_orders.append(Order("MAGNIFICENT_MACARONS", int(macaron_bid)+1, -new_pos))
                    else:
                        macaron_orders.append(Order("MAGNIFICENT_MACARONS", int(macaron_bid)+2, -new_pos)) 


            else:
                if macaron_pos < 0 and convert_existing_short_position_prc < avg_exec_price:
                    conversion_possible_amnt = min(10, abs(macaron_pos))
                    # because our current position is negative the conv amont will be a positive number
                    conv_amnt = conversion_possible_amnt
                else:
                    conv_amnt = 0

                new_pos = macaron_pos + conv_amnt

                sell_capacity = max(0, 75 + new_pos)
                buy_capacity = max(0, 75 - new_pos)

                if top_lyr_skew > .8:
                    macaron_orders.append(Order("MAGNIFICENT_MACARONS", int(macaron_ask)-1, -sell_capacity))
                else:
                    macaron_orders.append(Order("MAGNIFICENT_MACARONS", int(macaron_ask)-2, -sell_capacity))
                
                if new_pos < 0 :

                    if top_lyr_skew < -.8:
                        macaron_orders.append(Order("MAGNIFICENT_MACARONS", int(macaron_bid)+1, -new_pos))
                    else:
                        macaron_orders.append(Order("MAGNIFICENT_MACARONS", int(macaron_bid)+2, -new_pos))

        if macaron_orders:
            result["MAGNIFICENT_MACARONS"] = macaron_orders

        # print(f"ema_CSI: {self.ema_CSI}")
        # print(f"sunlight_direction: {self.sunlight_direction}")
        # print(f"last_sunlight: {self.last_sunlight}")

        trader_memory["ema_CSI"] = self.ema_CSI
        trader_memory["sunlight_direction"] = self.sunlight_direction        
        trader_memory["last_sunlight"] = self.last_sunlight
        # print(result)
        traderData = jsonpickle.encode(trader_memory)
        return result, conv_amnt, traderData
    