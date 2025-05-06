# IMC's Prosperity 3 Trading Competition

![Team Maurya's Banner](images/Team_Maurya_v1.png)

![GitHub Downloads (all assets, all releases)](https://img.shields.io/github/downloads/awatatani/imc-prosperity3-trading/total)
![GitHub top language](https://img.shields.io/github/languages/top/awatatani/imc-prosperity3-trading)
![GitHub Repo stars](https://img.shields.io/github/stars/awatatani/imc-prosperity3-trading)

Algorithmic and manual trading strategies, backtests, and analysis for IMC’s 15‑day Prosperity 3 challenge that resulted in `Top 25` in the World, and `Top 5` in the United States.

⭐ **If you find this project helpful, please consider leaving us a star!**

## Team Members

<table>
  <tr>
    <td align="center">
      <img src="images/Anish_Ranjan_Headshot.jpg" width="180"><br>
      <strong>Anish&nbsp;Ranjan</strong><br>
      <a href="https://www.linkedin.com/in/anishranjan28/">LinkedIn</a> ·
      <a href="mailto:anishranjan07@gmail.com">Email</a>
    </td>
    <td align="center">
      <img src="images/Allan_Watatani_Headshot.jpg" width="180"><br>
      <strong>Allan&nbsp;Watatani</strong><br>
      <a href="https://www.linkedin.com/in/allan-watatani-9575a4202/">LinkedIn</a> ·
      <a href="mailto:awatatani6402@gmail.com">Email</a>
    </td>
  </tr>
</table>

---

### What is **Prosperity**?  
> Prosperity is a **15‑day, 5‑round trading competition** where you build a Python algorithm to buy and sell island‑themed products and earn as many **SeaShells** as possible.  

* Each round rolls out new products and a one‑off manual‑trading puzzle.  
* Your python algorithm must handle positions, price dynamics, and conversions to maximize profit.  
* Your overall ranking is the total SeaShells you’ve accumulated by the end of day 15.

---

## Game Mechanics

| Phase | Details |
|-------|---------|
| **Simulation length** | 15 days, split into **five 72‑hour rounds** |
| **Before the timer ends** | Submit your **algorithmic bot** *and* a **manual trade**. |
| **After the deadline** | Submissions are locked in and run against Prosperity bots. |
| **Results** | When the round closes, results are revealed and the leaderboard updates. Submissions can’t be changed for closed rounds. |
| **Final standings** | After Round 5, total SeaShells determine the overall winner. |

---

## Algorithmic Trading Strategies
You have **72 hours per round** to upload your Python algorithm. The **last successfully processed submission** before the deadline is used for that round’s trading session.

---

### Round 1

* **Products introduced**  
  * `RAINFOREST_RESIN` – historically stable  
  * `KELP` – regular ups & downs  
  * `SQUID_INK` – large, fast swings; rumored price pattern  

* **Position limits**
  | Product | Limit |
  |---------|------:|
  | RAINFOREST_RESIN | 50 |
  | KELP | 50 |
  | SQUID_INK | 50 |

* **Key hint**  
  * Squid Ink’s volatility makes large open positions risky.  
  * Price spikes tend to **mean-revert**—track the deviation from a recent average and fade extreme moves for edge.

* **Further reading:**  
  📘 See our full Round&nbsp;1 write-up&nbsp;→&nbsp;[detailed notebook](Round_1/round1_strats&analysis.ipynb)

---

### Round 2

* **Products added**  
  * **Composite baskets**  
    * `PICNIC_BASKET1` = 6 × `CROISSANTS` + 3 × `JAMS` + 1 × `DJEMBES`  
    * `PICNIC_BASKET2` = 4 × `CROISSANTS` + 2 × `JAMS`  
  * **Individual legs** now trade on their own orderbooks: `CROISSANTS`, `JAMS`, `DJEMBES`

* **Position limits**
  | Product | Limit |
  |---------|------:|
  | CROISSANTS | 250 |
  | JAMS | 350 |
  | DJEMBES | 60 |
  | PICNIC_BASKET1 | 60 |
  | PICNIC_BASKET2 | 100 |

* **Further reading:**  
  📘 See our full Round&nbsp;2 write-up&nbsp;→&nbsp;[detailed notebook](Round_2/round2_strats&analysis.ipynb)

---

### Round 3  

* **Products introduced**

  * `VOLCANIC_ROCK` – the physical underlying  
  * `VOLCANIC_ROCK_VOUCHER_XXXX` (tradable options on the rock, all expiring in *7 trading days* at the start of Round 1):  
    | Voucher | Strike (SeaShells) |
    |---------|--------------------:|
    | VOLCANIC_ROCK_VOUCHER_9500  | 9 500
    | VOLCANIC_ROCK_VOUCHER_9750  | 9 750
    | VOLCANIC_ROCK_VOUCHER_10000 | 10 000
    | VOLCANIC_ROCK_VOUCHER_10250 | 10 250
    | VOLCANIC_ROCK_VOUCHER_10500 | 10 500

* **Position limits**

  | Product/Voucher | Limit |
  |---------|------:|
  | VOLCANIC_ROCK | 400 |
  | VOLCANIC_ROCK_VOUCHER_9500 | 200 |
  | VOLCANIC_ROCK_VOUCHER_9750 | 200 |
  | VOLCANIC_ROCK_VOUCHER_10000 | 200 |
  | VOLCANIC_ROCK_VOUCHER_10250 | 200 |
  | VOLCANIC_ROCK_VOUCHER_10500 | 200 |

* **Key hint**  
  * Estimate implied volatility (v_t) at each timestamp using Black-Scholes, plot it against moneyness:

  ```math
  m_t = \frac{\ln\!\bigl(K / S_t\bigr)}{\sqrt{\text{TTE}}}
  ```
  * Then, fit a parabola to filter noise, and watch the time-series of the fitted at-the-money IV for trading signals across strikes.

---

### Round 4

* **Product introduced**  
  * `MAGNIFICENT MACARONS` – can be bought or sold **only through conversion** with *Pristine Cuisine* at their posted bid / ask prices.

* **Position limits**
  | Product | Limit |
  |---------|------:|
  | MACARONS | 75 |

* **Other limits & micro-fees**   
  * **Conversion limit:** 10 units per request  
  * **Storage fee:** 0.1 SeaShells *per timestamp* on **net-long** macarons (no cost when short)  
  * Each conversion pays **transport fees** plus an **import/export tariff** on top of the quoted price.

* **Key hint**  
  * There exists a **Critical Sunlight Index (CSI)**.  
    * **Sunlight < CSI:** panic over tight sugar & macaron supply → prices can spike far above fair value.  
    * **Sunlight ≥ CSI:** both markets drift around fair value and react to normal supply-demand flows.  
  * Detect when the sunlight index crosses CSI and position accordingly to capture the premium/discount in macaron prices.

---

### Round 5

* **Product introduced**
  * *None* - the tradable set is identical to Round 4.

* **Position limits**
  * Unchanged from the previous round for every product.

* **Kye hint**
  * Your own fills now expose a `counter_party` field.

---

## Manual Trading Strategies 
In parallel, you also have **72 hours** to place **one manual trade** each round.  
Manual and algorithmic challenges are independent, each giving you separate profit opportunities.

---

### Round 1

* **Objective**  
  * Begin with **SeaShells**, convert through up to **5 hops** of foreign island currencies, and end back in SeaShells with more than you started.  

* **Inputs you receive**  
  * A **4 × 4 exchange-rate matrix** `R` where  
    `R[i][j]` = units of currency *j* you receive for **1** unit of currency *i*.  
  * All rates are quoted **net of fees**, so the matrix is your entire “price sheet” for the round.

* **Key hint**  
  * Store paths as lists of indices, use `numpy.prod` on the corresponding `R` entries, and keep the arg-max. The entire search runs in milliseconds.

---

### Round 2

* **Scenario**  
  Shipwrecked **containers** wash ashore, and you may open **up to 2** of them to claim a share of the treasure inside.

* **Costs**
  * **1st container:** free
  * **2nd container:** pay an opening fee of 50,000 (deducted from your share).

* **Key mechanics**   
  * Each container has two parameters:  
    * **Treasure multiplier** (≤ 90) – scales the base loot of 10,000.  
    * **Inhabitants** (≤ 10) – the number of islanders expected to pick that same container.
      * Note: This doesn't include the actual competitors.

* **Payout mechanics**  
  1. **Raw treasure** for a container 

     ```math
       \text{Treasure}=10{,}000 \times \text{multiplier}
     ```

  2. **Split** that treasure by

     ```math
       \text{divisor}=N_{\text{inhabitants}} + \bigl(\%\text{-times-chosen}\bigr)
     ```  

     *Example – 5 people choose a container that was picked on 10 % of all opens → divisor = 5 + 10 = 15.*

  3. Subtract any **opening cost** on paid containers; what remains is your profit.

  ```math
   \text{profit} \;=\; \frac{\text{Treasure}}{\text{divisor}} \;-\; \text{opening fee}
   ```

* **Key hint**
  * A high multiplier is attractive **only** if few others open the same box.
  * Popular containers dilute returns; obscure picks are lucrative.
  * Weigh the second-container fee against the marginal treasure you expect to gain.
 
* **Goal**  
  * Select one or two containers to **maximize net SeaShells** after splits and fees.

---

### Round 3

* **Scenario**  
  * **Sea Turtles** are ready to sell **FLIPPERS**.  
  * You may submit **two price bids**; each turtle decides independently whether to accept.

* **Bid #1 – “Reserve-price” auction**  
  * Each turtle carries a hidden reserve price drawn **uniformly** from  
    * **160 – 200** or **250 – 320** (nothing falls between 200 and 250).  
  * A turtle sells you one flipper **iff** your first bid is **≥ its reserve price**.

* **Bid #2 – “Average-aware” auction**  
  * Turtles compare all traders’ second bids.  
  * If **your bid ≥ the global average**, turtles use the same rule as Bid #1.  
  * If **your bid < average**, any profit from those trades is **down-scaled** by

    ```math
      p \;=\;\left(\frac{320 - \text{avgBid}}{320 - \text{yourBid}}\right)^{\!3}
    ```

* **Exit price**  
  * Every flipper you acquire can be **resold at 320 SeaShells** each at round end.

* **Goal**  
  * Choose two bid prices that balance fill-probability vs. margin, factoring in the unknown average second-bid.

---

### Round 4

* **Scenario**  
  * A new **game-show** lets every participant open **up to 3 suitcases** full of SeaShells.  
  * Suitcases differ by a hidden **prize multiplier** and will be **shared** with anyone else who picks the same case.

* **Costs**  
  * **1st suitcase:** free  
  * **2nd & 3rd suitcases:** pay an opening fee of 50,000 and 100,000 (deducted from your share).

* **Key mechanics**   
  * Each suitcase has two parameters:  
    * **Treasure multiplier** (≤ 90) – scales the base loot of 10,000.  
    * **Inhabitants** (≤ 10) – the number of islanders expected to pick that same suitcase.
      * Note: This doesn't include the actual competitors.

* **Payout mechanics**  
  1. **Raw treasure** for a suitcase 

     ```math
       \text{Treasure}=10{,}000 \times \text{multiplier}
     ```

  2. **Split** that treasure by

     ```math
       \text{divisor}=N_{\text{inhabitants}} + \bigl(\%\text{-times-chosen}\bigr)
     ```  

     *Example – 5 people choose a case that was picked on 10 % of all opens → divisor = 5 + 10 = 15.*

  3. Subtract any **opening cost** on paid suitcases; what remains is your profit.

  ```math
   \text{profit} \;=\; \frac{\text{Treasure}}{\text{divisor}} \;-\; \text{opening fee}
   ```

* **Key hint**  
  * Round 2 statistics reveal how frequently each container was picked—use that prior to estimate how crowded a suitcase might be.  
  * With three picks available, balance:  
    * **High multiplier & low crowding** (larger raw share)  
    * **Opening fees** on 2nd/3rd choices

* **Goal**  
  * Select one, two or three suitcases to **maximize net SeaShells** after splits and fees.

---

### Round 5

* **Scenario**  
  * A **one-day invitational** on the West Archipelago Exchange.  
  * Benny the Bull grants access to the **Goldberg news feed** for last-minute market intel.  

* **Costs & mechanics**  
  * **Initial capital:** 1 000 000 SeaShells.  
  * For each foreign good, specify the **% of capital to buy or sell**.  
  * **Sliding fees:** the **more volume you trade in a product, the higher its fee** band becomes.  

* **Goal**  
  * Use Goldberg’s hints to target the best risk-adjusted goods.  
  * Balance position size against escalating fees to **finish the day with the largest net SeaShell stack**.

---
