# imc-prosperity3-trading
![GitHub Downloads (all assets, all releases)](https://img.shields.io/github/downloads/awatatani/imc-prosperity3-trading/total)

> Algorithmic and manual trading strategies, backtests, and analysis for IMC’s 15‑day Prosperity 3 challenge that resulted in rank 25.

### What is **Prosperity**?  
Prosperity is a **15‑day, 5‑round trading competition** where you build a Python algorithm to buy and sell island‑themed products and earn as many **SeaShells** as possible.  

* Each round rolls out new products and a one‑off manual‑trading puzzle.  
* Your python algorithm must handle positions, price dynamics, and conversions to maximize profit.  
* Your overall ranking is the total SeaShells you’ve accumulated by the end of day 15.

---

### Who is **IMC**?  
> Founded in **1989** on the floor of the Amsterdam Equity Options Exchange, **IMC** quickly recognized that technology could transform manual market‑making.  

* Over 35 years they’ve grown into a global trading firm where **data‑driven algorithms** and **cutting‑edge execution platforms** meet deep trading expertise.  
* IMC’s entrepreneurial spirit and relentless innovation—fueled by significant investment in tools and talent—enable them to provide liquidity and shape the future of financial markets.

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

### Algorithmic Trading  
You have **72 hours per round** to upload your Python algorithm. The **last successfully processed submission** before the deadline is used for that round’s trading session.

### Manual Trading  
In parallel, you also have **72 hours** to place **one manual trade** each round.  
Manual and algorithmic challenges are independent, each giving you separate profit opportunities.
