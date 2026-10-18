# Phase 8 WP1: FNSPID coverage (28 fixed tickers)

FNSPID stamped dates: 2009-06-22 .. 2023-12-16. Dedupe: {'input': 213430, 'url_duplicates_removed': 46365, 'headline_duplicates_removed': 4361, 'output': 162704, 'raw_28_tickers': 213430}

Chosen dev_start: **2012-01-01** (rule: dev_start = 2012 unless a year >= 2012 has < 20% ticker-days with news; then the year after the last such year before 2022). Test starts 2022-01-01.

## Per year (effective trading day)

| year | articles | ticker_days_with_news | ticker_trading_days | pct_ticker_days_with_news |
|---|---|---|---|---|
| 2009 | 178 | 122 | 6552 | 1.9 |
| 2010 | 1902 | 881 | 6552 | 13.4 |
| 2011 | 2545 | 1154 | 6552 | 17.6 |
| 2012 | 3512 | 1522 | 6655 | 22.9 |
| 2013 | 3535 | 1527 | 7056 | 21.6 |
| 2014 | 5333 | 2032 | 7056 | 28.8 |
| 2015 | 6430 | 2401 | 7056 | 34.0 |
| 2016 | 10552 | 3061 | 7056 | 43.4 |
| 2017 | 11354 | 3253 | 7028 | 46.3 |
| 2018 | 13141 | 3734 | 7028 | 53.1 |
| 2019 | 17468 | 4494 | 7056 | 63.7 |
| 2020 | 19429 | 4060 | 7084 | 57.3 |
| 2021 | 11714 | 3002 | 7056 | 42.5 |
| 2022 | 21820 | 3670 | 7028 | 52.2 |
| 2023 | 33791 | 3926 | 6748 | 58.2 |

## Articles per ticker (all years)

| index | WFC | GOOGL | MRK | KO | MS | AAPL | CVX | MSFT | GS | WMT | CAT | SLB | XOM | COP | ABBV | BA | AMZN | EOG | HD | JNJ | JPM | MCD | PFE | BAC | PG | META | UNH |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| articles | 11072 | 10646 | 10568 | 10322 | 9124 | 9024 | 8416 | 8411 | 8384 | 8383 | 7375 | 7078 | 7061 | 6736 | 6356 | 5840 | 4874 | 4871 | 4414 | 2913 | 2844 | 2231 | 1817 | 1807 | 1310 | 428 | 399 |

## Source file mix per year

| year | fnspid | fnspid_nasdaq |
|---|---|---|
| 2009 | 158 | 20 |
| 2010 | 1583 | 319 |
| 2011 | 1992 | 553 |
| 2012 | 2448 | 1064 |
| 2013 | 2322 | 1213 |
| 2014 | 2987 | 2346 |
| 2015 | 3094 | 3336 |
| 2016 | 4675 | 5877 |
| 2017 | 3986 | 7368 |
| 2018 | 5684 | 7457 |
| 2019 | 8604 | 8864 |
| 2020 | 6433 | 12996 |
| 2021 | 0 | 11714 |
| 2022 | 0 | 21820 |
| 2023 | 0 | 33791 |

## Top publishers

| publisher | count |
|---|---|
| (none) | 118738 |
| Paul Quintaro | 4355 |
| Benzinga Newsdesk | 4165 |
| Seeking Alpha | 3602 |
| Zacks | 2485 |
| Lisa Levin | 2430 |
| GuruFocus | 2126 |
| Charles Gross | 2089 |
| Monica Gerson | 1501 |
| Jayson Derrick | 1142 |

## Headlines (normalized) appearing in more than one period (train<2021, val 2021, test>=2022)

899 distinct normalized headlines span >1 period. Top repeated:

| norm | rows |
|---|---|
| 5 dividend aristocrats where analysts see capital gains | 64 |
| 5 dividend growth stocks with upside to analyst targets | 64 |
| should you invest in the fidelity msci energy index etf feny | 48 |
| dow analyst moves ko | 42 |
| dow analyst moves mrk | 40 |
| 3 things to watch in the stock market this week | 37 |
| should you invest in the ishares us oil gas exploration production etf ieo | 36 |
| dow analyst moves cat | 35 |
| dow analyst moves gs | 30 |
| ishares core sp 500 etf experiences big inflow | 30 |
| dow analyst moves cvx | 30 |
| is schwab fundamental us broad market index etf fndb a strong etf right now | 24 |
| dow movers ba mrk | 24 |
| the energy select sector spdr fund experiences big inflow | 23 |
| should you invest in the fidelity msci consumer discretionary index etf fdis | 22 |
