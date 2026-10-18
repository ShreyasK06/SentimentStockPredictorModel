# Phase 8 WP1: FNSPID timestamp audit

Share of articles stamped exactly 00:00:00: **99.8%** (date-only stamps).

Hour-of-day histogram of non-midnight stamps (as written, labelled 'UTC'):

| index | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| articles | 1 | 10 | 16 | 20 | 32 | 29 | 57 | 33 | 24 | 20 | 13 | 12 | 9 | 11 | 6 | 3 | 2 | 6 | 9 | 5 | 19 | 8 | 10 | 13 |

Share where naive (trust-the-clock) effective day equals the conservative one: 7.1%.

Conclusion: clock times are unreliable (most rows are date-only, and the planning probe found intraday-specific Benzinga headlines stamped at impossible ET times), so headline numbers use the frozen conservative rule: article stamped on date D -> effective trading day = first NYSE session strictly after D; label = sign(Close[E+1]/Close[E]-1).