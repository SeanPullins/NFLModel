# Free signal scouting plan

This is the no-spend R&D queue for improving APEX while the current validation reports run.

## Operating rule

No new feature becomes part of the public score until it passes:

1. residual value after controlling for draft slot/log pick;
2. era stability across historical windows;
3. sufficient sample size by position;
4. rolling out-of-time validation;
5. no unacceptable worst-year drawdown.

## Agent deployment

| Priority | Agent | Signal family | First build |
|---:|---|---|---|
| 1 | Market Disagreement Agent | consensus drift and dispersion | expand consensus_board with source count, mock std, early-vs-late movement |
| 2 | CFBD PPA Agent | player PPA efficiency | build CFBD PPA career/final/peak features |
| 3 | CFBD Usage Agent | role and usage share | add player usage share and role stability by season |
| 4 | Recruiting Prior Agent | pedigree and talent context | join recruit ratings and team talent context |
| 5 | SackSEER-Style EDGE Agent | EDGE explosiveness + production proxy | combine sack/TFL productivity with vertical/shuttle/availability |
| 6 | Schedule Context Agent | opponent-adjusted production | bucket production by opponent/team strength |
| 7 | Experience Curve Agent | age, experience, early declare | standardize age/season/start proxies by position |
| 8 | Matriculation Agent | roster/snap floor | separate floor model from star model |

## Why this order

Draft slot and public consensus are still the market to beat, so the biggest free edge is disagreement and uncertainty around that market. CFBD PPA and usage are next because they may capture efficiency and role better than raw yards. Recruiting/talent context can act as a prior. EDGE gets its own sprint because historical edge models have shown that sack production needs athletic and availability context, not just sacks.

## Commands

```bash
python src/free_signal_backlog.py
```

Future builders should write optional files under `data/production/` or `data/consensus/`, then use `src/build_features.py` and a rolling backtest before promotion.
