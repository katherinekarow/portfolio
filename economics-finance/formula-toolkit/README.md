# Finance & demand-analytics formula toolkit
> 40 finance, economics and demand-analytics formulas, including NPV/IRR, amortisation, elasticities, curve normalisation, gross vs net targets and credibility weighting, as a tested Python package with an in-browser calculator.

**Tools:** Python, pytest, JavaScript, Playwright

**Live demo:** [Open the formula calculator](https://katherinekarow.github.io/portfolio/formulas/)

*This started as my first economics calculator (kept in [`original/`](original/econ_calculator.py)). It now also holds
the formulas I use at work to pace demand, set targets and check data. The examples are generic numbers, not work data.*

## What's inside
| Group | Formulas | Where they come up |
|---|---|---|
| **Time value of money** | future/present value, CAGR, NPV, IRR, payback, loan payment, amortisation schedule, effective annual rate | evaluating a campaign or project; loan books ([loan vintages](../loan-vintage-curves/), [marketing ROI](../marketing-roi/)) |
| **Money & interest rates** | money multiplier, reserve ratio, real and nominal rates (exact Fisher and the shortcut) | macroeconomics coursework |
| **Demand & price elasticity** | arc (midpoint) and point elasticity, log-log elasticity fit, demand at a new price, revenue effect of a price change, markup-rule optimal price, price that fills capacity, cross-price elasticity | pricing work: how price sensitivity is measured and then used ([produce pricing](../../professional-work/produce-pricing/)) |
| **Demand curves & pacing** | fill rate, position vs benchmark (normalisation), capacity-weighted curve blend, benchmark percentiles (P25/P50/P75), monotone repair, demand velocity, pace projection, residual-demand allocation | building target and pacing curves ([capacity planning](../../professional-work/capacity-planning/)) |
| **Gross vs net targets** | survival rate, gross target from a net target, expected net from gross, gross still needed | turning a net goal into what must be taken when some of it will cancel |
| **Benchmarking, credibility & data checks** | credibility weight and blend, Bühlmann k from data, shrinking small-sample rates, change-tolerance gate, reconciliation match, WAPE | trusting an item's own data vs its group; stopping a bad table from being published |

The full list, each with its formula, explanation and a computed example, is in **[FORMULAS.md](FORMULAS.md)**,
generated from the code.

## How elasticities are used, step by step
1. **Measure it.** From two price points, use the arc elasticity. From a history of prices and volumes, fit `ln Q = a + e·ln P`
   (`fit_elasticity`). The slope is the elasticity, and it is the same at every price.
2. **Predict volume.** `demand_at_price`: Q₁ = Q₀ × (P₁/P₀)^e. For example, e = −1.3 and a 10% price rise gives about −11.7% volume.
3. **Predict revenue.** `revenue_change`: (1 + Δp)^(1+e) − 1. The rise lifts revenue only when |e| < 1. At e = −1.3 the same rise *loses* 2.8% of revenue.
4. **Set the price.**
   - With a unit cost, use the markup rule P* = c·e/(1+e): at e = −2.5 that is a 67% markup.
   - With fixed capacity and almost no marginal cost, use the price where demand just fills capacity (`capacity_price`).
5. **Check for cannibalisation.** Use cross-price elasticity, so a price move on one grade isn't just shifting buyers to another.

## How a target curve is built, step by step
1. **Normalise.** Divide units by capacity (`fill_rate`) so items of different size share one scale.
2. **Benchmark.** Take percentiles of comparable completed items at each lead time (`percentile`), choosing P50 or P60 as the
   target and P25/P75 as the conservative/aggressive band. Then force the curve to be non-decreasing (`monotone_curve`).
3. **Blend components.** Weight the curves of parts that sell differently by their capacity share (`blend_curves`).
4. **Gross vs net.** The target is set on what will *remain*, so divide by survival to get what must be *taken*
   (`gross_target`, `gross_still_needed`).
5. **Project and close the gap.**
   - Project today's position with `pace_projection`, weighted against the benchmark by credibility (`credibility_weight`, `credibility_blend`).
   - Spread any shortfall over the remaining days in the benchmark's shape (`residual_allocation`). The parts always sum exactly to the gap.
6. **Gate before publishing.** If totals move more than the tolerance compared with the last run (`variance_gate`), stop and look.

## Run it
```bash
cd economics-finance/formula-toolkit
pip install -r requirements.txt
PYTHONPATH=src python -m formulakit list
PYTHONPATH=src python -m formulakit npv --rate 0.08 --flows "-1000,300,400,500,200"
PYTHONPATH=src python -m formulakit residual_allocation --gap 300
PYTHONPATH=src python -m formulakit          # interactive, like the original calculator
PYTHONPATH=src python -m formulakit docs --out FORMULAS.md
PYTHONPATH=src python -m formulakit export --out ../../docs/formulas/formulas.json
pytest -q                                    # 80 tests
```
```
Net present value:  NPV = Σ CFₜ ÷ (1 + r)^t,  t = 0, 1, 2 …
164.64
```

## What the tests check
| Rule | How |
|---|---|
| Finance formulas match hand calculations; IRR sets NPV to zero; principal in the amortisation schedule sums to the loan | `test_formulas.py` |
| Elasticity fit recovers a known elasticity exactly (no noise) and closely (with noise), and matches NumPy's regression | `test_fit_elasticity_recovers_truth` |
| Markup-rule price beats every price on a 20,000-point grid and satisfies the Lerner condition | `test_optimal_price_beats_grid` |
| Percentiles equal NumPy's; residual allocation sums exactly to the gap and is never more than one unit off its exact share | `test_percentile_*`, `test_residual_*` |
| Gross → net round-trips; Bühlmann k is recovered from simulated groups (true k = 9) | `test_gross_net_round_trip`, `test_buhlmann_recovers_k` |
| FORMULAS.md is up to date with the code | `test_formulas_md_is_current` |
| The browser calculator returns the same answer as Python for every formula on three sets of inputs | `test_page.py` |
