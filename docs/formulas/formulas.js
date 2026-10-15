// JavaScript ports of economics-finance/formula-toolkit/src/formulakit/formulas.py.
// Same names, same algorithms; tests/test_page.py checks every one against Python.
const FK = (() => {
  const sum = a => a.reduce((x, y) => x + y, 0);
  const f = {};
  // finance
  f.future_value = ({ pv, rate, n }) => pv * (1 + rate) ** n;
  f.present_value = ({ fv, rate, n }) => fv / (1 + rate) ** n;
  f.growth_rate = ({ start, end, n }) => (end / start) ** (1 / n) - 1;
  f.npv = ({ rate, flows }) => sum(flows.map((cf, t) => cf / (1 + rate) ** t));
  f.irr = ({ flows }) => {
    let lo = -0.99, hi = 10, flo = f.npv({ rate: lo, flows }), fhi = f.npv({ rate: hi, flows });
    if (flo * fhi > 0) return null;
    for (let i = 0; i < 200; i++) {
      const mid = (lo + hi) / 2, fm = f.npv({ rate: mid, flows });
      if (flo * fm <= 0) { hi = mid; fhi = fm; } else { lo = mid; flo = fm; }
    }
    return (lo + hi) / 2;
  };
  f.payback = ({ flows }) => {
    let cum = 0;
    for (let t = 0; t < flows.length; t++) {
      const cf = flows[t];
      if (cum + cf >= 0 && t > 0) return t - 1 + (cf ? -cum / cf : 0);
      cum += cf;
    }
    return null;
  };
  f.loan_payment = ({ principal, rate, n }) => rate === 0 ? principal / n : principal * rate / (1 - (1 + rate) ** -n);
  f.amortization = ({ principal, rate, n }) => {
    const pmt = f.loan_payment({ principal, rate, n }); let bal = principal; const rows = [];
    for (let t = 1; t <= Math.trunc(n); t++) {
      const interest = bal * rate, princ = pmt - interest; bal -= princ;
      rows.push({ period: t, payment: pmt, interest, principal: princ, balance: Math.max(bal, 0) });
    }
    return rows;
  };
  f.effective_rate = ({ apr, m }) => (1 + apr / m) ** m - 1;
  // macro
  f.money_multiplier = ({ reserve_ratio }) => 1 / reserve_ratio;
  f.reserve_ratio = ({ multiplier }) => 1 / multiplier;
  f.real_rate = ({ nominal, inflation }) => (1 + nominal) / (1 + inflation) - 1;
  f.nominal_rate = ({ real, inflation }) => (1 + real) * (1 + inflation) - 1;
  // demand
  f.arc_elasticity = ({ p1, q1, p2, q2 }) => ((q2 - q1) / ((q1 + q2) / 2)) / ((p2 - p1) / ((p1 + p2) / 2));
  f.point_elasticity = ({ slope, p, q }) => slope * p / q;
  f.fit_elasticity = ({ prices, quantities }) => {
    const x = prices.map(Math.log), y = quantities.map(Math.log), mx = sum(x) / x.length, my = sum(y) / y.length;
    let sxy = 0, sxx = 0; x.forEach((a, i) => { sxy += (a - mx) * (y[i] - my); sxx += (a - mx) ** 2; });
    return sxy / sxx;
  };
  f.demand_at_price = ({ q0, p0, p1, e }) => q0 * (p1 / p0) ** e;
  f.revenue_change = ({ price_change, e }) => (1 + price_change) ** (1 + e) - 1;
  f.optimal_price = ({ cost, e }) => e >= -1 ? null : cost * e / (1 + e);
  f.capacity_price = ({ q0, p0, capacity, e }) => p0 * (capacity / q0) ** (1 / e);
  f.cross_elasticity = ({ qa1, qa2, pb1, pb2 }) => f.arc_elasticity({ p1: pb1, q1: qa1, p2: pb2, q2: qa2 });
  // curves
  f.fill_rate = ({ units, capacity }) => units / capacity;
  f.normalized_position = ({ fill_today, benchmark }) => fill_today / benchmark;
  f.blend_curves = ({ curves, capacities }) => { const tot = sum(capacities); return sum(curves.map((c, i) => c * capacities[i] / tot)); };
  f.percentile = ({ values, q }) => {
    const v = [...values].sort((a, b) => a - b), pos = q * (v.length - 1), lo = Math.floor(pos), hi = Math.min(lo + 1, v.length - 1);
    return v[lo] + (v[hi] - v[lo]) * (pos - lo);
  };
  f.monotone_curve = ({ curve }) => { let m = -Infinity; return curve.map(v => (m = Math.max(m, v))); };
  f.velocity = ({ cumulative }) => { const fin = cumulative[cumulative.length - 1]; return cumulative.map(c => c / fin); };
  f.pace_projection = ({ units_today, share }) => units_today / share;
  f.residual_allocation = ({ gap, reference }) => {
    const inc = reference.slice(1).map((b, i) => b - reference[i]), tot = sum(inc);
    const raw = inc.map(i => gap * i / tot), base = raw.map(Math.floor);
    const short = pyRound(gap) - sum(base);
    const order = raw.map((_, i) => i).sort((a, b) => (-(raw[a] - base[a])) - (-(raw[b] - base[b])) || a - b);
    order.slice(0, Math.max(short, 0)).forEach(i => base[i] += 1);
    return base;
  };
  // targets
  f.survival_rate = ({ net, gross }) => net / gross;
  f.gross_target = ({ net_target, survival }) => net_target / survival;
  f.expected_net = ({ gross_today, survival_from_here }) => gross_today * survival_from_here;
  f.gross_still_needed = ({ net_target, gross_today, survival_existing, survival_new }) => Math.max(0, (net_target - gross_today * survival_existing) / survival_new);
  // stats
  f.credibility_weight = ({ n, k }) => n / (n + k);
  f.credibility_blend = ({ own, benchmark, z }) => z * own + (1 - z) * benchmark;
  f.buhlmann_k = ({ groups }) => {
    groups = groups.filter(g => g.length > 1);
    const n = groups.map(g => g.length), means = groups.map(g => sum(g) / g.length);
    const within = sum(groups.map((g, i) => sum(g.map(x => (x - means[i]) ** 2)) / (g.length - 1))) / groups.length;
    const grand = sum(groups.map(sum)) / sum(n), nbar = sum(n) / n.length;
    const between = sum(means.map(m => (m - grand) ** 2)) / (groups.length - 1) - within / nbar;
    return between <= 0 ? null : within / between;
  };
  f.shrink_rate = ({ successes, trials, prior, m }) => (successes + m * prior) / (trials + m);
  f.variance_gate = ({ old, new: nw, tolerance }) => { const change = Math.abs(nw - old) / Math.max(Math.abs(old), 1); return { change, pass: change <= tolerance }; };
  f.reconcile_match = ({ a, b, tolerance }) => { const change = Math.abs(a - b) / Math.max(Math.abs(a), Math.abs(b), 1e-12); return { change, pass: change <= tolerance }; };
  f.wape = ({ actual, forecast }) => sum(actual.map((a, i) => Math.abs(a - forecast[i]))) / sum(actual.map(Math.abs));

  // Python's round() is round-half-to-even
  function pyRound(x) { const r = Math.round(x); return Math.abs(x % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r; }

  function parse(value, kind) {
    if (kind === "list") return String(value).replace(/;/g, ",").split(",").filter(s => s.trim()).map(Number);
    if (kind === "groups") return String(value).split("|").filter(s => s.trim()).map(g => g.split(",").filter(s => s.trim()).map(Number));
    if (kind === "int") return Math.trunc(Number(value));
    return Number(value);
  }
  function run(spec, raw = {}) {
    const args = {};
    spec.params.forEach(p => args[p.name] = parse(raw[p.name] ?? p.default, p.kind));
    return f[spec.key](args);
  }
  return { f, run, parse };
})();
