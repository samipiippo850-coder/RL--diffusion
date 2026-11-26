"""Finite empirical Gibbs reweighting; no model or environment dependencies."""
import math


def gibbs_weights(costs, beta):
    costs = [float(c) for c in costs]
    beta = float(beta)
    if not costs or not math.isfinite(beta) or beta < 0 or not all(map(math.isfinite, costs)):
        raise ValueError('costs must be nonempty and finite; beta must be finite and nonnegative')
    minimum = min(costs)
    raw = [math.exp(-beta * (c - minimum)) for c in costs]
    total = math.fsum(raw)
    return [v / total for v in raw]


def diagnostics(costs, beta):
    costs = list(costs)
    p = gibbs_weights(costs, beta)
    n = len(p)
    ess = 1.0 / math.fsum(x * x for x in p)
    return {'n': n, 'beta': beta, 'ess': ess, 'ess_fraction': ess / n,
            'max_weight': max(p), 'weighted_cost': math.fsum(w * c for w, c in zip(p, costs)),
            'uniform_cost': math.fsum(costs) / n,
            'kl_to_uniform': math.fsum(w * math.log(w * n) for w in p if w > 0)}


if __name__ == '__main__':
    import argparse
    import csv
    import json
    parser = argparse.ArgumentParser(description='Analyze one cost per trajectory/window in a CSV file.')
    parser.add_argument('csv', help='CSV with a numeric cost column')
    parser.add_argument('--column', default='cost')
    parser.add_argument('--beta', type=float, nargs='+', default=[0.0, 0.1, 0.5, 1.0])
    args = parser.parse_args()
    with open(args.csv, newline='', encoding='utf-8-sig') as f:
        values = [float(row[args.column]) for row in csv.DictReader(f)]
    print(json.dumps([diagnostics(values, b) for b in args.beta], indent=2))
