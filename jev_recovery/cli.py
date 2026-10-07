import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import mean
from .environment import Action, RecoveryEnv, Scenario, TEST_FAULTS, TRAIN_FAULTS
from .jev import DiagnosticProvider
from .policy import Checklist, QPolicy, train

ROOT = Path(__file__).resolve().parent.parent


def _seeds(text):
    try:
        seeds = tuple(int(item.strip()) for item in text.split(','))
    except (ValueError, AttributeError) as exc:
        raise ValueError('Seeds must be comma-separated integers') from exc
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('At least one seed required; seeds must be distinct')
    return seeds


def research(args):
    from .benchmark import run_benchmark
    options = dict(episodes=args.episodes, count=args.count, seeds=_seeds(args.seeds),
                   advisor_mode=args.advisor)
    if args.config:
        config = json.loads(Path(args.config).expanduser().resolve().read_text())
        allowed = {'episodes', 'count', 'seeds', 'advisor_mode'}
        if not isinstance(config, dict) or set(config) - allowed:
            raise ValueError('Config must be an object containing only episodes/count/seeds/advisor_mode')
        options.update(config)
        if isinstance(options['seeds'], str):
            options['seeds'] = _seeds(options['seeds'])
        else:
            seeds = options['seeds']
            if not isinstance(seeds, (tuple, list)) or not seeds or any(type(s) is not int for s in seeds):
                raise ValueError('Config seeds must be a nonempty list of integers')
            options['seeds'] = _seeds(','.join(str(seed) for seed in seeds))
    for key in ('episodes', 'count'):
        if type(options[key]) is not int or options[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if options['advisor_mode'] not in ('offline', 'cached', 'live'):
        raise ValueError('Invalid advisor_mode')
    output = Path(args.output).expanduser().resolve()
    report = run_benchmark(output=output, cache_path=Path(args.cache).expanduser().resolve(), **options)
    print(f'Research report: {output / "report.md"}')
    print(report['advisor']['label'])
    print('policy                          recovery actions downtime tool_cost')
    for name, policy in report['policies'].items():
        values = policy['overall']
        print(f"{name:31} {values['recovery_rate']:7.1%} {values['mean_actions']:7.2f} "
              f"{values['mean_downtime_ticks']:8.2f} {values['mean_tool_cost_units']:9.2f}")


def inspect_run(args):
    from .benchmark import audit_run, replay_run
    result = (audit_run if args.command == 'audit' else replay_run)(Path(args.output).expanduser().resolve())
    print(json.dumps(result, indent=2))
    if result.get('valid') is False or result.get('ok') is False:
        raise ValueError('Artifact verification failed')


def run_episode(policy, scenario, provider):
    env = RecoveryEnv(scenario, provider)
    while not env.done:
        env.step(policy.choose(env))
    return {'recovered': env.success, 'actions': env.actions, 'downtime_ticks': env.downtime,
            'interventions': env.interventions, 'tool_cost_units': env.tool_cost,
            'api_cost_usd': env.api_cost_usd, 'input_tokens': env.input_tokens,
            'api_calls': env.api_calls, 'cache_hits': env.cache_hits, 'trace': env.trace}


def aggregate(rows):
    keys = ('recovered', 'actions', 'downtime_ticks', 'interventions', 'tool_cost_units',
            'api_cost_usd', 'input_tokens', 'api_calls', 'cache_hits')
    return {k: mean(float(row[k]) for row in rows) for k in keys}


def experiment(args):
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    provider = DiagnosticProvider(mode=args.advisor, cache_path=Path(args.cache).expanduser().resolve())
    seeds = [int(s) for s in args.seeds.split(',')]
    if args.episodes < 1 or args.repeats < 1 or not seeds:
        raise ValueError('Positive episode/repeat counts and at least one seed required')
    rows = []
    # Wording-only, combination-only and joint holdouts isolate generalization axes.
    suites = {'wording': (TRAIN_FAULTS, 'heldout'),
              'combinations': (TEST_FAULTS, 'train'), 'joint': (TEST_FAULTS, 'heldout')}
    for seed in seeds:
        policies = {'checklist': Checklist()}
        for name, use_jev in [('rl_without_jev', False), ('rl_with_jev', True)]:
            policies[name] = train(seed, args.episodes, use_jev, provider)
            policies[name].save(output / f'{name}-seed{seed}.json')
        # Evaluation freezes policies: run_episode cannot update Q values.
        for suite, (combinations, wording) in suites.items():
            for faults in combinations:
                for repeat in range(args.repeats):
                    scenario = Scenario(faults, wording, dependency_recovers_at=5 + repeat % 3)
                    for name, policy in policies.items():
                        row = run_episode(policy, scenario, provider)
                        row.update(policy=name, seed=seed, suite=suite, faults=list(faults),
                                   wording=wording, repeat=repeat)
                        rows.append(row)
    groups = defaultdict(list)
    seed_groups = defaultdict(list)
    for row in rows:
        groups[(row['suite'], row['policy'])].append(row)
        seed_groups[(row['suite'], row['policy'], row['seed'])].append(row)
    report = {'advisor_mode': args.advisor,
        'evidence': 'Offline heuristic; NOT measured hosted Jev accuracy or benefit.' if args.advisor == 'offline'
                    else 'Hosted Jev responses; health checks independently determine recovery.',
        'provider_total_acquisition': dict(provider.statistics),
        'training': {'episodes_per_policy_per_seed': args.episodes, 'seeds': seeds,
                     'faults': TRAIN_FAULTS, 'wording': 'train'},
        'evaluation': {'repeats': args.repeats, 'max_ticks': 24, 'sustained_probes': 5,
                       'paired_scenarios': True},
        'summary': {f'{suite}/{policy}': aggregate(items) for (suite, policy), items in groups.items()},
        'per_seed': {f'{suite}/{policy}/{seed}': aggregate(items)
                     for (suite, policy, seed), items in seed_groups.items()}}
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    (output / 'episodes.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    print(report['evidence'])
    print('suite/policy                  recovery  actions downtime tool_cost api_USD')
    for label, metrics in report['summary'].items():
        print(f"{label:30} {metrics['recovered']:8.1%} {metrics['actions']:7.2f} "
              f"{metrics['downtime_ticks']:8.2f} {metrics['tool_cost_units']:9.2f} "
              f"{metrics['api_cost_usd']:.8f}")
    print(f'Report: {output / "report.json"}')


def main():
    parser = argparse.ArgumentParser(description='Local recovery RL experiments; no production effects')
    commands = parser.add_subparsers(dest='command', required=True)
    exp = commands.add_parser('experiment', help='Train local controllers, freeze, and compare held-out incidents')
    exp.add_argument('--episodes', type=int, default=20000)
    exp.add_argument('--seeds', default='11,22,33')
    exp.add_argument('--repeats', type=int, default=9)
    exp.add_argument('--advisor', choices=('offline', 'cached', 'live'), default='offline')
    exp.add_argument('--cache', default=str(ROOT / 'artifacts' / 'diagnostics.json'))
    exp.add_argument('--output', default=str(ROOT / 'artifacts' / 'baseline'))
    exp.set_defaults(func=experiment)
    lab = commands.add_parser('benchmark', help='Train feature-based agents and run reproducible paired v2 benchmarks')
    lab.add_argument('--episodes', type=int, default=4000)
    lab.add_argument('--count', type=int, default=60, help='Independent incidents per suite and evaluation seed')
    lab.add_argument('--seeds', default='11,22,33')
    lab.add_argument('--advisor', choices=('offline', 'cached', 'live'), default='offline')
    lab.add_argument('--cache', default=str(ROOT / 'artifacts' / 'research-diagnostics.json'))
    lab.add_argument('--output', default=str(ROOT / 'artifacts' / 'research'))
    lab.add_argument('--config', help='JSON experiment config; its values override corresponding CLI settings')
    lab.set_defaults(func=research)
    for name in ('audit', 'replay'):
        command = commands.add_parser(name, help=f'{name.capitalize()} a saved v2 experiment without retraining')
        command.add_argument('--output', default=str(ROOT / 'artifacts' / 'research'))
        command.set_defaults(func=inspect_run)
    serve = commands.add_parser('serve', help='Expose an isolated fault-injected demo service on localhost')
    serve.add_argument('--port', type=int, default=8080)
    serve.add_argument('--output', default=str(ROOT / 'artifacts' / 'research'), help='Benchmark directory used for report/checkpoints')
    serve.set_defaults(func=lambda args: __import__('jev_recovery.service', fromlist=['serve']).serve(
        args.port, output=Path(args.output).expanduser().resolve()))
    args = parser.parse_args()
    try:
        args.func(args)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f'Error: {exc}\n')
    except KeyboardInterrupt:
        parser.exit(0, 'Stopped.\n')


if __name__ == '__main__':
    main()
