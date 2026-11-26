"""Independent-seed ablation after FIFO overflow and Yuumi pre-cast fixes."""
import tft_intense_suite as suite

suite.OUT = suite.BASE / 'tft_v2_suite_20261122'
suite.DATA_SEEDS = list(range(15001, 15065))
suite.CALIBRATION_SEEDS = list(range(16001, 16017))
suite.VALIDATION_SEEDS = list(range(17001, 17017))
suite.TEST_SEEDS = list(range(18001, 18017))
suite.TRAIN_SEEDS = [247, 283]

def balanced_collect(seeds):
    # Rotate behaviour across eight-seat blocks; over 32 games each behaviour
    # visits every seat once and all behaviours appear in the calibration set.
    kinds = ['teacher', 'econ', 'diffusion', 'diffusion_gibbs']
    result = []
    for i, seed in enumerate(seeds):
        kind = kinds[(i // 8 + i) % len(kinds)]
        model = suite.PREVIOUS / ('diffusion_gibbs.pt' if kind == 'diffusion_gibbs' else 'diffusion.pt')
        result.append(suite.task(seed, kind, True, model))
    return result

suite.collect_tasks = balanced_collect

if __name__ == '__main__':
    suite.main()
