"""Predeclared larger follow-up using the item-safety simulator copy.

The two model initializations share the 64 independently seeded games. The
simulator changes and stronger item-using opponents mean this is a new setting;
do not compare its raw ranks as if they were a controlled before/after trial.
"""
import tft_intense_suite as suite

suite.OUT = suite.BASE / 'tft_item_suite_20261122'
suite.DATA_SEEDS = list(range(10001, 10065))
suite.CALIBRATION_SEEDS = list(range(11001, 11017))
suite.VALIDATION_SEEDS = list(range(12001, 12017))
suite.TEST_SEEDS = list(range(13001, 13017))
suite.TRAIN_SEEDS = [147, 183]

if __name__ == '__main__':
    suite.main()
