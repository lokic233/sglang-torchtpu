"""Muse Spark full MMMU-Pro evaluation with the vision tower enabled."""

from copy import deepcopy

from benchmark.tpu.recipe.configs.muse_spark.accuracy import CONFIG as ACCURACY_CONFIG

CONFIG = deepcopy(ACCURACY_CONFIG)
CONFIG["server_args"].update(enable_multimodal=True)
