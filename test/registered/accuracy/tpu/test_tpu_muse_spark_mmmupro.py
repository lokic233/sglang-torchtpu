"""Muse Spark NVFP4 TPU quality on all 1,730 MMMU-Pro standard-10 questions."""

import unittest

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.tpu import MUSE_SPARK, make_model_config, make_runner
from sglang.test.tpu.utils.accuracy_base import AccuracyTestCase

register_tpu_ci(
    # The renderer reserves 1.5x this estimate under a six-hour runner cap.
    est_time=10800,
    suite="nightly-tpu-accuracy-muse-spark-mmmupro",
    nightly=True,
    labels=["singlehost", "muse-spark-open-816b-a42b", "accuracy", "multimodal"],
)

CONFIG_MODULE = "recipe/configs/muse_spark/accuracy_mmmupro.py"

GATE = {
    "dataset": "mmmu_pro",
    "label": "mmmu_pro",
    "baseline": 0.75,  # Initial regression reference, pending full TPU calibration.
    "atol": 0.10,
    "runner": make_runner(MUSE_SPARK, "mmmu_pro"),
}


def build_model_config():
    return make_model_config(
        MUSE_SPARK,
        test_name="muse-spark-mmmupro",
        series_id="accuracy-muse-spark-mmmupro",
        gate=GATE,
    )


class TestMuseSparkMMMUPro(AccuracyTestCase, CustomTestCase):
    """Score images and text through the multimodal recipe's server."""

    config_module = CONFIG_MODULE
    model_path = "/weights/Muse-Spark-Open-816B-A42B-NVFP4"
    tag = "muse-spark-mmmupro-e2e"
    health_timeout = 5400

    @classmethod
    def build_model_config(cls):
        return build_model_config()

    def test_accuracy(self):
        self.run_evaluations()


if __name__ == "__main__":
    unittest.main()
