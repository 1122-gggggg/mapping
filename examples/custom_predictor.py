"""Example adapter for a retrained ActLoc-style model.

The external model is intentionally not bundled. Replace `my_model_inference` with the
inference function trained/calibrated on your own GlueMap + localization pipeline.
"""

from sfm_diagnosis.actloc import ExternalPredictorAdapter
from sfm_diagnosis.diagnose import diagnose_pose
from sfm_diagnosis.io import load_gluemap
from sfm_diagnosis.models import Pose


def my_model_inference(map_data, pose: Pose) -> float:
    # Example only. Return a calibrated localization-success probability in [0, 1].
    raise NotImplementedError


map_data = load_gluemap("/path/to/gluemap/results")
predictor = ExternalPredictorAdapter(my_model_inference)
# diagnosis = diagnose_pose(map_data, pose, predictor=predictor)
