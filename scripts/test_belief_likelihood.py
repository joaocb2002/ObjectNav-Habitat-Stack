import numpy as np
from objectnav.belief.legacy_bridge import likelihood_from_detection

def main():
    # Fake detection categorical distribution for K=2 classes.
    score_vec = np.array([0.7, 0.3], dtype=np.float64)
    bbox_scale = 5.0

    # Minimal fake calibration data with the right structure:
    # dirichlet_priors[class_name] -> list of alpha vectors, one per bin
    # classes_bins[class_name] -> list of bin thresholds
    dirichlet_priors = {
        "classA": [np.array([2.0, 2.0], dtype=np.float64)],
        "classB": [np.array([2.0, 2.0], dtype=np.float64)],
    }
    classes_bins = {
        "classA": [100.0],
        "classB": [100.0],
    }

    lv = likelihood_from_detection(score_vec, bbox_scale, dirichlet_priors, classes_bins)

    # Legacy returns K+1 with background appended. :contentReference[oaicite:1]{index=1}
    assert lv.shape == (3,), f"Expected (3,), got {lv.shape}"
    assert np.isfinite(lv).all()
    assert abs(lv.sum() - 1.0) < 1e-6, f"Sum is {lv.sum()} (expected ~1.0)"
    assert lv[-1] > 0.0, "Background likelihood should be > 0"

    print("Belief likelihood OK:", lv)

if __name__ == "__main__":
    main()
