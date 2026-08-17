from wrist_fracture.paths import get_paths


def test_paths_creation(tmp_path):
    paths = get_paths(tmp_path).ensure()
    assert paths.raw.exists()
    assert paths.figures.exists()
