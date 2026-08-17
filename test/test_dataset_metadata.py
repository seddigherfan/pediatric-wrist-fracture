from scripts.download_dataset import FigshareFile, estimate_disk_requirement


def test_estimate_disk_requirement():
    files = [FigshareFile(1, "a.txt", None, 10, None), FigshareFile(2, "b.txt", None, 20, None)]
    assert estimate_disk_requirement(files) == 30
