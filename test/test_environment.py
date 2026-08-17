from wrist_fracture.environment import collect_environment_metadata


def test_environment_metadata():
    env = collect_environment_metadata()
    assert env.python_version
    assert env.ram_gb > 0
