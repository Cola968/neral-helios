from raw_math50_data import TrainingProblemSampler, load_benchmark, make_dev, validate_frozen_suite


def test_rawmath_suite_shape_and_balance():
    report=validate_frozen_suite()
    assert report["tasks"]==500
    assert report["languages"]=={"de":250,"en":250}
    assert len(report["categories"])==10
    assert set(report["categories"].values())=={50}
    assert report["benchmark_dev_overlap"]==0


def test_rawmath_training_sampler_never_hits_frozen_sets():
    frozen={r["canonical"] for r in load_benchmark()}
    frozen.update(r["canonical"] for r in make_dev())
    sampler=TrainingProblemSampler()
    for _ in range(5000):
        assert sampler.next()["canonical"] not in frozen
