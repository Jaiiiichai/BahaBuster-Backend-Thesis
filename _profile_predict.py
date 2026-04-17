import cProfile
from model_training.prediction import get_weather, build_moderate_weather_scenario, _predict_day1_depth_cm, _load_flood_condition_profiles, _load_feature_defaults

_load_flood_condition_profiles()
_load_feature_defaults()
w = get_weather()
sw = build_moderate_weather_scenario(w, 2.4, barangay="BANILAD")

def run():
    for _ in range(20):
        _predict_day1_depth_cm("BANILAD", sw)

cProfile.run("run()", sort="cumulative")
