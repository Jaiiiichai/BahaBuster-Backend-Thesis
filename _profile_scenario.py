import cProfile
from model_training.prediction import get_weather, build_moderate_weather_scenario, _load_flood_condition_profiles, _load_feature_defaults

_load_flood_condition_profiles()
_load_feature_defaults()
w = get_weather()

def run():
    for _ in range(20):
        build_moderate_weather_scenario(w, 2.4, barangay="BANILAD")

cProfile.run("run()", sort="cumulative")
