import os

import telluric as tl

#%% Constants
DATA_PATH = 'data/high_res_Para/classification'
BEFORE_DATASET_NAME = 'analytic_2019-06_2019-11_mosaic'
AFTER_DATASET_NAME = 'analytic_2020-06_2020-08_mosaic'

if __name__ == "__main__":
    before_dataset_path = os.path.join(DATA_PATH, BEFORE_DATASET_NAME + '.geojson')
    after_dataset_path = os.path.join(DATA_PATH, AFTER_DATASET_NAME + '.geojson')

    before_feature_collection = tl.FeatureCollection(tl.FileCollection.open(before_dataset_path))
    after_feature_collection = tl.FeatureCollection(tl.FileCollection.open(after_dataset_path))

    # Both collections are assumed to contain the same polygons, only differing in their labels.
    assert len(before_feature_collection) == len(after_feature_collection)

    for (before_feature, after_feature) in zip(before_feature_collection, after_feature_collection):
        assert before_feature.geometry == after_feature.geometry
        print(before_feature.properties)
        print(after_feature.properties)
