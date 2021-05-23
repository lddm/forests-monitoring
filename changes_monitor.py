import os

import telluric as tl

#%% Constants
from tqdm import tqdm

DATA_PATH = 'data/high_res_Para/classification'
BEFORE_DATASET_NAME = 'analytic_2016-06_2016-11_mosaic'
AFTER_DATASET_NAME = 'analytic_2020-06_2020-08_mosaic'
DEFORESTATION_LABELS = ['agriculture', 'habitation']


def _check_deforestation_labels(labels, deforestation_labels):
    return any(deforestation_label in labels for deforestation_label in deforestation_labels)


if __name__ == "__main__":
    before_dataset_path = os.path.join(DATA_PATH, BEFORE_DATASET_NAME + '.geojson')
    after_dataset_path = os.path.join(DATA_PATH, AFTER_DATASET_NAME + '.geojson')

    before_feature_collection = tl.FeatureCollection(tl.FileCollection.open(before_dataset_path))
    after_feature_collection = tl.FeatureCollection(tl.FileCollection.open(after_dataset_path))

    # Both collections are assumed to contain the same polygons, only differing in their labels.
    assert len(before_feature_collection) == len(after_feature_collection)

    geo_features_list = []
    for (before_feature, after_feature) in tqdm(zip(before_feature_collection, after_feature_collection)):
        # Check that geometries are the same to be sure we are comparing changes of the same territory
        if before_feature.geometry != after_feature.geometry:
            print('Different footprints')
            continue

        # Compare labels and generate FeatureCollection only containing geometries in which there was deforestation
        before_labels = before_feature.properties['labels'].split()
        after_labels = after_feature.properties['labels'].split()
        if not _check_deforestation_labels(before_labels, DEFORESTATION_LABELS) \
                and _check_deforestation_labels(after_labels, DEFORESTATION_LABELS):
            geo_features_list.append(after_feature)

    feature_collection = tl.FeatureCollection(geo_features_list)
    output_path = os.path.join(DATA_PATH, AFTER_DATASET_NAME + '_vs_' + BEFORE_DATASET_NAME + '.geojson')
    os.makedirs(DATA_PATH, exist_ok=True)
    feature_collection.save(output_path)
