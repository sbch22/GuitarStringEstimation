import extract_GuitarSet
import YMT3_inference
import find_partials
import calculate_features
import feature_classifier

from configparser import ConfigParser


# TODO: test-train-split filelist -> hand over into scripts, or find different solution like saving in JSON
def create_train_config():
    config = ConfigParser()

    config['train'] = {
        'W': '4096',
        'H': '256',
        'beta_max': '2e-4',
        'threshold': '-50',
        'plot': False
    }

    config['paths'] = {
        'track_directory': '../noteData/GuitarSet/train/dev/'
    }

    with open('config.ini', 'w') as f:
        config.write(f)


def main():
    extract_dataset.main()
    YMT3_inference.main()
    extract_partials.main()
    calculate_features.main()
    feature_classifier.main()

if __name__ == "__main__":
    main()