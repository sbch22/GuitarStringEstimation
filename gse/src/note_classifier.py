# import
import os
import sys
sys.path.append(os.path.abspath(''))

from configparser import ConfigParser

# import functions
import YMT3_inference
import extract_partials



# test config
def create_test_config():
    config = ConfigParser()
    config['test'] = {
        'W': '4096',
        'H': '256',
        'beta_max': '4e-4',
        'threshold': '-50',
        'plot': False
    }
    config['paths'] = {
        'track_directory': '../noteData/GuitarSet/test/'
    }

    with open('config.ini', 'w') as f:
        config.write(f)

    return config


def main():
    config = create_test_config()

    W = config.getint('test', 'W')
    H = config.getint('test', 'H')
    beta_max = config.getfloat('test', 'beta_max')
    threshold = config.getint('test', 'threshold')
    plot = config.getboolean('test', 'plot')
    track_directory = config.get('paths', 'track_directory')

    print(W, H, beta_max, threshold)
    print(track_directory)

    # Run inference
    YMT3_inference.main(track_directory)

    # Run partial extraction with custom config
    extract_partials.main(
        track_directory=track_directory,
        W=W,
        H=H,
        beta_max=beta_max,
        threshold=threshold,
        plot=plot
    )


if __name__ == "__main__":
    main()