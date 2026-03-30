import os
from configparser import ConfigParser
import csv

def get_filenames(subset):
    config = ConfigParser()

    if subset == 'comp':
        config.read('configs/config_test_comp.ini')
    elif subset == 'solo':
        config.read('configs/config_test_solo.ini')

    track_directory = config.get('paths', 'track_directory')

    filenames = sorted([
        os.path.splitext(fn)[0].replace('_track', '.csv')
        for fn in os.listdir(track_directory)
        if fn.endswith('.pkl')
    ])

    return filenames


def main():
    solo_files = get_filenames('solo')
    comp_files = get_filenames('comp')

    max_len = max(len(solo_files), len(comp_files))

    solo_files += [''] * (max_len - len(solo_files))
    comp_files += [''] * (max_len - len(comp_files))

    output_path = os.path.join('configs', 'test_split.csv')
    os.makedirs('configs', exist_ok=True)

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)

        # header
        writer.writerow(['solo', 'comp'])

        # rows
        for s, c in zip(solo_files, comp_files):
            writer.writerow([s, c])

    print(f"Saved CSV to: {output_path}")


if __name__ == "__main__":
    main()