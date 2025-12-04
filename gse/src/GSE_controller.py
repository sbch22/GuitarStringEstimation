import extract_dataset
import YMT3_inference
import extract_partials
import calculate_features
import feature_classifier

def main():
    extract_dataset.main()
    YMT3_inference.main()
    extract_partials.main()
    calculate_features.main()
    feature_classifier.main()

if __name__ == "__main__":
    main()