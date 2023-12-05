from fire_model_build import *

file_location = os.path.abspath(__file__)
root_directory = os.path.dirname(file_location)


def main():
    dataset_path = os.path.join(root_directory, '..', 'fire_dataset')
    model_path = os.path.join(root_directory, '..', 'saved_model', 'fire_model')
    load_model = os.path.exists(model_path)
    model = ModelBuilder(dataset_path, 'fire_model', load_model)
    model.pca_variance_test()
    model.build_model()
    model.train_model()
    model.evaluate_model()
    model.save_model()
    model.model_confusion()


if __name__ == "__main__":
    main()
