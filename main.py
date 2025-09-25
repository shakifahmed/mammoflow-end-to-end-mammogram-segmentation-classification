import os
import time
import numpy as np
from load_directory import directory
from segmentation import segmentation
from augmentation import augmentation
from loader import Loader
from visualization import Visualization
from model import MyModel

def main():
    source_dir = "/teamspace/studios/this_studio/images"
    output_dir = "/teamspace/studios/this_studio/segmentation"
    save_dir = "/teamspace/studios/this_studio/visualization"
    aug_dir = "/teamspace/studios/this_studio/augmentation"
    model_dir = "/teamspace/studios/this_studio/model"

    # Batch process and save outputs
    directory(source_dir, output_dir, save_dir)

    # Save ONE visualization image
    example_path = "/teamspace/studios/this_studio/images/Density1Benign/20588680.png"

    if example_path and os.path.exists(example_path):
        base = os.path.splitext(os.path.basename(example_path))[0]
        viz_path = os.path.join(save_dir, f"{base}_visualization.png")
        print("Saving visualization to:", viz_path)
        
        # Make sure save_dir exists
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        segmentation(example_path, visualize=True, save_path=viz_path)
    else:
        print("No PNG images found to visualize.")

    augmentation(input_dir=output_dir, output_dir=aug_dir)

    images = Loader.data_loader(aug_dir)
    # print(images)
    X, y = Loader.separator(images=images)
    Visualization.load_visual(save_dir=save_dir, X=X, y=y)
    df = Loader.dataframe(X, y)
    Visualization.countplot(save_dir, X=df, filename='label_count.png')
    X_train, X_test, y_train, y_test = Loader.split(df)

    print('Training Shape:')
    print(X_train.shape)
    print(y_train.shape)

    print('\nTesting Shape:')
    print(X_test.shape)
    print(y_test.shape)

    X_train, X_test, y_train = Loader.data_balance(X_train, X_test, y_train)
    # print(type(X_train))
    Visualization.countplot(save_dir, X=X_train, y=y_train, smote=True, filename="smote", title='SMOTE Apply')
    X_train_scaled,  X_test_scaled = Loader.normalize(X_train, X_test)

    X_train_scaled_dl, X_test_scaled_dl = Loader.revert(X_train_scaled,  X_test_scaled)
    Visualization.scaled_visual(X_train_scaled_dl, y_train, save_dir)

    model = MyModel.model()
    start_time = time.time()
    model_res, history = MyModel.compile(model, X_train_scaled_dl, y_train, X_test_scaled_dl, y_test)
    end_time = time.time()
    total_time = end_time-start_time
    print('Total Time:',np.round(total_time,2),'s')

    cm, accuracy, precision, recall, f1 = MyModel.performance(model_res, X_test_scaled_dl, y_test)

    print("Accuracy:", f"{accuracy:.3f}")
    print("Precision:", f"{precision:.3f}")
    print("Recall:", f"{recall:.3f}")
    print("F1 Score:", f"{f1:.3f}")

    Visualization.cm_visual(cm, save_dir)
    Visualization.acc_loss_visual(history, save_dir)

    if not os.path.exists(model_dir):
            os.makedirs(model_dir)
    model_path = os.path.join(model_dir, 'brcan_model.keras')
    model_res.save(model_path)

if __name__ == "__main__":
    main()