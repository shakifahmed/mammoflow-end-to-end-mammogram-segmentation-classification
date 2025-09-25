import matplotlib.pyplot as plt
import os
import seaborn as sns
import pandas as pd

class Visualization:

    def seg_visual(img, mask, opening, tot_mean_intensity, average_mean_intensity, save_path=None):
        print('total mean intensity :', tot_mean_intensity)
        print('average mean intensity :', average_mean_intensity)
        print("Images segmented successfully!")

        fig, ax = plt.subplots(1, 3, figsize=(10, 5))
        ax[0].imshow(img, cmap='gray'); ax[0].set_title('Original Image'); ax[0].axis('off')
        ax[1].imshow(mask, cmap='gray'); ax[1].set_title('Segmented Tumor'); ax[1].axis('off')
        ax[2].imshow(opening, cmap='gray'); ax[2].set_title('apply opening'); ax[2].axis('off')

        plt.tight_layout()

        if save_path is None:
            save_path = "visualization.png" 
        plt.savefig(save_path, dpi=150)
        plt.close(fig) 
        print(f"Saved visualization to: {save_path}")

    def load_visual(save_dir, X, y):
        plt.imshow(X[1], cmap='gray')
        plt.title(y[1])
        save_dir = os.path.join(save_dir, "features_and_label_separate.png")
        plt.savefig(save_dir)
        plt.close()
        print(f"Figure saved at: {save_dir}")

    def countplot(save_dir, X, y=None, filename="smote.png", title='Dataset Class Countplot', smote=False):
        save_path = os.path.join(save_dir, filename)
        
        # Get the data to plot
        data = pd.Series(y) if smote else X['label']
        if not smote:
            print(data.value_counts())
        
        # Create bar plot
        ax = data.value_counts().sort_index().plot(kind='bar', figsize=(10, 6))
        
        # Add value annotations
        for p in ax.patches:
            ax.annotate(f'{int(p.get_height())}', 
                    (p.get_x() + p.get_width()/2., p.get_height()),
                    ha='center', va='bottom')
        
        plt.xlabel('Label')
        plt.ylabel('Count')
        plt.title(title)
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"Count plot saved at: {save_path}")

    def scaled_visual(X, y, save_dir, filename='processed_img_to_model.png'):
        save_path = os.path.join(save_dir, filename)
        plt.imshow(X[1],cmap='gray')
        plt.title(y[1])
        plt.savefig(save_path)
        print(f"image saved at: {save_path}")

    def cm_visual(cm, save_dir, filename='confusion_matrix.png'):
        save_path = os.path.join(save_dir, filename)

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.savefig(save_path)
        print(f"confusion matrix saved at: {save_path}")

    def acc_loss_visual(history, save_dir, filename='accu_loss_curve.png'):
        save_path = os.path.join(save_dir, filename)

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], color='red', label='train')
        plt.plot(history.history['val_accuracy'], color='blue', label='validation')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Train and Validation Accuracy')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], color='red', label='train')
        plt.plot(history.history['val_loss'], color='blue', label='validation')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Train and Validation Loss')
        plt.legend()

        plt.tight_layout()
        plt.show()
        
        plt.savefig(save_path)
        print(f"Accuracy and loss curve saved at: {save_path}")


