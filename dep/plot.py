import os
import matplotlib.pyplot as plt


class Measurement(object):

    def __init__(self, test_df, classes):
        self.test_df = test_df.values
        self.classes = classes

    def max_acc(self):  # using kappa for MI, AUC for P300
        return self.test_df.max()

    def mean_acc(self):
        return self.test_df.mean()
    


def save_acc_loss_fig(self, fig_path, sub_id, mean_acc, max_acc):
        measure = Measurement(self.epoch_df['test_acc'], classes= 2)
        test_acc = self.epoch_df['test_acc'].values.tolist()
        test_loss = self.epoch_df['test_loss'].values.tolist()
        train_acc = self.epoch_df['train_acc'].values.tolist()
        train_loss = self.epoch_df['train_loss'].values.tolist()

        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        
        ax1.plot(range(len(train_acc)), train_acc, label='Train Accuracy', color='blue', linewidth=0.7)
        ax1.plot(range(len(test_acc)), test_acc, label='Test Accuracy', color='red', linewidth=0.7)
        ax1.set_title('Acc Performance')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend(loc='upper right')

        
        ax2.plot(range(len(train_loss)), train_loss, label='Train Loss', color='green', linewidth=0.7)
        ax2.plot(range(len(test_loss)), test_loss, label='Test Loss', color='purple', linewidth=0.7)
        ax2.set_title('Loss Performance')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend(loc='upper right')

        '''plt.plot(range(len(test_acc)), test_acc, label='test acc', linewidth=0.7)
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(loc='lower right')'''
        plt.tight_layout()  
        plt.savefig(os.path.join(fig_path, '%s_mean%.2f_max%.2f.png' % (sub_id, measure.mean_acc(), measure.max_acc())))