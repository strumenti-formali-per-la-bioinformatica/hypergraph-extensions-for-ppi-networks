import matplotlib.pyplot as plt

def plot_results(history):
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].plot(history['train']['loss'], label='Train loss')
    ax[0].plot(history['val']['loss'], label='Val loss')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[0].legend()
    ax[1].plot(history['train']['auc'], label='Train AUC')
    ax[1].plot(history['val']['auc'], label='Val AUC')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('AUC')
    ax[1].legend()
    plt.savefig('plots/history.png')
