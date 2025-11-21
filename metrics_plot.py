import matplotlib.pyplot as plt

def plot_training_loss_curve(metrics: dict):
    plt.plot(metrics['best_model_training_loss_curve'])
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss (MSE)")
    plt.title("Training Loss Curve for Best Hyperparameters")
    plt.show()

def plot_predicted_vs_actual(metrics: dict):
    Y_pred = metrics['Y_pred']
    Y_test = metrics['Y_test']
    col = metrics['column']
    fig, axes = plt.subplots(2,2, figsize=(12, 10))
    for i in range(Y_test.shape[1]):
        ax = axes[i//2, i%2]
        ax.scatter(Y_pred[:, i], Y_test[:, i])
        ax.plot([min(Y_test[:, i]), max(Y_test[:, i])], [min(Y_test[:, i]), max(Y_test[:, i])], color='red')  # 添加y=x参考线
        ax.set_xlabel(f"Predicted {col} t+{[1,6,12,24][i]}")
        ax.set_ylabel(f"Actual {col} t+{[1,6,12,24][i]}")
        ax.set_title(f"Predicted vs Actual {col} t+{[1,6,12,24][i]}")
    plt.show()

