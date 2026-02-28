import json
import matplotlib.pyplot as plt


def logging_plotting(training_loss_history, output_dir):
    
    # Save training loss as a figure
    steps = [entry["step"] for entry in training_loss_history]
    losses = [entry["loss"] for entry in training_loss_history]
    plt.figure()
    plt.plot(steps, losses, marker='o')
    plt.xlabel("Step")
    plt.ylabel("Training Loss")
    plt.title("Training Loss over Steps")
    plt.grid(True)
    plt.savefig(f"{output_dir}/training_loss_history.png")
    plt.close()



