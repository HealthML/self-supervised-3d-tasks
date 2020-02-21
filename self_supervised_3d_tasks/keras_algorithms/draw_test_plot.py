import matplotlib.pyplot as plt
import pandas


def draw_curve(path, name, metric):
    df = pandas.read_csv(path)
    df['split'] = [int(i[:-1]) for i in df['Train Split']]
    df = df.sort_values(by=['split'])

    plt.plot(df["Train Split"], df["Weights_initialized_"+metric], label=name + ' - Encoder Trainable')
    plt.plot(df["Train Split"], df["Weights_frozen_"+metric], label=name + ' - Encoder Frozen')
    plt.plot(df["Train Split"], df["Weights_random_"+metric], label='Random')

    plt.title(f"Comparison of averaged {metric} for kaggle retina 2019. \nbatch_size=32,repetitions=3,epochs=12", pad=30)

    plt.legend()
    plt.show()


def draw_convergence():
    x1 = [0.7515,0.7515,0.7479,0.7515,0.7503,0.7527,0.7636,0.8303,0.8861,0.8897]
    x2 = [0.8715,0.8861,0.8836,0.8788,0.8812,0.8776,0.8921,0.8521,0.8885,0.8897]

    plt.plot(x2, label="Weights copied from Jigsaw")
    plt.plot(x1, label="Weights randomly initialized")

    plt.title("Comparison of accuracy scores after n epochs")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    path = "/home/Winfried.Loetzsch/workspace/self-supervised-transfer-learning/jigsaw_kaggle_retina_2/weights-improvement-183_test_3/results.csv"
    name = "Jigsaw"
    # draw_convergence()
    draw_curve(path, name, "qw_kappa_kaggle")
