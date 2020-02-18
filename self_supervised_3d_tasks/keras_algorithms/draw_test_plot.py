import matplotlib.pyplot as plt
import pandas


def draw_curve(path, name):
    # TODO: load multiple algorithms here
    # helper function to plot results curve
    df = pandas.read_csv(path)

    plt.plot(df["Train Split"], df["Weights initialized"], label=name + ' Pretrained')
    plt.plot(df["Train Split"], df["Weights random"], label='Random')
    # plt.plot(df["Train Split"], df["Weights frozen"], label=name + 'Frozen')

    plt.legend()
    plt.show()

    print(df["Train Split"])


if __name__ == "__main__":
    path = ""
    name = ""
    draw_curve(path, name)
