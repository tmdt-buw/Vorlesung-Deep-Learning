import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets


def plot_hist(data, target):
    """ Task 1b) """
    fig = plt.figure(figsize=(10, 4))
    for i_feature in range(data.shape[1]):
        ax = plt.subplot2grid((1, 4), (0, i_feature))
        for i_class in range(len(np.unique(target))):
            ax.hist(data[:, i_feature][target == i_class],
                    range=((data[:, i_feature].min(), data[:, i_feature].max())),
                    alpha=0.5)
        ax.axvline(x=np.mean(data[:, i_feature]), c='k', ls='--')
        ax.set_xlabel("{}".format(iris.feature_names[i_feature]))
        ax.set_ylabel("Häufigkeit")

    plt.tight_layout()
    plt.show()

def plot_scatter(data, target):
    """ Task 1c) """
    fig = plt.figure(figsize=(10, 10))
    for i_feature in range(data.shape[1]):
        for j_feature in range(data.shape[1]):
            ax = plt.subplot2grid((4, 4), (i_feature, j_feature))
            if i_feature == j_feature:
                for i_class in range(len(np.unique(target))):
                    ax.hist(data[:, i_feature][target == i_class],
                            range=((data[:, i_feature].min(), data[:, i_feature].max())),
                            alpha=0.5)
                ax.axvline(x=np.mean(data[:, i_feature]), c='k', ls='--')
                ax.set_xlabel("{}".format(iris.feature_names[i_feature]))
                ax.set_ylabel("Häufigkeit")
            else:
                for i_class in range(len(np.unique(target))):
                    ax.scatter(data[:, i_feature][target == i_class],
                               data[:, j_feature][target == i_class],
                               alpha=0.5)
                ax.set_xlabel("{}".format(iris.feature_names[i_feature]))
                ax.set_ylabel("{}".format(iris.feature_names[j_feature]))

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    iris = datasets.load_iris()
    print(iris.keys(), '\n')

    """ Task 1a) """
    print("Es gibt {0} Klassen.".format(len(iris.target_names)))
    print("Die Namen der Klassen lauten {}, {} und {}".format(*iris.target_names))
    labels = np.unique(iris.target)
    print("Die Klassen haben jeweils {}, {} und {} Datenpunkte".format(sum(iris.target == labels[0]),
                                                                       sum(iris.target == labels[1]),
                                                                       sum(iris.target == labels[2])))
    print("Jeder Datenpunkt ist durch {} Features beschrieben".format(len(iris.data[0])))
    print("Die Features repräsentieren {}, {}, {} und {}".format(*iris.feature_names))

    feature_stats = np.zeros((4, 4))
    for i_feature in range(4):
        feature_stats[i_feature] = (np.min(iris.data[:, i_feature]),
                                    np.max(iris.data[:, i_feature]),
                                    np.mean(iris.data[:, i_feature]),
                                    np.std(iris.data[:, i_feature]))
    print("Die statistischen Ausprägungen Minimum, Maximum, Mittelwert und Standardabweichuing der 4 Features "
          "sind durch die folgende Matrix gegeben:")
    print(feature_stats)

    """ Task 1b) """
    plot_hist(iris.data, iris.target)

    """ Task 1c) """
    plot_scatter(iris.data, iris.target)