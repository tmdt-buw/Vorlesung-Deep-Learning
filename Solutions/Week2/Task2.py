import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.neighbors import KNeighborsClassifier as KNN


def plot_scores(_neighbors_list, _neighbor_scores):
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    ax1.plot(_neighbors_list, _neighbor_scores[:, 0], label="Training accuracy")
    ax1.plot(_neighbors_list, _neighbor_scores[:, 1], label='Test accuracy')
    ax2.plot(_neighbors_list, _neighbor_scores[:, 2], label="Cross validated training accuracy")
    ax2.plot(_neighbors_list, _neighbor_scores[:, 3], label='Cross validated test accuracy')
    for ax in [ax1, ax2]:
        ax.set_xticks(_neighbors_list)
        ax.set_xlabel("Number of nearest neighbors")
        ax.set_ylabel("Accuracy")
        ax.legend()
    plt.tight_layout()
    plt.show()


def visualize(model, X, y, X_new=None, y_new=None):
    # create a mesh grid
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # save the decision boundary of the model.
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # add some random noise to the data for plotting purposes only so that overlapping points can be identified.
    noise = np.random.normal(0, 0.02, X.shape)
    X = X + noise

    # create a custom colormap that maps a specific color to each flower class
    cmap = plt.cm.coolwarm
    colors = cmap(np.linspace(0, 1, 3, endpoint=True))

    # set the size of the figure
    plt.rcParams["figure.figsize"] = (10, 6)

    # plot the decision boundary of the model in color code
    plt.contourf(xx, yy, Z, cmap=cmap, alpha=0.5)

    # plot the data points for each flower class
    for i_class in np.unique(y):
        plt.scatter(X[:, 0][y == i_class], X[:, 1][y == i_class],
                    s=64, c=colors[i_class], edgecolor='k',
                    cmap=plt.cm.coolwarm, alpha=0.7,
                    label=iris['target_names'][i_class])

    if X_new is not None and y_new is not None:
        for i_class in np.unique(y_new):
            plt.scatter(X_new[:, 0][y_new == i_class],
                        X_new[:, 1][y_new == i_class],
                        marker='^', s=96, c=colors[i_class],
                        edgecolor='k', cmap=plt.cm.coolwarm, alpha=0.7)

    # set plotting parameters such as axis labels, ranges, a title and the legend.
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title('KNN Classification')
    plt.legend()

    # show the plot
    plt.show()


if __name__ == "__main__":

    iris = datasets.load_iris()

    """ Task 2a) """
    X = iris.data
    pca = PCA(n_components=2)
    X_trans = pca.fit_transform(X)

    """ Task 2b) """
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X_trans, y, test_size=0.3)

    """ Task 2c) """
    neighbors_list = np.arange(1, 51)
    neighbor_scores = np.zeros((len(neighbors_list), 4))
    for i, num_neighbors in enumerate(neighbors_list):
        clf = KNN(n_neighbors=num_neighbors)
        clf.fit(X_train, y_train)
        score_train = clf.score(X_train, y_train)
        score_test = clf.score(X_test, y_test)
        scores_cv = cross_validate(clf, iris.data, iris.target, cv=10, return_train_score=True)
        neighbor_scores[i] = (score_train, score_test,
                              np.mean(scores_cv["train_score"]), np.mean(scores_cv["test_score"]))

    plot_scores(neighbors_list, neighbor_scores)

    """ Task 2d) """
    feature_stats = np.zeros((4, 4))
    for i_feature in range(4):
        feature_stats[i_feature] = (np.min(iris.data[:, i_feature]),
                                    np.max(iris.data[:, i_feature]),
                                    np.mean(iris.data[:, i_feature]),
                                    np.std(iris.data[:, i_feature]))

    # Erstelle neue Datenpunkte
    X_new = np.array([np.random.normal(loc=feature_stats[0, 2], scale=feature_stats[0, 3], size=50),
                     np.random.normal(loc=feature_stats[1, 2], scale=feature_stats[1, 3], size=50),
                     np.random.normal(loc=feature_stats[2, 2], scale=feature_stats[2, 3], size=50),
                     np.random.normal(loc=feature_stats[3, 2], scale=feature_stats[3, 3], size=50)]).T

    X_new_trans = pca.transform(X_new)
    clf = KNN(n_neighbors=20).fit(X_train, y_train)
    y_new = clf.predict(X_new_trans)
    print(y_new)

    """ Task 2e) """
    visualize(clf, X_train, y_train, X_new_trans, y_new)