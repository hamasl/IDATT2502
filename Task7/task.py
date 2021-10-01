from sklearn import decomposition
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pandas as pd

if __name__ == "__main__":
    data = pd.read_csv("../CommonData/agaricus-lepiota.data")
    dummies = pd.get_dummies(data)
    #First index is for the i in the for loop, second is for the actual score, third is for the prediction
    info = []
    best = (-1,-1,-1)
    for i in range(2,31):
        kmeans = KMeans(n_clusters=i)
        kmeans.fit(dummies)
        labels = kmeans.labels_
        score = silhouette_score(dummies, labels, metric="euclidean")
        y_kmeans = kmeans.predict(dummies)
        info.append((i, score, y_kmeans))
        if score > best[1]:
            best = info[i-2]
    print(best)

    plt.title("Score vs clusters")
    plt.xlabel("num of clusters")
    plt.ylabel("Silhoutte score")
    plt.plot([info[i][0] for i in range(len(info))], [info[i][1] for i in range(len(info))])
    plt.show()
    pca = decomposition.PCA(n_components=3)
    x_pca = pca.fit_transform(dummies, best[2])
    print(x_pca.shape)


    fig = plt.figure("Linear regression 3d")
    ax = fig.add_subplot(projection='3d')
    ax.set_title("Linear regression 3d")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("y")
    # Scatter for the actual data
    ax.scatter(x_pca[:, 0], x_pca[:, 1], x_pca[:, 2], c=best[2], s=50, cmap="viridis")
    plt.show()