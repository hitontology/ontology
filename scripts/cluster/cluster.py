from collections import defaultdict
from rdflib import Graph
from sklearn.cluster import AffinityPropagation, KMeans
from sklearn.feature_extraction import DictVectorizer
from sklearn import preprocessing

from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import mplcursors

g = Graph()
HITO = "http://hitontology.eu/ontology/"
g.bind("hito", HITO)
# rdflib v5 as installed by pip doesn't support remote SPARQL querying.
# When rdflib v6 is officially release this can be changed using the SERVICE keyword.
# The ontology/combine script needs to be executed first to created the file.
FILENAME = "/tmp/hito-all.nt"
g.parse(FILENAME, format="nt")

QUERY = """SELECT ?source (GROUP_CONCAT(?target; separator=" ") AS ?targets) {
  ?source   a hito:SoftwareProduct;
            ?p ?citation.
  ?citation ?q ?target.

 ?p rdfs:subPropertyOf hito:citation.
 ?q rdfs:subPropertyOf hito:classified.
} GROUP BY ?source"""

# use sklearn dict vectorizers and feature extraction
def cluster():
    result = g.query(QUERY)
    print(len(result))
    D = []
    for row in result:
        #D.append({"uri": str(row["source"]), "classifieds": row["targets"].split()})
        D.append({"classifieds": row["targets"].split()})
    vec = DictVectorizer(sparse=False)
    data = vec.fit_transform(D)
    data = preprocessing.normalize(data, norm='l2')
    #print(vec.get_feature_names())
    reduced_data = PCA(n_components=2,whiten=True).fit_transform(data)
    print(reduced_data)
    # clusters = AffinityPropagation(random_state=0).fit(reduced)
    kmeans = KMeans(init="k-means++", n_clusters=4, n_init=2)
    kmeans.fit(reduced_data)

    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = 0.01  # point in the mesh [x_min, x_max]x[y_min, y_max].

    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Obtain labels for each point in mesh. Use last trained model.
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.clf()
    plt.imshow(
        Z,
        interpolation="nearest",
        extent=(xx.min(), xx.max(), yy.min(), yy.max()),
        cmap=plt.cm.Paired,
        aspect="auto",
        origin="lower",
    )

    plt.plot(reduced_data[:, 0], reduced_data[:, 1], "k.", markersize=2)
    # Plot the centroids as a white X
    #centroids = kmeans.cluster_centers_
    #plt.scatter(
    #    centroids[:, 0],
    #    centroids[:, 1],
    #    marker="x",
    #    s=169,
    #    linewidths=3,
    #    color="w",
    #    zorder=10,
    #)
    plt.title(
        "Clustering on the HITO software products PCA-reduced data)\n"
        #"Centroids are marked with white cross"
    )
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())

    cursor = mplcursors.cursor(hover=True)
    #cursor.connect("add"

    plt.show()


cluster()
