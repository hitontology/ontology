from collections import defaultdict
from rdflib import Graph
from sklearn.cluster import KMeans, DBSCAN, AffinityPropagation
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing

from sklearn.decomposition import PCA
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from adjustText import adjust_text
import mplcursors
import random
import math
import umap

# slow but nice
ADJUST_TEXT = True
NORM = "max" # l1, l2, max
CLASSIFIED_ONLY = True
BAG_OF_WORDS = False

g = Graph()
HITO = "http://hitontology.eu/ontology/"
g.bind("hito", HITO)
# rdflib v5 as installed by pip doesn't support remote SPARQL querying.
# When rdflib v6 is officially release this can be changed using the SERVICE keyword.
# The ontology/combine script needs to be executed first to created the file.
FILENAME = "/tmp/hito-all.nt"
g.parse(FILENAME, format="nt")

CLASSIFIED_ONLY_QUERY = """SELECT ?source (STR(SAMPLE(?label)) AS ?label) (GROUP_CONCAT(DISTINCT(?target); separator=" ") AS ?targets) {
  ?source   a hito:SoftwareProduct;
            rdfs:label ?label;
            ?p ?citation.
  ?citation ?q ?target.

 ?p rdfs:subPropertyOf hito:citation.
 ?q rdfs:subPropertyOf hito:classified.
} GROUP BY ?source ?label"""

QUERY = """PREFIX :<http://hitontology.eu/ontology/>
SELECT ?source (STR(SAMPLE(?label)) AS ?label) (GROUP_CONCAT(DISTINCT(?target); separator=" ") AS ?targets) {

  ?source   a hito:SoftwareProduct;
            rdfs:label ?label;
            ?p ?citation.
 {
  ?citation ?q ?target.
 }
 UNION {?source :license|:programmingLanguage|:interoperability|:operatingSystem|:client|:databaseSystem|:language ?target.}

 ?p rdfs:subPropertyOf hito:citation.
 ?q rdfs:subPropertyOf hito:classified.
} GROUP BY ?source ?label"""

CLASSIFIED_ONLY_BAG_OF_WORDS = """SELECT ?source (STR(SAMPLE(?label)) AS ?label) (GROUP_CONCAT(DISTINCT(?target); separator=" ") AS ?targets) {
SERVICE <https://hitontology.eu/sparql>
{
  ?source   a hito:SoftwareProduct;
            rdfs:label ?label;
            ?p ?citation.
  ?citation ?q [rdfs:label ?target].

 ?p rdfs:subPropertyOf hito:citation.
 ?q rdfs:subPropertyOf hito:classified.
}
} GROUP BY ?source ?label"""

BAG_OF_WORDS = """PREFIX :<http://hitontology.eu/ontology/>
SELECT ?source (STR(SAMPLE(?label)) AS ?label) (GROUP_CONCAT(DISTINCT(?target); separator=" ") AS ?targets) {
SERVICE <https://hitontology.eu/sparql>
{
  ?source   a hito:SoftwareProduct;
            rdfs:label ?label;
            ?p ?citation.
 {
  ?citation ?q [rdfs:label ?target].
 }
 UNION {?source :license|:programmingLanguage|:interoperability|:operatingSystem|:client|:databaseSystem|:language [rdfs:label ?target].}

 ?p rdfs:subPropertyOf hito:citation.
 ?q rdfs:subPropertyOf hito:classified.
}
} GROUP BY ?source ?label"""



def randompoint():
    deg = 2 * math.pi * random.random()
    R = 80
    return (math.cos(deg) * R, math.sin(deg) * R)


# use sklearn dict vectorizers and feature extraction
def cluster():
    #result = g.query(CLASSIFIED_ONLY_QUERY if CLASSIFIED_ONLY else QUERY)
    result = g.query(CLASSIFIED_ONLY_BAG_OF_WORDS if CLASSIFIED_ONLY else BAG_OF_WORDS)
    print(len(result))
    D = []
    E = []
    for row in result:
        #D.append({"classifieds": row["targets"].split()}) # labels for one-hot encoding
        D.append( str(row["targets"])) # bag of words
        E.append(
            {
                "uri": str(row["source"]),
                "label": ([row["label"].value][0]),
                "classifieds": row["targets"].split(),
            }
        )
    print(E[0]["label"])
    #print(D)
    #vec = DictVectorizer(sparse=False)
    vec = CountVectorizer()

    #data = vec.fit_transform(D)
    data = vec.fit_transform(D).toarray()
    print(data)
    data = preprocessing.normalize(data, norm=NORM)
    # print(vec.get_feature_names())
    #reduced_data = PCA(n_components=2, whiten=True).fit_transform(data)
    reduced_data = umap.UMAP(n_neighbors=15).fit_transform(data)
    print(reduced_data)
    clustering = AffinityPropagation(random_state=0).fit(reduced_data)
    #clustering = KMeans(init="k-means++", n_clusters=4, n_init=2)
    #clustering.fit(reduced_data)
    #clustering = DBSCAN(eps=0.3, min_samples=10).fit(reduced_data)

    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = 0.001  # point in the mesh [x_min, x_max]x[y_min, y_max].

    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = reduced_data[:, 0].min() - 0.2, reduced_data[:, 0].max() + 0.2
    y_min, y_max = reduced_data[:, 1].min() - 0.2, reduced_data[:, 1].max() + 0.2
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Obtain labels for each point in mesh. Use last trained model.
    Z = clustering.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(figsize=[30, 20])
    plt.clf()
    plt.imshow(
        Z,
        interpolation="nearest",
        extent=(xx.min(), xx.max(), yy.min(), yy.max()),
        cmap=plt.cm.Paired,
        aspect="auto",
        origin="lower",
    )

    plt.plot(reduced_data[:, 0], reduced_data[:, 1], "k.", markersize=5)
    # Plot the centroids as a white X
    # centroids = kmeans.cluster_centers_
    # plt.scatter(
    #    centroids[:, 0],
    #    centroids[:, 1],
    #    marker="x",
    #    s=169,
    #    linewidths=3,
    #    color="w",
    #    zorder=10,
    # )
    # plt.title("Clustering on the HITO software products (PCA-reduced data)"#"Centroids are marked with white cross")
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())

    # cursor = mplcursors.cursor(hover=True)
    # cursor.connect("add", lambda sel: sel.annotation.set_text(D[sel.target.index]["uri"]))

    # ax = plt.figure().add_subplot(111,autoscale_on=True)
    texts = []
    for i in range(len(D)):
        # a = plt.annotate(E[i]["label"], xy=reduced_data[i],xytext=randompoint(),textcoords="offset points", arrowprops=dict(facecolor='black', shrink=0.05, width=0.01, headwidth=0.01))
        a = plt.text(reduced_data[i][0], reduced_data[i][1], E[i]["label"])
        texts.append(a)

    if ADJUST_TEXT:
        adjust_text(texts, lim=10)

    plt.tight_layout()
    #plt.savefig("cluster-"+("classifiedonly-" if CLASSIFIED_ONLY else "")+NORM+".pdf", pad_inches=0)
    plt.savefig("cluster-bagofwords-"+("classifiedonly-" if CLASSIFIED_ONLY else "")+NORM+".pdf", pad_inches=0)
    plt.savefig("cluster.png", pad_inches=0)
    plt.show()


cluster()
