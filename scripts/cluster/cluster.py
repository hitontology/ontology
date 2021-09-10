from collections import defaultdict
from rdflib import Graph
from sklearn.cluster import KMeans, DBSCAN, AffinityPropagation, AgglomerativeClustering
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
from scipy.cluster.hierarchy import dendrogram

from sklearn.decomposition import PCA
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from adjustText import adjust_text
import mplcursors
import random
import math
import umap

plt.style.use('dark_background')

# slow but nice
ADJUST_TEXT = True
NORM = "max" # l1, l2, max
CLASSIFIED_ONLY = True
BAG_OF_WORDS = True
HIERARCHICAL = True

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

DEFAULT_QUERY = """PREFIX :<http://hitontology.eu/ontology/>
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

BAG_OF_WORDS_QUERY = """PREFIX :<http://hitontology.eu/ontology/>
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

D = []
E = []
L = []

def createData():
    #result = g.query(CLASSIFIED_ONLY_QUERY if CLASSIFIED_ONLY else DEFAULT_QUERY)
    if(CLASSIFIED_ONLY):
        if(BAG_OF_WORDS):
            QUERY = CLASSIFIED_ONLY_BAG_OF_WORDS
        else:
            QUERY = CLASSIFIED_ONLY_QUERY
    else:
        if(BAG_OF_WORDS):
            QUERY = BAG_OF_WORDS_QUERY
        else:
            QUERY = DEFAULT_QUERY
    result = g.query(QUERY)
    print(len(result))
    global D
    global E
    global L 
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
        L.append([row["label"].value][0])
    print(E[0]["label"])
    #print(D)
    #vec = DictVectorizer(sparse=False)
    vec = CountVectorizer()

    #data = vec.fit_transform(D)
    data = vec.fit_transform(D).toarray()
    print(data)
    return data

# use sklearn dict vectorizers and feature extraction
def clusterPlot(data):
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
        a = plt.text(reduced_data[i][0], reduced_data[i][1], E[i]["label"])
        texts.append(a)

    if ADJUST_TEXT:
        adjust_text(texts, lim=10)

    plt.tight_layout()
    #plt.savefig("cluster-"+("classifiedonly-" if CLASSIFIED_ONLY else "")+NORM+".pdf", pad_inches=0)
    plt.savefig("cluster-bagofwords-"+("classifiedonly-" if CLASSIFIED_ONLY else "")+NORM+".pdf", pad_inches=0)
    plt.savefig("cluster.png", pad_inches=0)
    plt.show()

# https://github.com/scikit-learn/scikit-learn/blob/70cf4a676caa2d2dad2e3f6e4478d64bcb0506f7/examples/cluster/plot_hierarchical_clustering_dendrogram.py
def plot_dendrogram(model, **kwargs):
    # Children of hierarchical clustering
    children = model.children_

    # Distances between each pair of children
    # Since we don't have this information, we can use a uniform one for plotting
    distance = np.arange(children.shape[0])

    # The number of observations contained in each cluster level
    no_of_observations = np.arange(2, children.shape[0]+2)

    # Create linkage matrix and then plot the dendrogram
    linkage_matrix = np.column_stack([children, distance, no_of_observations]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


def clusterTree(data):
    N_CLUSTERS = 10
    print("todo: hierarchical clustering")
    clustering = AgglomerativeClustering(linkage="single", n_clusters=N_CLUSTERS)
    clustering.fit(data)
    abbr = []
    for l in L:
        abbr.append(l[:16])
    clustering.labels = abbr
    #print(clustering.linkage_matrix)
    paramStr  = ("BOW " if BAG_OF_WORDS else "") + ("classified only " if CLASSIFIED_ONLY else "") 
    #plt.title("Hierarchical Clustering " + paramStr)
    #plot_dendrogram(clustering, labels=clustering.labels_)
    plot_dendrogram(clustering, labels=abbr,show_leaf_counts=False)
    plt.show()


data = createData()
if(HIERARCHICAL):
    clusterTree(data)
else:
    clusterPlot(data)
