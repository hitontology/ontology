from collections import defaultdict
from rdflib import Graph
from sklearn.cluster import KMeans, DBSCAN, AffinityPropagation, AgglomerativeClustering
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import pairwise_distances,matthews_corrcoef
from sklearn import preprocessing
from scipy.cluster.hierarchy import dendrogram
import scipy.spatial.distance as distance

from sklearn.decomposition import PCA
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from adjustText import adjust_text
import mplcursors
import random
import math
import umap

import networkx as nx
import pydot
import graphviz

plt.style.use('dark_background')

# slow but nice
ADJUST_TEXT = True
NORMALIZE = False
NORM = "max" # l1, l2, max
CLASSIFIED_ONLY = True
BAG_OF_WORDS = False
HIERARCHICAL = True
MIN_TARGETS = 10
GRAPH = True

g = Graph()
HITO = "http://hitontology.eu/ontology/"
g.bind("hito", HITO)
# rdflib v5 as installed by pip doesn't support remote SPARQL querying.
# When rdflib v6 is officially released, this can be changed using the SERVICE keyword.
# The ontology/combine script needs to be executed first to created the file.
FILENAME = "/tmp/hito-all.nt"
g.parse(FILENAME, format="nt")

#?citation hito:fCitClassifiedAs|hito:efCitClassifiedAs ?target.
CLASSIFIED_ONLY_QUERY = """SELECT ?source (STR(SAMPLE(?label)) AS ?label) (GROUP_CONCAT(DISTINCT(?target); separator=" ") AS ?targets)
(GROUP_CONCAT(DISTINCT(STR(?ast)); separator="|") AS ?asts)
{
SERVICE <https://hitontology.eu/sparql>
{
?source   a hito:SoftwareProduct;
            rdfs:label ?label;
            hito:feature|hito:enterpriseFunction ?citation.
  ?citation (hito:fCitClassifiedAs|hito:efCitClassifiedAs)/(hito:subFeatureOf|hito:subFunctionOf)? ?target.

  OPTIONAL {?source hito:applicationSystem/hito:applicationSystemClassified/rdfs:label ?ast.}
} 
}
GROUP BY ?source ?label"""

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
ast_set = set()
ast_list = []

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
    global ast_set
    global ast_list
    for row in result:
        targets = row["targets"].split()
        if(len(targets)<MIN_TARGETS):
            continue
        print(row["source"],len(targets))
        D.append({"classifieds": targets}) # labels for one-hot encoding, dict vectorizer
        #D.append(str(row["targets"])) # bag of words
        #D.append(targets)
        #application system types are only used for coloring, as they would bias the result towards the exising classification
        row_asts = row["asts"].split("|")
        if(len(row_asts)>0): # SPARQL "None" bug(?) workaround
            if(row_asts[0]=="None"):
                row_asts=[]

        #print("asts of",row["source"],row_asts)
        ast_set.update(row_asts)
        E.append(
            {
                "uri": str(row["source"]),
                "label": ([row["label"].value][0]),
                "classifieds": row["targets"].split(),
                "asts": row_asts
            }
        )
        L.append([row["label"].value][0])
    print(len(D),"/",len(result),"products with number of features >",MIN_TARGETS)
    #print(E[0]["label"])
    #print(E)
    #print(D)
    if(BAG_OF_WORDS):
        vec = CountVectorizer() # bag of words
        data = vec.fit_transform(D).toarray()
    else:
        vec = DictVectorizer(sparse=False)
        data = vec.fit_transform(D)
    #print(data)
    ast_list = list(ast_set)
    #print(ast_list)
    return data

# use sklearn dict vectorizers and feature extraction
def clusterPlot(data):
    if(NORMALIZE):
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
    # The number of observations contained in each cluster level
    no_of_observations = np.arange(2, model.children_.shape[0]+2)
    # Create linkage matrix and then plot the dendrogram
    linkage_matrix = np.column_stack([model.children_, model.distances_, no_of_observations]).astype(float)
    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)
    return linkage_matrix

def label(i):
    if(i<len(L)):
        return L[i]
    return i

colors = ["red","green","blue","yellow","magenta","lightblue","turquoise","aqua","lightsalmon","chocolate","khaki","lavender","maroon","olive","gold","lemonchiffon","whitesmoke","lightgreen","crimson","yellowgreen","mistyrose","lightsteelblue"]
astcolors = dict()

def color(index):
    if(index>=len(E)):
        return "white"
    e = E[index]
    asts = e["asts"]
    if(len(asts)==0):
        return "white"
    ast = asts[0]
    astindex = ast_list.index(ast)
    #color = cm.get_cmap("Pastel1")(astindex) # does not accept this tuple format
    color = colors[astindex]
    global astcolors
    astcolors[ast]=color
    return color

# https://stackoverflow.com/questions/9838861/scipy-linkage-format
# https://datascience.stackexchange.com/questions/101854/how-to-visualize-a-hierarchical-clustering-as-a-tree-of-labelled-nodes-in-python
def showTree(linkage_matrix):
    G = nx.Graph()
    n = len(linkage_matrix)
    for i in range(n):
        row = linkage_matrix[i]
        G.add_node(label(int(row[0])),fillcolor=color(int(row[0])),style="filled")
        G.add_node(label(int(row[1])),fillcolor=color(int(row[1])),style="filled")
        G.add_edge(label(int(row[0])),label(n+i+1),len=1+0.1*(math.log(1+row[2])))
        G.add_edge(label(int(row[1])),label(n+i+1),len=1+0.1*(math.log(1+row[2])))
    for key,value in astcolors.items():
        G.add_node(key,fillcolor=value,style="filled")
    dot = nx.nx_pydot.to_pydot(G).to_string()
    dot = graphviz.Source(dot, engine='neato')
    dot.render(format='pdf',filename='tree')

def showGraph(distances):
    THRESHOLD = 0.5
    G = nx.Graph()
    n = len(distances)
    for i in range(n):
        for j in range(i+1,n):
            dist = distances[i][j]
            if(dist>THRESHOLD):
                continue
            print(dist,i,j,label(i),label(j))
            G.add_node(label(i),fillcolor=color(i),style="filled")
            G.add_node(label(j),fillcolor=color(j),style="filled")
            G.add_edge(label(i),label(j),weight=dist)
    #for key,value in astcolors.items():
    #    G.add_node(key,fillcolor=value,style="filled")
    dot = nx.nx_pydot.to_pydot(G).to_string()
    dot = graphviz.Source(dot, engine='neato')
    dot.render(format='pdf',filename='graph')

def clusterTree(data):
    N_CLUSTERS = 10
    clustering = AgglomerativeClustering(linkage="single", n_clusters=N_CLUSTERS, compute_distances=True, affinity="precomputed")
    distances = pairwise_distances(data,metric="dice")
    if(GRAPH):
        showGraph(distances)
        return
    distances = pairwise_distances(data,metric=lambda u,v: (1-matthews_corrcoef(u,v))/2) # invert similarity to distance and normalize [-1,1] to [0,1]
    clustering.fit(distances)
    #clustering = AgglomerativeClustering(linkage="average", n_clusters=N_CLUSTERS, compute_distances=True, affinity="l1")
    #clustering.fit(data)
    abbr = []
    for l in L:
        abbr.append(l[:16])
    clustering.labels = abbr
    paramStr  = ("BOW " if BAG_OF_WORDS else "") + ("classified only " if CLASSIFIED_ONLY else "") 
    #plt.title("Hierarchical Clustering " + paramStr)
    #plot_dendrogram(clustering, labels=clustering.labels_)
    linkage_matrix = plot_dendrogram(clustering, labels=abbr,show_leaf_counts=False)
    #print(linkage_matrix)
    showTree(linkage_matrix)
    #plt.show()

    

data = createData()
if(HIERARCHICAL):
    clusterTree(data)
else:
    clusterPlot(data)

print(astcolors)
