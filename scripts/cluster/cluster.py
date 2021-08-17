from collections import defaultdict
from rdflib import Graph
from sklearn.cluster import KMeans
from sklearn.feature_extraction import DictVectorizer

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
def vectorize():
    result = g.query(QUERY)
    print(len(result))
    D = []
    for row in result:
        D.append({"uri": str(row["source"]), "classifieds": row["targets"].split()})
    vec = DictVectorizer()
    X = vec.fit_transform(D)
    print(X)


vectorize()
