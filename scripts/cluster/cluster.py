from collections import defaultdict
from rdflib import Graph
from sklearn.cluster import KMeans

g = Graph()
HITO = "http://hitontology.eu/ontology/"
g.bind("hito", HITO)
FILENAME = "/tmp/hito-all.nt"
#with open(FILENAME,"r") as f:
g.parse(FILENAME, format="nt")

QUERY = """SELECT DISTINCT ?source ?target {
  ?source   a hito:SoftwareProduct;
            ?p ?citation.
  ?citation ?q ?target.

 ?p rdfs:subPropertyOf hito:citation.
 ?q rdfs:subPropertyOf hito:classified.
}"""

result = g.query(QUERY)

m = defaultdict(set)
targets = set()
print(len(result))

for row in result:
    targets.add(row.target)

targets = list(targets)
index = dict()

for i in range(len(targets)):
        index[targets[i]]=i

for row in result:
    m[row.source].add(index[row.target])

print(m)
