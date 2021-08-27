from collections import defaultdict
from rdflib import Graph

import numpy as np
import random
import math

# slow but nice
CLASSIFIED_ONLY = True

g = Graph()
HITO = "http://hitontology.eu/ontology/"
g.bind("hito", HITO)
# rdflib v5 as installed by pip doesn't support remote SPARQL querying.
# When rdflib v6 is officially release this can be changed using the SERVICE keyword.
# The ontology/combine script needs to be executed first to created the file.
FILENAME = "/tmp/hito-all.nt"
g.parse(FILENAME, format="nt")

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

allTargets = set()

def products():
    result = g.query(QUERY)
    S = []
    for row in result:
        targets = row["targets"].split()
        S.append(
            {
                "uri": str(row["source"]),
                "label": ([row["label"].value][0]),
                "targets": targets,
            }
        )
        global allTargets
        allTargets = allTargets.union(targets)
    return S

P = products()
N = len(allTargets)
nswdMax = (math.log(math.floor(N/2)+1))/(math.log(N)-math.log(math.ceil(N/2)))

def nswd(x,y):
    #print(x,y,N)
    logx = math.log(len(x["targets"]))
    logy = math.log(len(y["targets"]))
    sect = set(x["targets"]).intersection(y["targets"])
    if(len(sect)==0):
        #print("no intersection between",x["uri"],"and",y["uri"],"returning",nswdMax)
        return nswdMax
    #print(sect)
    logxy = math.log(len(sect))
    #print(logx,logy,logxy)
    dist = (max(logx,logy)-logxy)/(math.log(N)-min(logx,logy))
    #if(dist==0):
    #    print("Dist=0 between",x,y,"sect",sect)
    return dist

scores = []

def nswdAll(S):
    print(len(S))
    for i in range(1,len(S)):
        for j in range(i):
            dist = nswd(S[i],S[j])
            #print(i,j,dist))
            scores.append({"x":S[i]["uri"],"y":S[j]["uri"],"dist":dist})
            
nswdAll(products())
scores = sorted(scores, key=lambda x: x["dist"])
print(scores)
