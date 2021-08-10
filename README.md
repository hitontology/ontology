# HITO—The Health IT Ontology

The source of truth for HITO, excluding software products and related attributes, whose source is the database initially filled by <https://github.com/hitontology/database>.

To create a combined file, execute the `combine` script, which creates `/tmp/hito-all.nt` and `/tmp/hito-all.ttl` (including possibly outdated software products).
This file can then be uploaded to the HITO SPARQL endpoint:

1. Login to <https://hitontology.eu/conductor/>
2. Delete graph http://hitontology.eu/ontology at Linked Data -> Graphs -> Graphs
3. Upload the file at Linked Data -> Quad Store Upload to the graph <http://hitontology.eu>

Warning: This is the old procedure and will not include updated software products from the database.

## Documentation
[Download widoco](https://github.com/dgarijo/Widoco/releases) and make it available as `widoco`, then run `./doc` under Linux to generate the ontology documentation.
Adapt the `doc` script accordingly on other operating systems.

## List of files

file			| source of truth	| description
--			| --			| --
bb.ttl			| csv2rdf	 	|
combine			| 			|
database.ttl		| database		|
dbpedia/os.nt		| dbpedia		| 	
dbpedia			| 			|
dbpedia.nt		| dbpedia		|
doc			| 			|
hito.ttl		| here			|
hl7ehrsfm.ttl		| csv2rdf		|
individual.ttl		| here			|
joshipacs.ttl		| csv2rdf		|
limes			| 			|
medfloss.ttl		| ?			|
prefix.ttl		| 			|
programminglibrary.ttl	| ? 			|
README.md		| 			|
snomed.ttl		| csv2rdf		|
sparql			| 			|
standard.ttl		| ?			|
swp.ttl			| ?			|
whodhi.ttl		| csv2rdf		|


## Import software products from the database
See <https://github.com/hitontology/database>.

## Import catalogues
See <https://github.com/hitontology/csv2rdf>.
