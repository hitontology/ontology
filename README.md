# HITO—The Health IT Ontology

The source of truth for HITO, excluding software products and related attributes, whose source is the database initially filled by <https://github.com/hitontology/database>, and catalogues, which are created in spreadsheets and transformed using <https://github.com/hitontology/csv2rdf>.

To create a combined file, execute the `build` script, which creates `/dist/hito.ttl` and `/dist/hito.nt`.
This file can then be uploaded to the HITO SPARQL endpoint:

1. Login to <https://hitontology.eu/conductor/>
2. Delete graph http://hitontology.eu/ontology at Linked Data -> Graphs -> Graphs
3. Upload the file at Linked Data -> Quad Store Upload to the graph <http://hitontology.eu>

Warning: This will not include updated software products from the database.

## Documentation
[Download WIDOCO](https://github.com/dgarijo/Widoco/releases) and make it available as `widoco`, then run `./doc` under Linux to generate the ontology documentation.
Adapt the `doc` script accordingly on other operating systems.

## List of files

file					| source of truth	| description
--						| --				| --
ontology.ttl			| here				| The HITO ontology
database.ttl			| here				| instances of hito:DatabaseSystem
programminglibrary.ttl	| here 				| instances of hito:ProgrammingLibrary
standard.ttl			| here				| instances of hito:Interoperability
individual.ttl			| here				| journal descriptions
medfloss.ttl			| here				| <https://www.medfloss.org> sources
shacl.ttl				| here				| SHACL shapes for validation
hl7ehrsfm.ttl			| csv2rdf			| HL7 EHR-S FM catalogues
bb.ttl					| csv2rdf	 		| "Blue Book" catalogues
joshipacs.ttl			| csv2rdf			| PACS feature catalogue
snomed.ttl				| csv2rdf			| SNOMED catalogues
whodhi.ttl				| csv2rdf			| WHO DHI catalogues
swp.ttl					| database			| software descriptions
dist/dbpedia.ttl		| DBpedia			| (programming) languages and operating systems from DBpedia
dist/swo.ttl			| SWO				| licenses from the Software Ontology (SWO)
build					| 					| combine all HITO files into one
prefix.ttl				| 					| RDF namespace prefixes
dist/hito.ttl			| 					| output of build
scripts/doc				| 					| ontology documentation build script
scripts/sparql			| 					| SPARQL queries
scripts/sparql/swo.sparql|					| Query on SWO to create dist/swo.ttl
scripts/limes			| 					| generate interlinks

The relevant DBpedia and SWO resources are put under version control in the dist folder to keep the referential integrity of the database.
You do not need to regenerate them but if you do, using `/scripts/dbpedia` and /scripts/sparql

## Import software products from the database
See <https://github.com/hitontology/database>.

## Import catalogues
See <https://github.com/hitontology/csv2rdf>.

## Validation
While syntactic validation can be done using `rapper -c`, syntactically correct data may still violate the HITO ontology or the HITO diagram including cardinalities.

### Quality Check
The [HITO Quality Check Tool](https://hitontology.eu/qualitycheck/) contains custom error and warning categories based on SPARQL queries.

### SHACL
SHACL shapes including cardinalities for closed-world validation are included in `shacl.ttl`.
Validate using `scripts/shacl`.
Requires [pySHACL](https://github.com/RDFLib/pySHACL) to be installed and available as `pyshacl`.
