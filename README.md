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
LICENSE					| 					| CC0 license text

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
Requires [pySHACL](https://github.com/RDFLib/pySHACL) to be installed and available as `pyshacl`, for example via `pip install pyshacl`.

## Build

The `build` Linux shell script combines the ontology and all instances into a single file `./dist/hito.ttl`.
It requires the [Redland Raptor RDF syntax parsing and serializing utility (Rapper)](https://librdf.org/).

## Docker

Executes the `build` script and deploys it into the `/ontology/dist` volume.
Used in the [docker compose setup](https://github.com/hitontology/docker).

## License

HITO is dedicated to the public domain using Creative Commons Zero v1.0 Universal, see [LICENSE](LICENSE).
However, a small amount SNOMED CT terms are used with special permission from SNOMED in <http://hitontology.eu/ontology/Snomed>.
Licensee agrees and acknowledges that HITO may not own all right, title, and interest, in and to the Materials and that the Materials may contain and/or reference intellectual property owned by third parties (“Third Party IP”).
Acceptance of these License Terms does not grant Licensee any rights with respect to Third Party IP.
Licensee alone is responsible for identifying and obtaining any necessary licenses or authorizations to utilize Third Party IP in connection with the Materials or otherwise.
Any actions, claims or suits brought by a third party resulting from a breach of any Third Party IP right by the Licensee remains the Licensee’s liability.
