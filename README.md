# HITO—The Health IT Ontology

The source of truth for HITO, excluding software products and related attributes, whose source is the database initially filled by <https://github.com/hitontology/database>.

To create a combined file, execute the `combine` script, which creates `/tmp/hito-all.nt` and `/tmp/hito-all.ttl` (including possibly outdated software products).
This file can then be uploaded to the HITO SPARQL endpoint:

1. Login to <https://hitontology.eu/conductor/>
2. Delete graph http://hitontology.eu/ontology at Linked Data -> Graphs -> Graphs
3. Upload the file at Linked Data -> Quad Store Upload

Warning: This is the old procedure and will not include updated software products from the database.
