# cypherbench

[![](https://img.shields.io/badge/license-apache2.0-green.svg)](LICENSE) 
[![](https://img.shields.io/badge/🤗-HuggingFace-red.svg)](https://huggingface.co/datasets/megagonlabs/cypherbench)
[![](https://img.shields.io/badge/paper-arxiv-yellow.svg)](https://arxiv.org/pdf/2412.18702)

### Demo graphs
[nba](https://browser.neo4j.io/?dbms=neo4j%2Bs%3A%2F%2Fneo4j@36535562.databases.neo4j.io&db=neo4j)
(password: `cypherbench`)

Sample queries:
```cypher
MATCH (n:Player {name: 'LeBron James'})-[r]-(m) RETURN *
```
```cypher
MATCH (t:Team)-[]-(d:Division)-[]-(c:Conference) RETURN *
```

### Release Plan

- [x] text2cypher tasks
- [x] 11 property graphs
- [ ] EX/PSJS implementation and evaluation scripts
- [ ] Wikidata RDF-to-property-graph engine
- [ ] Text2cypher task generation pipeline
