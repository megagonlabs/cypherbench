import neo4j
from cypherbench.neo4j_connector import Neo4jConnector


def executable(pred_cypher: str,
               target_cypher: str,
               neo4j_connector: Neo4jConnector,
               timeout: int = 120) -> float:
    """Whether the predicted Cypher query is executable"""
    try:
        neo4j_connector.run_query(pred_cypher, timeout=timeout)
    except (
            neo4j.exceptions.CypherSyntaxError,
            neo4j.exceptions.DatabaseError,
            neo4j.exceptions.CypherTypeError,
            neo4j.exceptions.ClientError,
    ) as e:
        return 0.0
    except Exception as e:
        print(f"Warning: Exception {e} occurred while executing the predicted Cypher query {pred_cypher}")
        return 0.0

    return 1.0
