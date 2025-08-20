import kuzu


class KuzuDatabaseManager:
    """Manages Kuzu database connection and schema retrieval."""

    def __init__(self, db_path: str = "ldbc_1.kuzu"):
        self.db_path = db_path
        self.db = kuzu.Database(db_path, read_only=True)
        self.conn = kuzu.Connection(self.db)

    @property
    def get_schema_dict(self) -> dict[str, list[dict]]:
        response = self.conn.execute("CALL SHOW_TABLES() WHERE type = 'NODE' RETURN *;")
        nodes = [row[1] for row in response]  # type: ignore
        response = self.conn.execute("CALL SHOW_TABLES() WHERE type = 'REL' RETURN *;")
        rel_tables = [row[1] for row in response]  # type: ignore
        relationships = []
        for tbl_name in rel_tables:
            response = self.conn.execute(f"CALL SHOW_CONNECTION('{tbl_name}') RETURN *;")
            for row in response:
                relationships.append({"name": tbl_name, "from": row[0], "to": row[1]})  # type: ignore
        schema = {"nodes": [], "edges": []}

        for node in nodes:
            node_schema = {"label": node, "properties": []}
            node_properties = self.conn.execute(f"CALL TABLE_INFO('{node}') RETURN *;")
            for row in node_properties:  # type: ignore
                node_schema["properties"].append({"name": row[1], "type": row[2]})  # type: ignore
            schema["nodes"].append(node_schema)

        for rel in relationships:
            edge = {
                "label": rel["name"],
                "from": rel["from"],
                "to": rel["to"],
                "properties": [],
            }
            rel_properties = self.conn.execute(f"""CALL TABLE_INFO('{rel["name"]}') RETURN *;""")
            for row in rel_properties:  # type: ignore
                edge["properties"].append({"name": row[1], "type": row[2]})  # type: ignore
            schema["edges"].append(edge)
        return schema
