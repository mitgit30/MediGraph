CREATE CONSTRAINT entity_id_unique IF NOT EXISTS
FOR (n:Entity)
REQUIRE n.id IS UNIQUE;

CREATE INDEX entity_name_index IF NOT EXISTS
FOR (n:Entity)
ON (n.name);

CREATE INDEX entity_type_index IF NOT EXISTS
FOR (n:Entity)
ON (n.type);

CREATE INDEX relation_name_index IF NOT EXISTS
FOR ()-[r:RELATED_TO]-()
ON (r.relation_name);
