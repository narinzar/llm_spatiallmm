# custom_gradio_utils.py

def json_schema_to_python_type(schema, defs=None):
    if schema is True or schema is False:
        return "bool"
    if "type" not in schema:
        return "Any"
    type_ = schema["type"]
    if isinstance(type_, list):
        return ", ".join(json_schema_to_python_type({"type": t}, defs) for t in type_)
    if type_ == "array":
        return f"List[{json_schema_to_python_type(schema['items'], defs)}]"
    if type_ == "object":
        if "additionalProperties" in schema:
            return f"Dict[str, {json_schema_to_python_type(schema['additionalProperties'], defs)}]"
        else:
            des = [
                f"{n}: {json_schema_to_python_type(v, defs)}{get_desc(v)}"
                for n, v in schema.get("properties", {}).items()
            ]
            return "TypedDict(\"_Anonymous\", {" + ", ".join(des) + "})"
    if type_ == "string":
        if "enum" in schema:
            return "Literal[" + ", ".join(f'"{e}"' for e in schema["enum"]) + "]"
        return "str"
    if type_ == "integer":
        return "int"
    if type_ == "number":
        return "float"
    if type_ == "boolean":
        return "bool"
    if type_ == "null":
        return "None"
    if "$ref" in schema:
        path = schema["$ref"].split("/")
        return path[-1] if defs is None else defs[path[-1]]["x-typename"]
    return "Any"

def get_desc(schema):
    return ""
