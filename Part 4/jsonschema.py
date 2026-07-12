"""
Minimal offline shim for the `jsonschema` package.

WHY THIS FILE EXISTS: this sandbox has no internet access, so `pip install
jsonschema` fails (no cached wheel available either). Rather than skip the
jsonschema.validate() requirement from the task spec, this file provides a
small drop-in replacement implementing exactly the subset of JSON Schema
functionality this project's schemas use: `type: object`, `properties`
with scalar types (string / number / boolean), `enum`, and `required`.

If you install the real `jsonschema` package (`pip install jsonschema`),
simply delete this file (or make sure it isn't on the Python path ahead of
the real package) - `from jsonschema import validate, ValidationError` will
then use the genuine library with zero code changes elsewhere, since this
shim exposes the same two names with matching call signatures/behaviour
for the schemas used here.
"""

_TYPE_MAP = {
    "string": str,
    "number": (int, float),
    "boolean": bool,
    "object": dict,
    "array": list,
}


class ValidationError(Exception):
    """Mirrors jsonschema.exceptions.ValidationError's .message attribute."""
    def __init__(self, message):
        super().__init__(message)
        self.message = message


def validate(instance, schema):
    """
    Minimal equivalent of jsonschema.validate(instance, schema) for schemas
    of the shape: {"type": "object", "properties": {...}, "required": [...]}
    where each property is {"type": "string"/"number"/"boolean", optionally "enum": [...]}.
    Raises ValidationError on the first problem found (matching jsonschema's
    fail-fast behaviour).
    """
    if schema.get("type") == "object" and not isinstance(instance, dict):
        raise ValidationError(f"{instance!r} is not of type 'object'")

    required = schema.get("required", [])
    for field in required:
        if field not in instance:
            raise ValidationError(f"{field!r} is a required property")

    properties = schema.get("properties", {})
    for field, value in instance.items():
        if field not in properties:
            continue  # additional properties allowed (default JSON Schema behaviour)
        prop_schema = properties[field]
        expected_type = prop_schema.get("type")
        if expected_type is not None:
            py_type = _TYPE_MAP.get(expected_type)
            # bool is a subclass of int in Python - guard against numbers
            # accidentally validating as booleans and vice versa
            if expected_type == "number" and isinstance(value, bool):
                raise ValidationError(f"{value!r} is not of type 'number'")
            if expected_type == "boolean" and not isinstance(value, bool):
                raise ValidationError(f"{value!r} is not of type 'boolean'")
            if py_type is not None and not isinstance(value, py_type):
                raise ValidationError(f"{value!r} is not of type {expected_type!r}")
        if "enum" in prop_schema and value not in prop_schema["enum"]:
            raise ValidationError(f"{value!r} is not one of {prop_schema['enum']}")

    return True
