"""
Utility functions to convert dataclasses to dictionaries, handling special types like numpy arrays, PyTorch tensors, and enums.
This is useful for serializing configurations and results.
"""
from dataclasses import fields
from typing import Optional, Set

from torch import dtype, Tensor
from numpy import ndarray

def convert_special_types(value):
    """Convert special types to JSON-serializable formats"""
    if isinstance(value, dtype):
        return str(value)
    elif isinstance(value, ndarray):
        return value.tolist()
    elif isinstance(value, Tensor):
        return value.tolist()
    elif hasattr(value, '__name__'):  # Handle function types
        return value.__name__
    elif hasattr(value, 'name'):  # Handle enum types
        return value.name
    elif hasattr(value, 'value'):  # Handle enum values
        return value.value
    else:
        return value

def dataclass_to_dict(obj, exclude_fields: Optional[Set[str]] = None) -> dict:
    """
    Recursively convert a dataclass to a dictionary, handling special types
    and nested dataclasses. If a nested object has a to_dict() method, it will
    be called with the exclude_fields parameter.
    
    Args:
        obj: The dataclass instance to convert
        exclude_fields: Set of field names to exclude from conversion
        
    Returns:
        dict: Dictionary representation with all special types converted
    """
    if exclude_fields is None:
        exclude_fields = set()
    
    if not hasattr(obj, '__dataclass_fields__'):
        # Not a dataclass, try to convert special types
        return convert_special_types(obj)
    
    result = {}
    
    for field_info in fields(obj):
        field_name = field_info.name
        
        # Skip excluded fields
        if field_name in exclude_fields:
            continue
            
        field_value = getattr(obj, field_name)
        
        if field_value is None:
            result[field_name] = None
        elif hasattr(field_value, 'to_dict') and callable(getattr(field_value, 'to_dict')):
            # Object has a to_dict method, use it
            try:
                # Try calling with exclude_fields parameter
                result[field_name] = field_value.to_dict(exclude_fields=exclude_fields)
            except TypeError:
                # If it doesn't accept exclude_fields, call without parameters
                try:
                    result[field_name] = field_value.to_dict()
                except Exception:
                    # Fallback to recursive conversion
                    result[field_name] = dataclass_to_dict(field_value, exclude_fields=exclude_fields)
        elif hasattr(field_value, '__dataclass_fields__'):
            # Nested dataclass without to_dict method
            result[field_name] = dataclass_to_dict(field_value, exclude_fields=exclude_fields)
        elif isinstance(field_value, (list, tuple)):
            # Handle lists/tuples that might contain dataclasses or special types
            converted_list = []
            for item in field_value:
                if hasattr(item, 'to_dict') and callable(getattr(item, 'to_dict')):
                    # Item has to_dict method
                    try:
                        converted_list.append(item.to_dict(exclude_fields=exclude_fields))
                    except TypeError:
                        try:
                            converted_list.append(item.to_dict())
                        except Exception:
                            converted_list.append(dataclass_to_dict(item, exclude_fields=exclude_fields))
                elif hasattr(item, '__dataclass_fields__'):
                    converted_list.append(dataclass_to_dict(item, exclude_fields=exclude_fields))
                else:
                    converted_list.append(convert_special_types(item))
            result[field_name] = converted_list
        elif isinstance(field_value, dict):
            # Handle dictionaries
            converted_dict = {}
            for key, val in field_value.items():
                if hasattr(val, 'to_dict') and callable(getattr(val, 'to_dict')):
                    # Value has to_dict method
                    try:
                        converted_dict[key] = val.to_dict(exclude_fields=exclude_fields)
                    except TypeError:
                        try:
                            converted_dict[key] = val.to_dict()
                        except Exception:
                            converted_dict[key] = dataclass_to_dict(val, exclude_fields=exclude_fields)
                elif hasattr(val, '__dataclass_fields__'):
                    converted_dict[key] = dataclass_to_dict(val, exclude_fields=exclude_fields)
                else:
                    converted_dict[key] = convert_special_types(val)
            result[field_name] = converted_dict
        else:
            # Regular field, convert special types
            result[field_name] = convert_special_types(field_value)
    
    return result