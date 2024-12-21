def validate_policy_dimensions(policy_dict, allowed_dims):
    """
    Ensures that each dimension in policy_dict is in allowed_dims.
    Raises an error or logs a warning otherwise.
    """
    for dim in policy_dict.keys():
        if dim not in allowed_dims:
            raise ValueError(
                f"Dimension '{dim}' not in allowed: {allowed_dims}"
            )
