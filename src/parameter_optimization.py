def optimize_parameters(strategy, param_grid, X_train=None, y_train=None):
    """
    Optimize parameters for a given strategy using grid search.
    """
    if X_train is None or y_train is None:
        raise ValueError("X_train and y_train must be provided for parameter optimization.")

    from sklearn.model_selection import GridSearchCV

    # Ensure `strategy` is a valid estimator compatible with GridSearchCV
    if not hasattr(strategy, 'fit'):
        raise ValueError("The provided strategy must have a 'fit' method to be compatible with GridSearchCV.")
    grid_search = GridSearchCV(strategy, param_grid, cv=2, scoring='accuracy')  # Use cv=2
    grid_search.fit(X_train, y_train)

    return grid_search.best_params_
