from sklearn.model_selection import GridSearchCV

def optimize_parameters(strategy, param_grid, X_train, y_train, cv=3, scoring='neg_mean_squared_error'):
    """
    Optimize parameters for the given strategy using GridSearchCV.
    :param strategy: The trading strategy to optimize.
    :param param_grid: Dictionary of parameters to search.
    :param X_train: Training features.
    :param y_train: Training target.
    :param cv: Number of cross-validation folds.
    :param scoring: Scoring metric for optimization.
    :return: Best parameters found by GridSearchCV.
    """
    grid_search = GridSearchCV(
        estimator=strategy,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,  # Use the scoring metric passed to the function
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    return grid_search.best_params_
