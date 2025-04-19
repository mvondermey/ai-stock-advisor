from sklearn.model_selection import GridSearchCV

def optimize_parameters(strategy, param_grid, X_train, y_train, cv=3):
    """Optimize strategy parameters using GridSearchCV with cross-validation."""
    grid_search = GridSearchCV(estimator=strategy, param_grid=param_grid, cv=cv, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    return grid_search.best_params_
