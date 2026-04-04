from __future__ import annotations

import os
from functools import wraps
from typing import Any, Dict, Tuple


def _silence_lightgbm_warnings() -> None:
    """Suppress noisy LightGBM alias warnings while keeping model behavior unchanged."""
    try:
        import lightgbm as lgb

        class _SilentLogger:
            def info(self, msg):
                pass

            def warning(self, msg):
                pass

            def error(self, msg):
                pass

        lgb.register_logger(_SilentLogger())
    except Exception:
        pass


_silence_lightgbm_warnings()


def release_runtime_memory() -> None:
    """Release Python and CUDA caches after heavy training work."""
    try:
        import gc

        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    except Exception:
        pass


def cleanup_training_memory(func):
    """Decorator to guarantee memory cleanup around training methods."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        finally:
            release_runtime_memory()

    return wrapper


def configure_catboost_cpu_continuation(model: Any) -> None:
    """CatBoost continuation only works reliably on CPU."""
    if not hasattr(model, "_init_params"):
        return
    model._init_params["task_type"] = "CPU"
    model._init_params["thread_count"] = 1
    model._init_params["allow_writing_files"] = False


def catboost_has_trained_trees(model: Any) -> bool:
    """Continuation requires an already-fitted CatBoost model."""
    try:
        # CatBoost >= 1.2 uses tree_count_ property
        if hasattr(model, "tree_count_"):
            return model.tree_count_ > 0
        # Older CatBoost versions use get_tree_count() method
        if hasattr(model, "get_tree_count"):
            return model.get_tree_count() > 0
        # Fallback: check is_fitted()
        if hasattr(model, "is_fitted"):
            return model.is_fitted()
        return False
    except Exception:
        return False


def ensure_catboost_cpu_metadata(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Mark saved checkpoints as CPU CatBoost lineage."""
    metadata = dict(payload.get("metadata") or {})
    metadata["catboost_backend"] = "cpu"
    payload["metadata"] = metadata
    return payload


def xgboost_has_trained_booster(model: Any) -> bool:
    """Check whether an XGBoost sklearn wrapper has fitted trees."""
    try:
        booster = model.get_booster()
        return booster is not None and booster.num_boosted_rounds() > 0
    except Exception:
        return False


def lightgbm_has_trained_booster(model: Any) -> bool:
    """Check whether a LightGBM sklearn wrapper has a fitted booster."""
    try:
        booster = getattr(model, "booster_", None)
        return booster is not None and booster.num_trees() > 0
    except Exception:
        return False


def _native_model_artifact_path(path: str, model_name: str) -> str:
    """Return the sidecar artifact path for one ensemble member."""
    suffixes = {
        "XGBoost": ".xgboost.json",
        "LightGBM": ".lightgbm.txt",
        "CatBoost": ".catboost.cbm",
    }
    return f"{os.fspath(path)}{suffixes[model_name]}"


def _remove_native_model_artifact(path: str, model_name: str) -> None:
    """Delete a stale sidecar artifact if it exists."""
    artifact_path = _native_model_artifact_path(path, model_name)
    if os.path.exists(artifact_path):
        try:
            os.remove(artifact_path)
        except OSError:
            pass


def _infer_native_model_name(model: Any) -> str | None:
    """Best-effort mapping from model instance to ensemble member name."""
    module_name = getattr(type(model), "__module__", "")
    class_name = getattr(type(model), "__name__", "")
    qualified_name = f"{module_name}.{class_name}".lower()
    if "xgboost" in qualified_name:
        return "XGBoost"
    if "lightgbm" in qualified_name:
        return "LightGBM"
    if "catboost" in qualified_name:
        return "CatBoost"
    return None


def _extract_native_models(payload_or_model: Any) -> Dict[str, Any]:
    """Return the native-model members that should use sidecar persistence."""
    if isinstance(payload_or_model, dict) and isinstance(payload_or_model.get("all_models"), dict):
        return dict(payload_or_model.get("all_models") or {})

    model_name = _infer_native_model_name(payload_or_model)
    if model_name is None:
        return {}
    return {model_name: payload_or_model}


def save_native_model_artifacts(payload_or_model: Any, path: str) -> None:
    """Persist fitted XGBoost/LightGBM/CatBoost models using native formats."""
    models = _extract_native_models(payload_or_model)
    for model_name in ("XGBoost", "LightGBM", "CatBoost"):
        model = models.get(model_name)
        try:
            if model_name == "XGBoost":
                if model is not None and xgboost_has_trained_booster(model):
                    model.save_model(_native_model_artifact_path(path, model_name))
                else:
                    _remove_native_model_artifact(path, model_name)
            elif model_name == "LightGBM":
                if model is not None and lightgbm_has_trained_booster(model):
                    model.booster_.save_model(_native_model_artifact_path(path, model_name))
                else:
                    _remove_native_model_artifact(path, model_name)
            elif model_name == "CatBoost":
                if model is not None and catboost_has_trained_trees(model):
                    model.save_model(_native_model_artifact_path(path, model_name), format="cbm")
                else:
                    _remove_native_model_artifact(path, model_name)
        except Exception:
            pass


def _restore_single_native_model(model_name: str, model: Any, path: str, is_classifier: bool = False) -> Any:
    """Load one model sidecar back into an sklearn wrapper (creating one if needed)."""
    artifact_path = _native_model_artifact_path(path, model_name)
    if not os.path.exists(artifact_path):
        return model

    try:
        if model_name == "XGBoost":
            import xgboost as xgb
            if model is None:
                model = xgb.XGBClassifier() if is_classifier else xgb.XGBRegressor()
            model.load_model(artifact_path)
            return model

        if model_name == "LightGBM":
            import lightgbm as lgb
            booster = lgb.Booster(model_file=artifact_path)
            if model is None:
                model = lgb.LGBMClassifier() if is_classifier else lgb.LGBMRegressor()
            model._Booster = booster
            model.fitted_ = True
            num_features = booster.num_feature()
            model._n_features = num_features
            model._n_features_in = num_features
            return model

        if model_name == "CatBoost":
            import catboost as cb
            if model is None:
                model = cb.CatBoostClassifier() if is_classifier else cb.CatBoostRegressor()
            model.load_model(artifact_path, format="cbm")
            return model
    except Exception:
        return model

    return model


def restore_native_model_artifacts(payload_or_model: Any, path: str, is_classifier: bool = False) -> Any:
    """Restore any native model sidecars that exist for the saved checkpoint.

    Args:
        payload_or_model: The loaded checkpoint data
        path: Path to the checkpoint file (sidecars are path + suffix)
        is_classifier: If True, create Classifier models; else Regressor models
    """
    if isinstance(payload_or_model, dict) and isinstance(payload_or_model.get("all_models"), dict):
        restored_payload = dict(payload_or_model)
        restored_models = dict(restored_payload.get("all_models") or {})

        for model_name in ("XGBoost", "LightGBM", "CatBoost"):
            original_model = restored_models.get(model_name)
            restored_model = _restore_single_native_model(model_name, original_model, path, is_classifier)
            if restored_model is not None:
                restored_models[model_name] = restored_model

        restored_payload["all_models"] = restored_models

        # Update model/best_model references based on best_name
        best_name = restored_payload.get("best_name")
        if best_name and best_name in restored_models and restored_models[best_name] is not None:
            restored_payload["model"] = restored_models[best_name]
            restored_payload["best_model"] = restored_models[best_name]

        return restored_payload

    model_name = _infer_native_model_name(payload_or_model)
    if model_name is None:
        return payload_or_model
    return _restore_single_native_model(model_name, payload_or_model, path, is_classifier)


def reset_legacy_catboost_member(payload: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    """Remove only CatBoost from older checkpoints so it can restart cleanly on CPU."""
    if not isinstance(payload, dict) or "all_models" not in payload:
        return payload, False

    metadata = dict(payload.get("metadata") or {})
    if metadata.get("catboost_backend") == "cpu":
        return payload, False

    all_models = payload.get("all_models")
    if not isinstance(all_models, dict) or "CatBoost" not in all_models:
        return payload, False

    sanitized = dict(payload)
    sanitized_models = dict(all_models)
    removed_model = sanitized_models.pop("CatBoost", None)
    sanitized["all_models"] = sanitized_models

    all_scores = sanitized.get("all_scores")
    if isinstance(all_scores, dict):
        sanitized_scores = dict(all_scores)
        sanitized_scores.pop("CatBoost", None)
        sanitized["all_scores"] = sanitized_scores

    best_name = sanitized.get("best_name")
    current_model = sanitized.get("model", sanitized.get("best_model"))
    if best_name == "CatBoost" or current_model is removed_model:
        remaining_scores = sanitized.get("all_scores", {})
        if remaining_scores:
            replacement_name = max(remaining_scores, key=remaining_scores.get)
            replacement_model = sanitized_models.get(replacement_name)
            sanitized["best_name"] = replacement_name
            if "best_score" in sanitized:
                sanitized["best_score"] = remaining_scores[replacement_name]
            if "model" in sanitized:
                sanitized["model"] = replacement_model
            if "best_model" in sanitized:
                sanitized["best_model"] = replacement_model
        else:
            sanitized["best_name"] = None
            if "best_score" in sanitized:
                sanitized["best_score"] = 0.0
            if "model" in sanitized:
                sanitized["model"] = None
            if "best_model" in sanitized:
                sanitized["best_model"] = None

    metadata["catboost_backend"] = "reset_pending_cpu"
    sanitized["metadata"] = metadata
    return sanitized, True
