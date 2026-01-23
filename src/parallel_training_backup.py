# This is a backup of the current function before fixing
def universal_model_worker_backup(task: Dict) -> Dict:
    """
    Worker function that trains one model for one task.
    
    Args:
        task: Dict with task_type ('ticker' or 'ai_portfolio') and training parameters
    
    Returns:
        Dict with training results
    """
    global _GPU_SEMAPHORE
    
    task_type = task.get('task_type')
    model_type = task.get('model_type')
    
    # ============================================
    # TICKER MODEL TRAINING
    # ============================================
    if task_type == 'ticker':
        ticker = task.get('ticker')
        df_train_period = task.get('df_train_period')
        class_horizon = task.get('class_horizon')
        feature_set = task.get('feature_set')
        
        try:
            # Acquire GPU semaphore if this is a GPU model
            gpu_acquired = False
            if not FORCE_CPU and model_type in ['LSTM', 'TCN', 'GRU', 'XGBoost']:
                if _GPU_SEMAPHORE is not None:
                    _GPU_SEMAPHORE.acquire()
                    gpu_acquired = True
            
            # Prepare training data
            df_train, actual_feature_set = fetch_training_data(
                ticker, df_train_period
            )
            
            if df_train.empty or len(df_train) < 50:
                return {
                    'task_type': 'ticker',
                    'ticker': ticker,
                    'model_type': model_type,
                    'status': 'skipped',
                    'reason': 'insufficient_data'
                }
            
            # Train the model
            result = train_single_model_type(
                df_train=df_train,
                ticker=ticker,
                model_type=model_type,
                class_horizon=class_horizon,
                feature_set=actual_feature_set
            )
            
            if result is None:
                return {
                    'task_type': 'ticker',
                    'ticker': ticker,
                    'model_type': model_type,
                    'status': 'failed',
                    'reason': 'training_failed'
                }
            
            # Save model to disk
            models_dir = Path("logs/models")
            _ensure_dir(models_dir)
            
            model_path = models_dir / f"{ticker}_{model_type}_temp_model.joblib"
            scaler_path = models_dir / f"{ticker}_{model_type}_temp_scaler.joblib"
            
            joblib.dump(result['model'], model_path)
            joblib.dump(result['scaler'], scaler_path)
            
            # Save y_scaler if present
            y_scaler_path = None
            if result.get('y_scaler') is not None:
                y_scaler_path = models_dir / f"{ticker}_{model_type}_temp_y_scaler.joblib"
                joblib.dump(result['y_scaler'], y_scaler_path)
            
            return {
                'task_type': 'ticker',
                'ticker': ticker,
                'model_type': model_type,
                'status': 'success',
                'mse': result['mse'],
                'model_path': str(model_path),
                'scaler_path': str(scaler_path),
                'y_scaler_path': str(y_scaler_path) if y_scaler_path else None
            }
        
        except Exception as e:
            import traceback
            error_msg = str(e)
            tb_str = traceback.format_exc()
            print(f"  ❌ ERROR {ticker} {model_type}: {error_msg[:100]}")
            print(f"     Traceback: {tb_str[-500:]}")  # Last 500 chars of traceback
            return {
                'task_type': 'ticker',
                'ticker': ticker,
                'model_type': model_type,
                'status': 'error',
                'reason': str(e)[:200]
            }
        
        finally:
            # Release GPU semaphore
            if gpu_acquired and _GPU_SEMAPHORE is not None:
                _GPU_SEMAPHORE.release()
    
    # ============================================
    # AI PORTFOLIO MODEL TRAINING
    # ============================================
    elif task_type == 'ai_portfolio':
        X = task.get('X')
        y = task.get('y')
        
        try:
            # Acquire GPU semaphore if XGBoost
            gpu_acquired = False
            if not FORCE_CPU and model_type == 'XGBoost' and XGBOOST_USE_GPU:
                if _GPU_SEMAPHORE is not None:
                    _GPU_SEMAPHORE.acquire()
                    gpu_acquired = True
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Initialize model based on type
            model = None
            if model_type == 'XGBoost':
                import xgboost as xgb
                common_kwargs = {
                    'random_state': SEED,
                    'n_jobs': 1,
                    'eval_metric': 'logloss',
                    'use_label_encoder': False
                }
                if XGBOOST_USE_GPU and CUDA_AVAILABLE and not FORCE_CPU:
                    common_kwargs["device"] = "cuda"
                model = xgb.XGBClassifier(n_estimators=200, max_depth=7, **common_kwargs)
            
            elif model_type == 'LightGBM':
                from lightgbm import LGBMClassifier
                model = LGBMClassifier(
                    n_estimators=200, max_depth=7, random_state=SEED, 
                    verbosity=-1, n_jobs=TRAINING_NUM_PROCESSES
                )
            elif model_type == 'Ridge':
                from sklearn.linear_model import LogisticRegression
                model = LogisticRegression(random_state=SEED, max_iter=1000, n_jobs=TRAINING_NUM_PROCESSES)
            
            if model is None:
                return {
                    'task_type': 'ai_portfolio',
                    'model_type': model_type,
                    'status': 'failed',
                    'reason': 'model_not_available'
                }
            
            # Cross-validation
            cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
            cv_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='accuracy', n_jobs=1)
            
            # Train on full data
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                model.fit(X_scaled, y)
            
            train_score = model.score(X_scaled, y)
            
            # Save model to disk
            models_dir = Path("logs/models")
            _ensure_dir(models_dir)
            
            model_path = models_dir / f"ai_portfolio_{model_type}_temp_model.joblib"
            scaler_path = models_dir / f"ai_portfolio_{model_type}_temp_scaler.joblib"
            
            joblib.dump(model, model_path)
            joblib.dump(scaler, scaler_path)
            
            return {
                'task_type': 'ai_portfolio',
                'model_type': model_type,
                'status': 'success',
                'cv_score': np.mean(cv_scores),
                'cv_std': np.std(cv_scores),
                'train_score': train_score,
                'model_path': str(model_path),
                'scaler_path': str(scaler_path)
            }
        
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {
                'task_type': 'ai_portfolio',
                'model_type': model_type,
                'status': 'error',
                'reason': str(e)[:200]
            }
        
        finally:
            # Release GPU semaphore
            if gpu_acquired and _GPU_SEMAPHORE is not None:
                _GPU_SEMAPHORE.release()
    
    else:
        return {
            'task_type': task_type,
            'status': 'error',
            'reason': f'unknown_task_type: {task_type}'
        }
