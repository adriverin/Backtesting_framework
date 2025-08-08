# A Practical Guide to `MLConfig` Hyperparameters (Gemini 2.5 Pro)

The `MLConfig` dataclass controls every aspect of the machine learning strategy, from how it interprets the data to how the model is built and trained.

### Core & Data Interpretation

These parameters define the basic data context.

*   **`price_column: str = "close"`**
    *   **What it is:** The column from your OHLCV data to use as the primary price for all calculations (returns, features, etc.).
    *   **How to use it:** Can be set to `"open"`, `"high"`, `"low"`, or `"close"`. The default is `"close"`.

*   **`interval: str = "1h"`**
    *   **What it is:** The timeframe of your data bars (e.g., '1h', '4h', '1d').
    *   **How to use it:** This is crucial for time-based calculations and should match the `--tf` argument provided to your backtesting scripts. It is set automatically by the `MLStrategy`'s `__init__` method.

### Forecasting & Feature Engineering

These parameters define *what* the model learns from.

*   **`forecast_horizon_hours: int = 6`**
    *   **What it is:** This is the **most important parameter**. It defines the "question" you are asking the model: **"Based on the data *now*, what will the return be in `X` hours?"** The model's goal is to predict the quantile of the price return `forecast_horizon_hours` into the future.
    *   **How to use it:**
        *   A **short horizon** (e.g., 1-4 hours) trains a short-term, high-frequency predictor which may be sensitive to noise.
        *   A **long horizon** (e.g., 24-72 hours) trains the model to identify longer-term trends, making it less sensitive to short-term volatility.
        *   This should be chosen based on your desired holding period.

*   **`vol_window_hours: int = 60`**
    *   **What it is:** The lookback period (in hours) used to calculate rolling volatility. This volatility is used to create a `norm_return` (normalized return), which helps the model perform consistently across different market conditions.
    *   **How to use it:** A longer window gives a smoother, more stable volatility measure.

*   **`enable_regime_features: bool = True`**
    *   **What it is:** A switch to turn on/off the calculation of "market regime" features (e.g., a high/low volatility indicator).
    *   **How to use it:** `True` gives the model more context about the market's state. `False` simplifies the model by removing these features.

*   **`sma_windows`, `momentum_windows`, `rsi_windows`: Tuple[int, ...]`**
    *   **What they are:** These tuples define the lookback periods (in number of bars, not hours) for calculating technical indicators that serve as features for the model.
    *   **How to use them:** Providing a variety of windows (e.g., short, medium, long) gives the model a multi-dimensional view of the market at different scales.

### Model Architecture

These parameters define the structure of the neural network.

*   **`n_quantiles: int = 5`**
    *   **What it is:** The number of "buckets" or categories the model will predict the future return into. `5` means it predicts one of: "very negative," "negative," "neutral," "positive," or "very positive."
    *   **How to use it:** `3` is simple ("down, neutral, up"), while `5` or `7` provide more granularity. More than 10 can make the problem too difficult. `5` is a strong default.

*   **`hidden_sizes: Tuple[int, ...] = (64, 32, 16)`**
    *   **What it is:** Defines the architecture of the neural network's hidden layers. `(64, 32, 16)` means a network with three hidden layers of 64, 32, and 16 neurons, respectively.
    *   **How to use it:** Controls the model's complexity. More layers/neurons can learn more complex patterns but risk overfitting. Fewer layers/neurons create a simpler model that generalizes better but might miss subtle patterns.

*   **`dropout_rate: float = 0.5`**
    *   **What it is:** A regularization technique to prevent overfitting. It randomly "drops" a fraction of neurons during training, forcing the network to learn more robustly.
    *   **How to use it:** A value between `0.2` and `0.5` is typical. Higher values provide stronger regularization.

### Training Parameters

These parameters control *how* the model learns.

*   **`n_epochs: int = 30`**
    *   **What it is:** The maximum number of times the training algorithm will iterate over the entire training dataset.
    *   **How to use it:** Training will stop before this number if `early_stopping_patience` is met.

*   **`lr: float = 1e-4`**
    *   **What it is:** The **Learning Rate**. This critical parameter controls how much the model's weights are adjusted during each training step.
    *   **How to use it:** Too high (e.g., `1e-2`) can cause instability; too low (e.g., `1e-6`) can make training too slow. `1e-3` to `1e-5` is a common and effective range.

*   **`weight_decay: float = 0.01`**
    *   **What it is:** Another regularization technique (L2 regularization) that penalizes large weights, encouraging the model to find simpler solutions and reducing overfitting.
    *   **How to use it:** A small value like `0.01` or `1e-3` is common. `0` disables it.

*   **`batch_size: int = 256`**
    *   **What it is:** The number of training examples used in one iteration before updating the model's weights.
    *   **How to use it:** Larger batch sizes can speed up training but require more memory. `128`, `256`, or `512` are common values.

*   **`train_ratio: float = 0.8` & `val_ratio: float = 0.2`**
    *   **What they are:** Defines how the data inside the `optimize` method is split. `0.8` means 80% is used for active training and `0.2` is used as a validation set to monitor for overfitting and trigger early stopping.
    *   **How to use them:** `0.8 / 0.2` is a standard split.

*   **`early_stopping_patience: int = 5`**
    *   **What it is:** The number of epochs to wait for improvement on the validation set before stopping training. If validation loss doesn't improve for `5` epochs, training halts.
    *   **How to use it:** A crucial defense against overfitting. A value between 3 and 10 is typical.

### Signal Generation

This parameter controls how final model predictions are converted into trade signals.

*   **`signal_percentiles: Tuple[int, int] = (10, 90)`**
    *   **What it is:** After the model predicts probabilities, a composite "extreme preference score" is calculated. This parameter defines the percentile thresholds for this score to generate a signal.
        *   A `long` signal (`+1`) is generated if the score is above the **90th percentile**.
        *   A `short` signal (`-1`) is generated if the score is below the **10th percentile**.
        *   Otherwise, the signal is `neutral` (`0`).
    *   **How to use it:** This directly controls trading frequency.
        *   **Narrower percentiles** (e.g., `(25, 75)`) lead to more frequent trading on less confident predictions.
        *   **Wider percentiles** (e.g., `(5, 95)`) lead to less frequent, more selective trading on highly confident predictions. This is often preferred to reduce noise.