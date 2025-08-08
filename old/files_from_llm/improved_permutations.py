"""
Improved permutation module with better performance and cleaner interface.
"""
import numpy as np
import pandas as pd
from typing import Union, List, Optional
from dataclasses import dataclass


@dataclass
class PermutationConfig:
    """Configuration for permutation generation."""
    start_index: int = 0
    seed: Optional[int] = None
    validate_data: bool = True


class PermutationGenerator:
    """High-performance permutation generator for OHLC data."""
    
    def __init__(self, config: PermutationConfig = None):
        self.config = config or PermutationConfig()
    
    def generate(self, ohlc: Union[pd.DataFrame, List[pd.DataFrame]]) -> Union[pd.DataFrame, List[pd.DataFrame]]:
        """
        Generate permuted OHLC data with improved performance.
        
        Args:
            ohlc: Single DataFrame or list of DataFrames with OHLC data
            
        Returns:
            Permuted OHLC data in same format as input
        """
        if self.config.seed is not None:
            np.random.seed(self.config.seed)
        
        # Normalize input to list format
        is_single_df = not isinstance(ohlc, list)
        ohlc_list = [ohlc] if is_single_df else ohlc
        
        if self.config.validate_data:
            self._validate_data(ohlc_list)
        
        # Get common properties
        n_markets = len(ohlc_list)
        time_index = ohlc_list[0].index
        n_bars = len(ohlc_list[0])
        
        # Calculate permutation parameters
        perm_start = self.config.start_index + 1
        perm_length = n_bars - perm_start
        
        if perm_length <= 0:
            raise ValueError(f"Start index {self.config.start_index} too large for data length {n_bars}")
        
        # Pre-allocate arrays for better performance
        start_bars = np.empty((n_markets, 4))
        relative_data = {
            'open': np.empty((n_markets, perm_length)),
            'high': np.empty((n_markets, perm_length)),
            'low': np.empty((n_markets, perm_length)),
            'close': np.empty((n_markets, perm_length))
        }
        
        # Vectorized calculation of relative values
        for mkt_idx, df in enumerate(ohlc_list):
            log_ohlc = np.log(df[['open', 'high', 'low', 'close']].values)
            
            # Store start bar
            start_bars[mkt_idx] = log_ohlc[self.config.start_index]
            
            # Calculate relative values vectorized
            relative_data['open'][mkt_idx] = (log_ohlc[perm_start:, 0] - 
                                            log_ohlc[perm_start-1:-1, 3])
            relative_data['high'][mkt_idx] = (log_ohlc[perm_start:, 1] - 
                                            log_ohlc[perm_start:, 0])
            relative_data['low'][mkt_idx] = (log_ohlc[perm_start:, 2] - 
                                           log_ohlc[perm_start:, 0])
            relative_data['close'][mkt_idx] = (log_ohlc[perm_start:, 3] - 
                                             log_ohlc[perm_start:, 0])
        
        # Generate permutations
        perm_indices = np.arange(perm_length)
        
        # Separate permutations for intrabar and gap data
        intrabar_perm = np.random.permutation(perm_indices)
        gap_perm = np.random.permutation(perm_indices)
        
        # Apply permutations
        for key in ['high', 'low', 'close']:
            relative_data[key] = relative_data[key][:, intrabar_perm]
        relative_data['open'] = relative_data['open'][:, gap_perm]
        
        # Reconstruct permuted data
        permuted_ohlc = []
        for mkt_idx, original_df in enumerate(ohlc_list):
            perm_data = self._reconstruct_ohlc(
                original_df, start_bars[mkt_idx], relative_data, 
                mkt_idx, time_index, n_bars, perm_start
            )
            permuted_ohlc.append(perm_data)
        
        return permuted_ohlc[0] if is_single_df else permuted_ohlc
    
    def _validate_data(self, ohlc_list: List[pd.DataFrame]):
        """Validate input data consistency."""
        if not ohlc_list:
            raise ValueError("Empty OHLC data provided")
        
        reference_index = ohlc_list[0].index
        required_columns = ['open', 'high', 'low', 'close']
        
        for i, df in enumerate(ohlc_list):
            # Check required columns
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                raise ValueError(f"DataFrame {i} missing columns: {missing_cols}")
            
            # Check index consistency
            if not reference_index.equals(df.index):
                raise ValueError(f"DataFrame {i} has inconsistent index")
    
    def _reconstruct_ohlc(self, original_df: pd.DataFrame, start_bar: np.ndarray,
                         relative_data: dict, mkt_idx: int, time_index: pd.Index,
                         n_bars: int, perm_start: int) -> pd.DataFrame:
        """Reconstruct OHLC data from permuted relative values."""
        # Initialize with original data before start index
        log_original = np.log(original_df[['open', 'high', 'low', 'close']].values)
        perm_log = log_original.copy()
        
        # Set start bar
        perm_log[self.config.start_index] = start_bar
        
        # Vectorized reconstruction of permuted section
        for i in range(perm_start, n_bars):
            k = i - perm_start
            prev_close = perm_log[i-1, 3]
            
            perm_log[i, 0] = prev_close + relative_data['open'][mkt_idx, k]  # open
            perm_log[i, 1] = perm_log[i, 0] + relative_data['high'][mkt_idx, k]  # high
            perm_log[i, 2] = perm_log[i, 0] + relative_data['low'][mkt_idx, k]   # low
            perm_log[i, 3] = perm_log[i, 0] + relative_data['close'][mkt_idx, k] # close
        
        # Convert back to price space
        perm_prices = np.exp(perm_log)
        
        return pd.DataFrame(
            perm_prices, 
            index=time_index, 
            columns=['open', 'high', 'low', 'close']
        )
    
    def generate_multiple(self, ohlc: Union[pd.DataFrame, List[pd.DataFrame]], 
                         n_permutations: int) -> List[Union[pd.DataFrame, List[pd.DataFrame]]]:
        """Generate multiple permutations efficiently."""
        return [self.generate(ohlc) for _ in range(n_permutations)]


def create_permutation(ohlc: Union[pd.DataFrame, List[pd.DataFrame]], 
                      start_index: int = 0, seed: Optional[int] = None) -> Union[pd.DataFrame, List[pd.DataFrame]]:
    """
    Convenience function for single permutation generation.
    Maintains compatibility with original interface.
    """
    config = PermutationConfig(start_index=start_index, seed=seed)
    generator = PermutationGenerator(config)
    return generator.generate(ohlc)