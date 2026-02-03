from recpack.datasets import AdressaOneWeek, MillionSongDataset
import pandas as pd
import numpy as np

class IDMappingDatasetMixin:
    """Mixin to add ID mapping functionality to RecPack datasets."""
    def _map_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        if df[self.USER_IX].dtype == object or df[self.USER_IX].dtype == str:
            user_mapping = {val: i for i, val in enumerate(df[self.USER_IX].unique())}
            df[self.USER_IX] = df[self.USER_IX].map(user_mapping)
            self.user_mapping_ = user_mapping
            
        if df[self.ITEM_IX].dtype == object or df[self.ITEM_IX].dtype == str:
            item_mapping = {val: i for i, val in enumerate(df[self.ITEM_IX].unique())}
            df[self.ITEM_IX] = df[self.ITEM_IX].map(item_mapping)
            self.item_mapping_ = item_mapping
            
        return df

class AdressaOneWeekCustom(AdressaOneWeek, IDMappingDatasetMixin):
    """
    Custom AdressaOneWeek dataset that maps string user and item IDs to integers.
    """
    def _load_dataframe(self) -> pd.DataFrame:
        df = super()._load_dataframe()
        return self._map_ids(df)

class MillionSongDatasetCustom(MillionSongDataset, IDMappingDatasetMixin):
    """
    Custom MillionSongDataset that maps string user and item IDs to integers.
    """
    def _load_dataframe(self) -> pd.DataFrame:
        df = super()._load_dataframe()
        return self._map_ids(df)

class TasteProfileCustom(MillionSongDatasetCustom):
    """
    Custom TasteProfile dataset (alias for MillionSongDataset).
    """
    pass
