from recpack.datasets import (
    CiteULike, CosmeticsShop, DummyDataset, Globo,
    MillionSongDataset, MovieLens100K, MovieLens10M, MovieLens1M,
    MovieLens25M, Netflix, RecsysChallenge2015, RetailRocket,
    TasteProfile, ThirtyMusicSessions
)
from recpack.preprocessing.filters import (
    Deduplicate, MaxItemsPerUser, MinItemsPerUser, MinRating,
    MinUsersPerItem, NMostPopular, NMostRecent
)
from .recpack_extensions import AdressaOneWeekCustom, MillionSongDatasetCustom, TasteProfileCustom

DATASETS = {
    "MovieLens100K": MovieLens100K,
    "MovieLens1M": MovieLens1M,
    "MovieLens10M": MovieLens10M,
    "MovieLens25M": MovieLens25M,
    "Netflix": Netflix,
    "CiteULike": CiteULike,
    "AdressaOneWeek": AdressaOneWeekCustom,
    "MillionSongDataset": MillionSongDatasetCustom,
    "TasteProfile": TasteProfileCustom,
    "RecsysChallenge2015": RecsysChallenge2015,
    "RetailRocket": RetailRocket,
    "ThirtyMusicSessions": ThirtyMusicSessions
}

FILTERS = {
    "MinItemsPerUser": MinItemsPerUser,
    "MinUsersPerItem": MinUsersPerItem,
    "MinRating": MinRating,
    "MaxItemsPerUser": MaxItemsPerUser,
    "NMostPopular": NMostPopular,
    "NMostRecent": NMostRecent,
    "Deduplicate": Deduplicate
}

from recpack.algorithms import Popularity, BPRMF, SVD, Random, EASE
from .recpack_extensions import ItemKNNExtended

ALGORITHMS = {
    "ItemKNN": ItemKNNExtended,
    "Popularity": Popularity,
    "BPRMF": BPRMF,
    "SVD": SVD,
    "Random": Random,
    "EASE": EASE
}
