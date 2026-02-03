from recpack.datasets import (
    AdressaOneWeek, CiteULike, CosmeticsShop, DummyDataset, Globo, 
    MillionSongDataset, MovieLens100K, MovieLens10M, MovieLens1M, 
    MovieLens25M, Netflix, RecsysChallenge2015, RetailRocket, 
    TasteProfile, ThirtyMusicSessions
)
from recpack.preprocessing.filters import (
    Deduplicate, MaxItemsPerUser, MinItemsPerUser, MinRating, 
    MinUsersPerItem, NMostPopular, NMostRecent
)

DATASETS = {
    "MovieLens100K": MovieLens100K,
    "MovieLens1M": MovieLens1M,
    "MovieLens10M": MovieLens10M,
    "MovieLens25M": MovieLens25M,
    "AdressaOneWeek": AdressaOneWeek,
    "CiteULike": CiteULike,
    "CosmeticsShop": CosmeticsShop,
    "DummyDataset": DummyDataset,
    "Globo": Globo,
    "MillionSongDataset": MillionSongDataset,
    "Netflix": Netflix,
    "RecsysChallenge2015": RecsysChallenge2015,
    "RetailRocket": RetailRocket,
    "TasteProfile": TasteProfile,
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

ALGORITHMS = [
    "ItemKNN (Cosine)",
    "ItemKNN (Jaccard)", 
    "Popularity"
]
