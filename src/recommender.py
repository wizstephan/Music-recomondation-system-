import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Any, Optional

from config import get_spotify_client


# 5. Find a song in the dataset
def find_song(dataset: pd.DataFrame, name: str, artist: str) -> Optional[pd.Series]:
    """Find a song by name and artist in the dataset."""
    song = dataset[
        (dataset["name"].str.lower() == name.lower()) &
        (dataset["artists"].str.lower().str.contains(artist.lower()))
    ]
    if not song.empty:
        return song.iloc[0]  # return first match
    return None


# 6. Fetch song data (from dataset or Spotify API)
def get_song_data(dataset: pd.DataFrame, name: str, artist: str) -> Optional[Dict[str, Any]]:
    """Return song data from dataset if available, otherwise fetch from Spotify API."""
    song = find_song(dataset, name, artist)
    if song is not None:
        return song.to_dict()

    # If not in dataset → query Spotify
    sp = get_spotify_client()
    query = f"track:{name} artist:{artist}"
    results = sp.search(q=query, limit=1, type="track")
    tracks = results.get("tracks", {}).get("items", [])
    if not tracks:
        return None

    track = tracks[0]
    return {
        "name": track["name"],
        "artists": ", ".join([a["name"] for a in track["artists"]]),
        "id": track["id"],
        "popularity": track["popularity"],
        "duration_ms": track["duration_ms"],
    }


# 7. Calculate mean vector of song features
def get_mean_vector(dataset: pd.DataFrame, songs: List[Dict[str, str]]) -> np.ndarray:
    """Compute the mean vector of numerical features for a given list of songs."""
    features = []
    numeric_cols = dataset.select_dtypes(include=[np.number]).columns

    for song in songs:
        song_data = find_song(dataset, song["name"], song["artist"])
        if song_data is not None:
            features.append(song_data[numeric_cols].values)

    if not features:
        raise ValueError("No valid songs found in dataset for computing mean vector.")

    features = np.array(features)
    return np.mean(features, axis=0).reshape(1, -1)


# 8. Flatten a list of dicts
def flatten_dict_list(dict_list: List[Dict]) -> Dict[str, List]:
    """Flatten a list of dictionaries into a dictionary with keys and list of values."""
    flat_dict = {}
    for d in dict_list:
        for key, value in d.items():
            flat_dict.setdefault(key, []).append(value)
    return flat_dict


# 9. Recommend songs
def recommend_songs(dataset: pd.DataFrame, seed_songs: List[Dict[str, str]], n: int = 10) -> pd.DataFrame:
    """Recommend songs similar to the seed songs using cosine similarity."""
    scaler = StandardScaler()
    numeric_cols = dataset.select_dtypes(include=[np.number]).columns

    # Scale dataset features
    scaled_features = scaler.fit_transform(dataset[numeric_cols])

    # Compute mean vector of input songs
    mean_vector = get_mean_vector(dataset, seed_songs)
    scaled_mean = scaler.transform(mean_vector)

    # Compute cosine similarity
    similarities = cosine_similarity(scaled_mean, scaled_features).flatten()

    # Add similarity column
    dataset = dataset.copy()
    dataset["similarity"] = similarities

    # Remove input songs from recommendations
    for song in seed_songs:
        dataset = dataset[~(
            (dataset["name"].str.lower() == song["name"].lower()) &
            (dataset["artists"].str.lower().str.contains(song["artist"].lower()))
        )]

    # Return top N
    return dataset.sort_values("similarity", ascending=False).head(n)[["name", "artists", "similarity"]]
