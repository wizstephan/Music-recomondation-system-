from config import get_spotify_client
from recommender import recommend_songs
import pandas as pd
import os


def main():
    # 1️⃣ Load your dataset
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(project_root, "data", "spotify_tracks.csv")

    # adjust path if needed
    dataset = pd.read_csv(dataset_path)

    # 2️⃣ Create Spotify client (for API fallback)
    sp = get_spotify_client()

    # 3️⃣ Define seed songs for testing
    seed_songs = [
        {"name": "Shape of You", "artist": "Ed Sheeran"},
        {"name": "Blinding Lights", "artist": "The Weeknd"}
    ]

    # 4️⃣ Get recommendations
    recommendations = recommend_songs(dataset, seed_songs, n=10)

    # 5️⃣ Print results
    if recommendations.empty:
        print("No recommendations found. Check your dataset and seed songs.")
    else:
        print("\nTop recommendations based on seed songs:")
        print(recommendations.to_string(index=False))

if __name__ == "__main__":
    main()
