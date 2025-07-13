import os
import tempfile
import psycopg2
import librosa
import numpy as np
import yt_dlp
import time

# 🔧 DB Config
DB_CONFIG = {
    "dbname": "spotify_app_db",
    "user": "postgres",
    "password": "pogomuffboo0823",
    "host": "localhost",
    "port": 5432
}

def get_songs_to_analyze(limit=10):
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute("""
        SELECT id, name, artist
        FROM songs
        WHERE bpm IS NULL OR key IS NULL
        LIMIT %s;
    """, (limit,))
    songs = cur.fetchall()
    cur.close()
    conn.close()
    return songs

def download_audio(song_name, artist_name):
    query = f"{song_name} {artist_name} audio"
    ydl_opts = {
        'format': 'bestaudio/best',
        'noplaylist': True,
        'quiet': True,
        'outtmpl': os.path.join(tempfile.gettempdir(), '%(id)s.%(ext)s'),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(f"ytsearch1:{query}", download=True)
            filepath = ydl.prepare_filename(info['entries'][0])
            return filepath.replace('.webm', '.mp3')  # or .m4a, depends
        except Exception as e:
            print(f"❌ Failed to download {song_name} by {artist_name}: {e}")
            return None

def update_song(song_id, bpm, key):
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute("""
        UPDATE songs
        SET bpm = %s, key = %s
        WHERE id = %s;
    """, (bpm, key, song_id))
    conn.commit()
    cur.close()
    conn.close()

def detect_major_minor(y, sr, key_index):
    chroma_cq = librosa.feature.chroma_cqt(y=y, sr=sr)
    major_profile = [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1]
    minor_profile = [1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0]

    major_score = np.dot(major_profile, chroma_cq.mean(axis=1))
    minor_score = np.dot(minor_profile, chroma_cq.mean(axis=1))
    return "major" if major_score > minor_score else "minor"

def run_pipeline():
    songs = get_songs_to_analyze()
    total_start = time.time()

    for song_id, name, artist in songs:
        print(f"\n🎧 Processing: {name} — {artist}")
        song_start = time.time()

        filepath = download_audio(name, artist)
        if not filepath or not os.path.exists(filepath):
            continue

        try:
            y, sr = librosa.load(filepath)
            duration = len(y) / sr
            print(f"⏱️ Duration: {duration:.2f} sec")

            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            tempo = float(tempo.item()) if isinstance(tempo, np.ndarray) else float(tempo)

            chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
            chroma_mean = np.mean(chroma, axis=1)
            key_index = np.argmax(chroma_mean)

            key_mapping = ['C', 'C#', 'D', 'D#', 'E', 'F',
                           'F#', 'G', 'G#', 'A', 'A#', 'B']
            key_note = key_mapping[key_index]
            mode = detect_major_minor(y, sr, key_index)
            full_key = f"{key_note} {mode}"

            update_song(song_id, tempo, full_key)
            print(f"✅ Updated DB: {name} → BPM: {tempo:.2f}, Key: {full_key}")

        except Exception as e:
            print(f"❌ Analysis failed: {e}")

        try:
            os.remove(filepath)
        except Exception as e:
            print(f"⚠️ Could not delete temp file: {e}")

        print(f"🕒 Song processed in {time.time() - song_start:.2f} sec")

    print(f"\n✅ Pipeline completed in {time.time() - total_start:.2f} sec")

if __name__ == "__main__":
    run_pipeline()
