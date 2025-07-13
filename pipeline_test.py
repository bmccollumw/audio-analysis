import os
import tempfile
import psycopg2
import librosa
import numpy as np
import yt_dlp

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

def analyze_audio(filepath):
    try:
        y, sr = librosa.load(filepath)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        key_index = np.argmax(chroma_mean)
        key_mapping = ['C', 'C#', 'D', 'D#', 'E', 'F',
                       'F#', 'G', 'G#', 'A', 'A#', 'B']
        key = key_mapping[key_index]
        return float(tempo), key
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        return None, None

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

def run_pipeline():
    songs = get_songs_to_analyze()
    for song_id, name, artist in songs:
        print(f"\n🎧 Processing: {name} — {artist}")
        filepath = download_audio(name, artist)
        if not filepath or not os.path.exists(filepath):
            continue

        bpm, key = analyze_audio(filepath)
        if bpm and key:
            update_song(song_id, bpm, key)
            print(f"✅ Updated DB: {name} → BPM: {bpm:.2f}, Key: {key}")
        else:
            print("⚠️ Could not analyze song.")

        try:
            os.remove(filepath)
        except Exception as e:
            print(f"⚠️ Could not delete temp file: {e}")

if __name__ == "__main__":
    run_pipeline()
