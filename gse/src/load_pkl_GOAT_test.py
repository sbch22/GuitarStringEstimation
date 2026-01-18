import pickle

# path to your saved track
track_path = "../noteData/GOAT/item_0.pkl"
# load it back
with open(track_path, "rb") as f:
    track = pickle.load(f)

# now you can inspect the contents
print("Track name:", track.name)
print("Metadata:", track.metadata)
print("Number of notes:", len(track.notes))

# check one note
first_note = track.notes[0]

notes = track.notes

# check if audio loaded
if track.audio.mono_mic:
    print("Mono mic signal shape:", track.audio.mono_mic.time.shape)
else:
    print("Mono mic audio not loaded")