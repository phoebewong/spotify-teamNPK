import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

client_credentials_manager = SpotifyClientCredentials(client_id='fc03c036ef2e4552810bcfa717b8c9d6',
                                                      client_secret='3f2bc1d385394fdebbfbca47a9b07268')
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

#spotify playlist info demo 
playlists = sp.user_playlists('spotify')
while playlists:
    for i, playlist in enumerate(playlists['items']):
        print("%4d %s %s" % (i + 1 + playlists['offset'], playlist['uri'],  playlist['name']))
    if playlists['next']:
        playlists = sp.next(playlists)
    else:
        playlists = None

#spotify audio features demo

list_uri = #list of track uri

playlists = sp.audio_features(list_uri)
