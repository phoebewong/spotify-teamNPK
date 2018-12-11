Database:
1. features_partial: raw GitHub audio features data
2. track: all tracks (from MPD playlist), non-unique


* 66346428 rows


3. trackPlus: (complete merge from features_partial to MPD data (track))


* 66346428 rows


<!-- 4. track_meta: merged audio features (data from GitHub) with tracks from MPD data, dropped all playlist with non-complete audio features


* 65936696 rows -->


5. track_meta_100subset: a random subset of 100 from track_meta (for milestone 3)

6. track_api_query


- a list of track_uri that needs to be queried from SpotifyAPI (because NOT in GitHub data)


- 17810499 row


7. artist_info


- a table of info queried from Spotify API
- 295860 row
- popularity: The popularity of the artist. The value will be between 0 and 100, with 100 being the most popular. The artist’s popularity is calculated from the popularity of all the artist’s tracks.
- genres
- followers


8. track_info

- audio features queried from Spotify API


CSV:

1. track_meta_milestone3.csv

- Should use this to replace track_meta

INSERT INTO track_api_query(Track_uri)
SELECT t.Track_uri FROM track AS t
LEFT JOIN features_partial AS fp ON t.Track_uri = fp.Track_uri
WHERE fp.Track_uri is NULL
