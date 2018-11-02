import os
import json
import numpy as np
import pandas as pd
from collections import defaultdict



extract = defaultdict(list)

def processMPD():
    
    directory = os.getcwd()
    
    for jsonfile in os.listdir(directory):
    
        if '.json' in jsonfile:

            data = json.load(open(jsonfile, "rb" ))
            
            for info in data['playlists']:
                
                pName = info['name']
                pAlbum = info['num_albums']
                pTrack = info['num_tracks']
                pArtist = info['num_artists']
                pEdit = info['num_edits']
                pFollower = info['num_followers']
                pCollab = info['collaborative']
                extract['Playlist'].append(pName)
                extract['Album'].append(pAlbum)
                extract['Track'].append(pTrack)
                extract['Artist'].append(pArtist)
                extract['Edit'].append(pEdit)
                extract['Follower'].append(pFollower)
                extract['Collaborative'].append(pCollab)
                extract['MPDid'].append(info['pid'])
                
    return extract

processMPD()

dfEDA = pd.DataFrame.from_dict(extract)
dfEDA.to_csv('dfEDA.csv')


    
