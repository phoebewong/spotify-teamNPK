def nholdout(playlist_id, df):
    '''Pass in a playlist id to get number of songs held out in val/test set'''

    return len(df[df.Playlistid == playlist_id].Track_uri)

def cPredict(dfCluster, playlist_id, clusterRank, df):

    k = nholdout(playlist_id, df)*15 #number of predictions
    pred = list() #cache list for prediction

    playlist = dfCluster[dfCluster['Playlistid'] == playlist_id] #get playlist from cluster data frame
    tracks = playlist.Track_uri.values #list of existing tracks

    #refined model - prioritize artist
    artistCNT = playlist.groupby('Artist_Name').agg({
        'Track_uri': len
    }).reset_index().sort_values(by = 'Track_uri', ascending = False) #get count of artist by name

    try:
        pred_artist = artistCNT[artistCNT['Track_uri'] >=3].Artist_Name #get artist names if they appear at least three times in playlist
        for artist in pred_artist:
            fit = dfCluster[dfCluster['Artist_Name'] == artist] #subset artist from prediction set
            pred_uri_by_artist = fit.Track_uri.values
            for uri in pred_uri_by_artist:
                if uri not in tracks and uri not in pred:
                    pred.append(uri)
        
        if len(pred) > k:
            pred = pred[0:k]
            return pred
        elif len(pred) == k:
            return pred
        else:
            #get cluster label count from playlist
            clusterCNT = playlist.groupby('cluster_label').agg({
            'Playlistid': len
            }).reset_index().sort_values(by = 'Playlistid', ascending = False)

            #cluster labels order by occurance in descending order
            labels = clusterCNT.cluster_label.values.tolist()

            #populate cluster labels based on computed euclidean distances
            for label in labels:
                add = clusterRank[str(label)]
                for c in add:
                    if c not in labels:
                        labels.append(c)

            #predict based on cluster popularity
            for label in labels:
                fit = dfCluster[dfCluster['cluster_label'] == label] #subset tracks with the same label
                rankTrack = fit.sort_values(by = ['mode_artist','mode_track'], ascending = [False,False]) #rank tracks in fit by artist and track
                pred_uri = rankTrack.Track_uri.values
                for uri in pred_uri:
                    if uri not in tracks and uri not in pred:
                        pred.append(uri)
                        if len(pred) == k:
                            break
                    if len(pred) == k:
                        break
                if len(pred) == k:
                    break

            return pred

    except: #in case there is the playlist has a very diverse artist pool, just predict based on cluster

        #get cluster label count from playlist
        clusterCNT = playlist.groupby('cluster_label').agg({
        'Playlistid': len
        }).reset_index().sort_values(by = 'Playlistid', ascending = False)

        #cluster labels order by occurance in descending order
        labels = clusterCNT.cluster_label.values.tolist()

        #populate cluster labels based on computed euclidean distances
        for label in labels:
            add = clusterRank[str(label)]
            for c in add:
                if c not in labels:
                    labels.append(c)

        #predict based on cluster popularity
        for label in labels:
            fit = dfCluster[dfCluster['cluster_label'] == label] #subset tracks with the same label
            rankTrack = fit.sort_values(by = ['mode_artist','mode_track'], ascending = [False,False]) #rank tracks in fit by artist and track
            pred_uri = rankTrack.Track_uri.values
            for uri in pred_uri:
                if uri not in tracks and uri not in pred:
                    pred.append(uri)
                    if len(pred) == k:
                        break
                if len(pred) == k:
                    break
            if len(pred) == k:
                break

        return pred
