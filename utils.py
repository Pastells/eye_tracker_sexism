import pandas as pd
import numpy as np

def z_score(df, col, participant_col):
    """Compute z-score of col for each participant"""
    stats = df[[participant_col, col]].groupby(participant_col)[col].agg(['mean', 'std'])
    df = df.merge(stats, on=participant_col, suffixes=('', '_stats'))
    df[col + ' z-score'] = (
        (df[col] - df['mean']) / df['std']
    )
    df = df.drop(columns=['mean', 'std'])
    return df

def get_hits(df, texts, aoi_hit, part_col, normalize = True) -> dict:
    text_hits = {}
    for text in texts:
        aoi_hit_text = [col for col in aoi_hit if f"[{text} - " in col]
        text_hits[text] = get_hits_text(df, text, aoi_hit_text, part_col, normalize = normalize)
    return text_hits

def get_hits_text(df, text, aoi_hit_text, part_col, normalize = True) -> dict:
    res = {}
    participants = df[part_col].unique().tolist()    
    res["tokens"] = [t.split("- ")[1].split("]")[0] for t in aoi_hit_text]
    hits = df[[part_col] + aoi_hit_text].groupby(part_col).sum()[aoi_hit_text]
    hits = hits.to_numpy()
    if normalize:
        hits / hits.sum(axis=1, keepdims=True)
    res["hits"] = hits
    
    z_pupil = []
    for token in aoi_hit_text:
        hit_1 = df[df[token] == 1]['Pupil diameter filtered z-score']
        z = 0 if hit_1.empty else hit_1.mean()
        z_pupil.append(z)
    res["z_pupil"] = np.array(z_pupil)

    # Create a MultiIndex for all combinations of participants and tokens
    idx = pd.MultiIndex.from_product([participants, aoi_hit_text], names=[part_col, 'token'])

    # Calculate z_pupil for each participant and token
    hit_mask = df[aoi_hit_text] == 1
    filtered_df = df.loc[hit_mask.any(axis=1), [part_col, 'Pupil diameter filtered z-score'] + aoi_hit_text]
    z_pupil_df = (
        filtered_df.melt(id_vars=[part_col, 'Pupil diameter filtered z-score'],
                value_vars=aoi_hit_text,
                var_name='token')
        .query('value == 1')
        .groupby([part_col, 'token'])['Pupil diameter filtered z-score']
        .mean()
    )
    z_pupil_df = z_pupil_df.reindex(idx, fill_value=0)
    z_pupil = z_pupil_df.unstack().to_numpy()
    res["z_pupil"] = z_pupil
    return res