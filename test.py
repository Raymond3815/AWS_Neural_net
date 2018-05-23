import station_data as sd
from scipy.stats import pearsonr

data = sd.loadStationToDict(['Oost'], False)['Oost']['data'].dropna()

for c in data.keys():
    r, p = pearsonr(data[c], data['Rain_{Tot}'])
    print(c, 'to rain intensity R =', r, ', R*R =', r**2, '(p =', p, ')')