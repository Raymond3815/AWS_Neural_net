import tables
import numpy as np
from pyproj import Proj # added this is essentially https://mygeodata.cloud/cs2cs/
from os.path import isfile
import datetime
import time

def stringToDateTime(string, format = "%Y/%m/%d %H:%M"):
    return datetime.datetime.strptime(string, format)

def localToUtc(t):
    return t - datetime.datetime.now() + t.utcnow()

def generateFilePath(timestamp, basePath):
    mi = str(timestamp.minute)
    if (len(mi) == 1):
        mi = '0' + mi
    h = str(timestamp.hour)
    if (len(h) == 1):
        h = '0' + h
    d = str(timestamp.day)
    if (len(d) == 1):
        d = '0' + d
    m = str(timestamp.month)
    if (len(m) == 1):
        m = '0' + m
    y = str(timestamp.year)

    return basePath + '\\' + y + '\\' + m + '\\' + "RAD_NL25_RAC_MFBS_EM_5min_" + y + m + d + h + mi + "_NL.h5"


def loadImage(timestamp, basePath = "Z:\\RAD_NL25_RAC_MFBS_EM_5min"):
    fp = generateFilePath(timestamp, basePath)
    if (isfile(fp)):
        with tables.open_file(fp) as h5:
            data = np.array(h5.get_node('/image1', 'image_data')[452, 326]) * 0.01
    return data

def loadImageSpecific(timestamp, indicesArr, basePath = "Z:\\RAD_NL25_RAC_MFBS_EM_5min"):
    fp = generateFilePath(timestamp, basePath)

    if (isfile(fp)):
        data = []
        with tables.open_file(fp) as h5:
            node = h5.get_node('/image1', 'image_data')
            for i in indicesArr:
                data.append(node[i[0], i[1]])
            data = np.array(data, dtype=np.float32)
            data[data == 65535] = np.nan

        return data * 0.01
    return np.ones([len(indicesArr)]) * np.nan

radar_corners = np.array([
    [49.362, 0],
    [55.974, 0],
    [55.389, 10.856],
    [48.895, 9.009]
])

getProj = Proj('+proj=stere +lat_0=90 +lon_0=0.0 +lat_ts=60.0 +a=6378.137 +b=6356.752 +x_0=0 +y_0=0') # funct(long,lat)

# convert corners to WGS84->projection
radar_corners = [getProj(c[1], c[0]) for c in radar_corners]

def latLongRelProj(latlong, noRound = False):
    proj = getProj(latlong[1], latlong[0])
    if (noRound == False):
        return [int(radar_corners[2][1] - proj[1] + 0.5), int(proj[0] - radar_corners[0][0] + 0.5)]
    return [radar_corners[2][1] - proj[1], proj[0] - radar_corners[0][0]]

#def loadImages(year, month=-1, day=-1, hour=-1, minute=-1, basePath="Z:\\RAD_NL25_RAC_MFBS_EM_5min"):




if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import pandas as pd
    from time import time

    c = latLongRelProj([51.92530, 4.547670])
    print(c)

    t0 = stringToDateTime("2017/05/12 00:00")

    data = {'t': [], 'r': []}
    start = time()
    for i in range(1*12*24):

        d = t0 + datetime.timedelta(minutes=i*5)

        val = loadImageSpecific(d, [c], "C:\\Users\\raymo\\Downloads\\RAD_NL25_RAC_MFBS_EM_5min")[0]
        data['t'].append(d)
        data['r'].append(val)

        if i % 288 == 0: # every day
            print(d, val)
        #print(d[c[0]][c[1]], 'mm')

        #max_real_value = np.max(d[d != 655.35])
        #print(max_real_value, 'mm')
        #plt.subplot(3, 3, i+1)
        #plt.imshow(d[c[0]-50:c[0]+50, c[1]-50:c[1]+50], clim=(0, max_real_value));
        #plt.colorbar()
        #plt.title(t0 + datetime.timedelta(minutes=5*i))
    print(time() - start)
    df = pd.DataFrame(data, index = data['t']) # .resample('H', kind='period').mean()*12
    df['r'].plot()
    plt.show()



