import requests
from pandas import read_csv
from os.path import isfile

## ['DateTime', 'Tair', 'RH', 'vapor_pressure_{Avg}', 'WindSpd_{Avg}', 'WindDir_{Avg}', 'Rain_{Tot}']
##  datetime    C       %       kPa                     m/s                degrees          mm


def downloadStationCsv(stationName):
    print("Downloading data of station", stationName)
    req = requests.get("http://weather.tudelft.nl/csv/" + stationName + '.csv')

    # remove rows not containting 33 columns (assuming once found 33 the rest is correct)
    # reason is for reading data using c function, require constant number of columns
    print("Filtering data of station", stationName)
    req.encoding = 'UTF-8'
    rows = req.text.replace('NAN', 'NaN').split('\r\n')

    correctLine=0
    for r in rows:
        if (len(r.split(',')) == 33):
            break
        correctLine += 1
    del rows[0:correctLine]
    print(correctLine, 'rows not containing 33 columns removed')

    print("Storing filtered data of station", stationName)
    with open("Station_Data\\"+ stationName + '.csv', 'w') as f:
        f.write('\n'.join(rows))


def loadStationToDict(specificStations = [], forceDownload=False):
    object = {}

    station_info = read_csv("Station_Data\\stations_Rotterdam.txt", index_col=0)

    avail_stations = station_info.index.values
    selected_stations = []
    if (len(specificStations)):
        for s in specificStations:
            if (s in avail_stations):
                selected_stations.append(s)
    else:
        selected_stations = avail_stations

    for station in selected_stations:

        # define object
        object[station] = {
            'latitude': station_info['latitude'][station],
            'longitude': station_info['longitude'][station],
            'comment': station_info['comment'][station]
        }

        # aquire data
        fn = "Station_Data\\" + station + '.csv'
        if ((forceDownload) | (isfile(fn) == 0)):
            downloadStationCsv(station)
        # read data
        print("Loading in",fn)
        object[station]['data'] = read_csv(fn,
            names = ["DateTime", "Name", "Battery_{Min}", "LithiumBattery_{Min}", "PanelTemp_{Max}", "CS215Error_{Tot}", "Tair", "RH", "vapor_pressure_{Avg}", "vapor_pressure_{s, Avg}", "WindSonicError_{Tot}", "WindSpd_{Max}", "WindSpd_{Std}", "WindSpd_{Avg}", "WindDir_{Avg}", "WindDir_{Std}", "WindDirError_{Tot}", "Rain_{Tot}", "SR01Up_{Avg}", "SR01Dn_{Avg}", "IR01Up_{Avg}", "IR01Dn_{Avg}", "Tir01_{Avg}", "Tglobe_{Avg}", "Mir01", "T_{sky}", "T_{surface}", "NetRs", "NetRl", "NetR", "TotUp", "TotDn", "Albedo"],
            usecols = ['DateTime', 'Tair', 'RH', 'vapor_pressure_{Avg}', 'WindSpd_{Avg}', 'WindDir_{Avg}', 'Rain_{Tot}'],
            parse_dates=['DateTime'], index_col= 0
        )
        # remove duplicates
        object[station]['data'] = object[station]['data'][~object[station]['data'].index.duplicated()]

    return object

def detectDateErrorCsv(array):
    from dateutil.parser import parse
    for i in range(len(array)):
        if (i%1000 == 0):
            print("At", i)
        try:
            parse(array[i]) # will fail once not able to parse
        except ValueError:
            print(i, array[i], 'went wrong')

def detectNumericalErrorCsv(array):
    # require numpy to convert str to float (applied on array, python float is slow)
    import numpy as np
    checkSize = len(array) // 2
    errorIndex = 0
    errorPart = -1
    while (checkSize > 1):
        try:
            np.array(array[0:checkSize]).astype((np.float))
        except ValueError:
            errorPart = 0
        try:
            np.array(array[checkSize:]).astype((np.float))
        except ValueError:
            errorPart = 1

        if (errorPart == -1):
            print("No errors")
            return -1

        if(errorPart == 0):
            array = array[0:checkSize]
        else:
            array = array[checkSize:]
            errorIndex += checkSize

        checkSize = len(array)//2

    print(errorIndex, array[errorPart])
    return errorIndex

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    ## NOTE Oommoord has a error in CSV at line 453856 (after processing, look for 2018-02-12 05:40)
    ## data gap of 30 minutes
    sDict = loadStationToDict(['Oost', 'Capelle', 'Delfshaven', 'Lansingerland', 'Ommoord', 'Rijnhaven', 'SpaansePolder'], forceDownload=False);







    #exit() ## remove this to make plots

    ## Generate count plot
    period = 'W'
    for s in sDict.keys():
        print("Analysing", s)
        cols = sDict[s]['data'].keys()
        hourCountSum = sDict[s]['data'][cols[0]].resample(period, kind = 'period').count()
        for i in range(1, len(cols)):
            hourCountSum += sDict[s]['data'][cols[i]].resample(period, kind = 'period').count()
        hourCountSum.plot(label=s)

    plt.title("Measurement count (sum of col count) period set to " + period)
    plt.legend(loc='best')
    #plt.show()

    period = 'M'
    theshold = 8 # mm/h (fairly good pick)
    for s in sDict.keys():
        if (s == 'Ridderkerk'):
            continue
        print("Analysing count high intensity", s)
        cols = sDict[s]['data'].keys()
        subcel = sDict[s]['data']["Rain_{Tot}"][((sDict[s]['data']["Rain_{Tot}"] > theshold/12))]
        hourCountSum = subcel.resample(period, kind='period').count()
        hourCountSum.plot(label=s)

    plt.title("Measurement count (sum of high intensity count) period set to " + period)
    plt.legend(loc='best')
    #plt.show()

    # Generate mean plot to detect outliers
    plotCols = ['Tair', 'RH', 'vapor_pressure_{Avg}',  'WindSpd_{Avg}', 'WindDir_{Avg}', 'Rain_{Tot}']
    for c in plotCols:

        all_rain = []
        i = 0
        for s in sDict.keys():
            print("Analysing", c, 'of', s)


            all_rain = np.concatenate((all_rain, sDict[s]['data'][c].values))

            i+=1
            #plt.subplot(5,2,i)
            #subcel.box()
            #plt.legend([s], loc='best')
            #plt.title(c)

        #plt.show()

        std, mu = 0, 0

        all_rain = all_rain[~np.isnan(all_rain)]

        print(all_rain, np.isnan(all_rain).any())

        if (c == 'Rain_{Tot}'):
            std = np.std(all_rain[all_rain != 0]) # note condition made only for rain
            mu = np.mean(all_rain[all_rain != 0]) #
        else:
            std = np.std(all_rain)  # note condition made only for rain
            mu = np.mean(all_rain)  #

        min, max = np.min(all_rain), np.max(all_rain)


        print(c, 'mean',mu,'std', std, 'min', min, 'max', max)
        print('low', mu - std, 'up', mu + std)
        nBins = int(5*(max-min) / std)
        print('nbin', nBins)
        #n,_,_ = plt.hist(all_rain, bins=nBins, normed=1)
        #print('bins', n[0:int(5 * (mu + std*3) / std)])
        #plt.axvline(mu - std)
        #plt.axvline(mu)
        #plt.axvline(mu + std)
        #plt.xlim(mu - 3*std, mu + 3*std)
        #plt.title(c)
        #plt.show()
