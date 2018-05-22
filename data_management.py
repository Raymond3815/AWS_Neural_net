from os.path import isfile
from  datetime import datetime
import pandas as pd
import numpy as np

import radar_data

def generateCombinedData(toCombine = [], forceRedownload = False):

    import station_data
    data = station_data.loadStationToDict(toCombine, forceRedownload)

    coordinatesOnRadar = []
    stations = list(data)
    base_index = datetime(2200, 1, 1, 0, 0)
    end_index = datetime(1900, 1, 1, 0, 0)
    for s in stations:
        coordinatesOnRadar.append(radar_data.latLongRelProj([data[s]['latitude'], data[s]['longitude']]))
        if (data[s]['data'].index[0] < base_index):
            base_index = data[s]['data'].index[0]
        if (data[s]['data'].index[-1] > end_index):
            end_index = data[s]['data'].index[-1]


    ds = {s:[] for s in stations}
    ds['index']  = pd.date_range( base_index, end_index, freq = '5T')
    counter = 0
    for cDate in ds['index']:
        set = radar_data.loadImageSpecific(cDate, coordinatesOnRadar, "C:\\Users\\raymo\\Downloads\\RAD_NL25_RAC_MFBS_EM_5min")
        for i in range(len(stations)):
            ds[stations[i]].append(set[i])

        if (counter % 8000 == 0):
            print(cDate)
        counter += 1

    dframe  =pd.DataFrame(ds)
    dframe.set_index('index', inplace=True)

    for s in stations:

        data[s]['data'] = pd.merge(
            data[s]['data'].reindex(pd.date_range(data[s]['data'].index[0], data[s]['data'].index[-1], freq='5T'), copy=False), # assure full index
            dframe[[s]], # already contains full index
            left_index=True, right_index=True, copy=False) # both need to exist
        data[s]['data'].rename(columns={s:'Radar_{Tot}'}, inplace=True)

        data[s]['data'].to_csv("combined_data\\" + s + '.csv')

    return data


def loadCombinedData(specificStations = []):
    '''
    Does not generate if not found
    '''
    object = {}

    station_info = pd.read_csv("Station_Data\\stations_Rotterdam.txt", index_col=0)

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
        fn = "combined_data\\" + station + '.csv'

        # read data
        print("Loading in", fn)
        object[station]['data'] = pd.read_csv(fn, index_col=0, parse_dates=[0])
    return object


# consider moving to C++ or finding handy function for
def generateInputOutputSingleStation(combinedData = None, threshold = 8/12, previousTimes = 2, predictAhead=3):
    if (combinedData==None):
        combinedData = loadCombinedData()
    inputArrHeavy,inputArrNonHeavy = [],[]

    for s in combinedData.keys():
        print("Single-Station on", s)
        sLen = len(combinedData[s]['data'])

        offset = 0
        i = 0
        while i < (previousTimes + offset):
            if (combinedData[s]['data'].iloc[i].loc[combinedData[s]['data'].columns != 'Radar_{Tot}'].isnull().any()):
                offset = i
            i += 1


        print("Starting at", i)
        rfr = 0
        while i < (sLen - predictAhead):
            # Invalid radar
            if (combinedData[s]['data'].iloc[i + predictAhead]['Radar_{Tot}'] != combinedData[s]['data'].iloc[i + predictAhead]['Radar_{Tot}']):
                i += 1
                continue

            # Invalid station
            if (combinedData[s]['data'].iloc[i].loc[combinedData[s]['data'].columns != 'Radar_{Tot}'].isnull().any()):
                # look at next steps beyond deciding offset
                print("NaN in input", i)
                j = i + 1
                while j < (i + 1 + previousTimes):
                    if (combinedData[s]['data'].iloc[j].loc[combinedData[s]['data'].columns != 'Radar_{Tot}'].isnull().any()):
                        i = j
                    j += 1
                i += previousTimes + 1
                print("moved to", i, j)
                continue

            if (i % 8064 == 0): # every 4 weeks
                print(combinedData[s]['data'].index[i])

            ## valid set found
            cInput = []
            for j in range(i, i - previousTimes - 1, -1):  # check i as wel as previous ones
                for ciu in combinedData[s]['data'].columns[combinedData[s]['data'].columns != 'Radar_{Tot}']:
                    cInput.append(combinedData[s]['data'].iloc[j][ciu])

            if (combinedData[s]['data'].iloc[i + predictAhead]['Radar_{Tot}'] > threshold):
                inputArrHeavy.append(cInput)
                if combinedData[s]['data'].iloc[i]['Radar_{Tot}'] > threshold:
                    rfr += 1
            else:
                inputArrNonHeavy.append(cInput)
            i+=1
        print(rfr)



    # save
    #np.savetxt('neural_net_data\\raw_single_station_input_below.csv', inputArrNonHeavy)
    #np.savetxt('neural_net_data\\raw_single_station_input_above.csv', inputArrHeavy)

    return inputArrHeavy,inputArrNonHeavy

def generateInputOutputMultipleStations(combinedData = None, otherStations=2, threshold = 8/12, previousTimes = 2, predictAhead=3):
    if (combinedData==None):
        combinedData = loadCombinedData()
    inputArrHeavy,inputArrNonHeavy = [],[]

    if (len(combinedData) < (otherStations + 1)) | (otherStations < 1):
        print("Not enough stations provided")
        return [],[]

    stations = list(combinedData)

    # create list clocest staions for each station
    xy = [] # station,x,y
    for s in stations:
        xy.append(radar_data.latLongRelProj([combinedData[s]['latitude'], combinedData[s]['longitude']], True))
    xy = np.array(xy)

    # merge
    for i in range(len(xy)):
        print("Multi-Station on", stations[i])

        neighboringStations = []
        for j in [k for l in (range(i), range(i+1, len(xy))) for k in l]:
            dv = xy[j] - xy[i]
            dist = np.sqrt(dv.dot(dv))
            dir = np.arccos(dv.dot([-1,0]) / dist) * 180 / np.pi
            if (dv[1] > 0): # angle greater than 180 degrees
                dir = 360 - dir

            neighboringStations.append([j, dist, dir])

            #print(dv, stations[i], 'to', stations[j])
            #print(dist, dir)

        neighboringStations = sorted(neighboringStations, key=lambda os: os[1])[0:otherStations] # sort closest to most far away (limit to required)

        print(neighboringStations)
        # continue # for cpp

        mergedSet = pd.merge(
        combinedData[stations[i]]['data'].add_suffix('_0'),
        combinedData[stations[neighboringStations[0][0]]]['data'].loc[:, combinedData[stations[neighboringStations[0][0]]]['data'].columns != 'Radar_{Tot}' ].add_suffix('_1'),
        left_index=True, right_index=True, copy=False
        )
        for n in range(1,otherStations):
            mergedSet = pd.merge(mergedSet,
                                 combinedData[stations[neighboringStations[n][0]]]['data'].loc[:, combinedData[stations[neighboringStations[n][0]]]['data'].columns != 'Radar_{Tot}' ].add_suffix('_' + str(n+1)),
                                 left_index=True, right_index=True, copy=False)

        print(mergedSet.head()) # 0 is current station, 1+n is neighboring n (in input values)

        sLen = len(mergedSet)

        offset = 0
        i = 0
        while i < (previousTimes + offset):
            if (mergedSet.iloc[i][mergedSet.columns != "Radar_{Tot}_0"].isnull().any()):
                offset = i
            i += 1


        print("Starting at", i)
        rfr = 0
        while i < (sLen - predictAhead):
            # Invalid radar
            if (mergedSet.iloc[i + predictAhead]['Radar_{Tot}_0'] != mergedSet.iloc[i + predictAhead]['Radar_{Tot}_0']):
                i += 1
                continue

            # Invalid station
            if (mergedSet.iloc[i][mergedSet.columns != "Radar_{Tot}_0"].isnull().any()):
                # look at next steps beyond deciding offset
                print("NaN in input", i)
                j = i + 1
                while j < (i + 1 + previousTimes):
                    if (mergedSet.iloc[j][mergedSet.columns != "Radar_{Tot}_0"].isnull().any()):
                        i = j
                    j += 1
                i += previousTimes + 1
                print("moved to", i, j)
                continue

            if (i % 8064 == 0): # every 4 weeks
                print(mergedSet.index[i])

            ## valid set found
            cInput = [k[l] for k in neighboringStations for l in [1,2]]
            for j in range(i, i - previousTimes - 1, -1):  # check i as wel as previous ones
                for ciu in mergedSet.columns[mergedSet.columns != 'Radar_{Tot}_0']:
                    cInput.append(mergedSet.iloc[j][ciu])


            if (mergedSet.iloc[i + predictAhead]['Radar_{Tot}_0'] > threshold):
                inputArrHeavy.append(cInput)
                if mergedSet.iloc[i]['Radar_{Tot}_0'] > threshold:
                    rfr += 1
            else:
                inputArrNonHeavy.append(cInput)

            i+=1
        print(rfr)

    # save
    #np.savetxt('neural_net_data\\raw_multi_station_input_below.csv', inputArrNonHeavy)
    #np.savetxt('neural_net_data\\raw_multi_station_input_above.csv', inputArrHeavy)

    return inputArrHeavy,inputArrNonHeavy

if __name__ == "__main__":

    ## Create data for NN (raw)

    #generateCombinedData()

    suited_stations = ["Capelle", "Delfshaven", "Lansingerland", "Ommoord", "Ridderkerk", "Oost", "SpaansePolder", 'Rijnhaven']

    print("Generating training data")
    trainingIni = radar_data.stringToDateTime("2015/02/24 00:00")
    trainingEnd = radar_data.stringToDateTime("2017/02/25 00:00")
    sDict = loadCombinedData(suited_stations)
    for s in sDict.keys():
        sDict[s]['data'] = sDict[s]['data'][(sDict[s]['data'].index >= trainingIni) & (sDict[s]['data'].index < trainingEnd)]
    heavy,notHeavy = generateInputOutputMultipleStations(combinedData=sDict, otherStations=3, threshold=8/12, predictAhead=3, previousTimes=2)
    print('Input #1 has NaN', np.isnan(heavy).any(), 'length:', len(heavy))
    print('Input #2 has NaN', np.isnan(notHeavy).any(), 'length:', len(notHeavy))
    np.savetxt("neural_net_data\\raw_3+1_station_-10+15min_train_non_heavy.csv", notHeavy)
    np.savetxt("neural_net_data\\raw_3+1_station_-10+15min_train_heavy.csv", heavy)


    print("Generating testing data")
    testingIni = radar_data.stringToDateTime("2017/02/24 00:00")
    testingEnd = radar_data.stringToDateTime("2018/02/25 00:00")
    sDict = loadCombinedData(suited_stations)
    for s in sDict.keys():
        sDict[s]['data'] = sDict[s]['data'][(sDict[s]['data'].index >= testingIni) & (sDict[s]['data'].index < testingEnd)]
    heavy,notHeavy = generateInputOutputMultipleStations(combinedData=sDict, otherStations=3, threshold=8/12, predictAhead=3, previousTimes=2)
    print('Input #1 has NaN', np.isnan(heavy).any(), 'length:', len(heavy))
    print('Input #2 has NaN', np.isnan(notHeavy).any(), 'length:', len(notHeavy))
    np.savetxt("neural_net_data\\raw_3+1_station_-10+15min_test_non_heavy.csv", notHeavy)
    np.savetxt("neural_net_data\\raw_3+1_station_-10+15min_test_heavy.csv", heavy)




