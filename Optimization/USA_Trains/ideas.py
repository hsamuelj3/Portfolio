# Filename: ideas.py
# Author: Joseph Heal
# Date created: 2023.06.14
# Description: Train route exploration ideas
# Source: https://simplemaps.com/data/us-cities
# Important columns: 0,2,6,7,8,14
# 0: City name
# 2: State ID (Two letters)
# 6: Latitude
# 7: Longitude
# 8: Population
# 14: Ranking
# Range: 0-16
# We only care about cities which are greater than longitude -100 (East of -100 which is 100W and on the more populated side of America)
# Maybe include texan cities? 
# Pick a few cities which can be used as starting examples, but then give the option of picking. 
# Try between two and find cities which are close to the line?


# New to try:                >>>>>
# Take midpoint, evaluate closest cities, add the weight into the raiting
import random
import math
from math import radians, cos, sin, asin, sqrt
import csv
import numpy as np
from scipy.optimize import minimize, NonlinearConstraint
import scipy
import matplotlib.pyplot as plt 
import time

# import mlrose
# import sklearn

# If I want to plot stuff...
import matplotlib as mpl
# from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt


# functions (Which I have defined or found since 2023.06.15):
def havSinDistance(lat1, lng1, lat2, lng2, miles=False):
    # use decimal degrees (as given) to find distance between points on earth (planet) within 0.5%
    # Chagne this to just call the index of the file... use the row numbers? and then just use __[row][latIndex] or row[latIndex] in the formulas where latIndex is 6 or 7 for longitude
    r=6371 # radius of the earth in km
    lat1=radians(lat1)
    lat2=radians(lat2)
    lat_dif=lat2-lat1
    lng_dif=radians(lng2-lng1)
    a=sin(lat_dif/2.0)**2+cos(lat1)*cos(lat2)*sin(lng_dif/2.0)**2
    d=2*r*asin(sqrt(a))
    if miles:
        return d * 0.621371 # return miles
    else:
        return d # return km
    
def havSinDistance_Indexes(index1, index2, cities, miles=False):
    # use decimal degrees (as given) to find distance between points on earth (planet) within 0.5%
    # Chagne this to just call the index of the file... use the row numbers? and then just use __[row][latIndex] or row[latIndex] in the formulas where latIndex is 6 or 7 for longitude
    r=6371 # radius of the earth in km
    lat1=radians(float(cities[index1][6]))
    lat2=radians(float(cities[index2][6]))
    lat_dif=lat2-lat1
    lng1 = float(cities[index1][7])
    lng2 = float(cities[index2][7])
    lng_dif=radians(lng2-lng1)
    a=sin(lat_dif/2.0)**2+cos(lat1)*cos(lat2)*sin(lng_dif/2.0)**2
    d=2*r*asin(sqrt(a))
    if miles == True:
        return d * 0.621371 # return miles
    else:
        return d # return km

def findClosestCity_index(cityList,cityIndex):
    closestCities = []
    # tmpCityIndex = 0
    tmp = []
    for i in range(len(cityList)):
        if i == cityIndex:
            pass
        else:
            distance = havSinDistance(float(cityList[cityIndex][6]),float(cityList[cityIndex][7]),float(cityList[i][6]),float(cityList[i][7]))
            tmp = [distance, i, cityList[i][0]]
            closestCities.append(tmp)
    closestCities.sort() # This sorts luckily by the first value in each row
    return closestCities

def findClosestCity_latlon(cityList,lat,lon):
    closestCities = []
    tmp = []
    for i in range(len(cityList)):
        distance = havSinDistance(lat,lon,float(cityList[i][6]),float(cityList[i][7]))
        tmp = [distance, i, cityList[i][0]]
        closestCities.append(tmp)
    closestCities.sort()
    return closestCities

def newClosestCity_latlon(cityList,lat,lon,distLimit = 80):
    closestCities = []
    tmp = []
    for i in range(len(cityList)):
        distance = havSinDistance(lat,lon,float(cityList[i][6]),float(cityList[i][7]))
        if distance < distLimit:
            tmp = [distance, i, cityList[i][0]]
            closestCities.append(tmp)
        
    closestCities.sort()
    return closestCities
# function which finds closest cities to line (close = within 10% of the total distance?) 
# Can I interpolate latitude and longitude coordinates to form a line? 
#LERP: 

def slerpLine(index1,index2,cities): # use to check if a point is on the line between two cities? 
    # convert lat/lon to p(x,y,z)
    r = 6371
    lat1 = radians(float(cities[index1][6]))
    lat2 = radians(float(cities[index2][6]))
    lon1 = radians(float(cities[index1][7]))
    lon2 = radians(float(cities[index2][7]))
    newXYZ_1 = [cos(lon1)*cos(lat1), sin(lon1)*cos(lat1), sin(lat1)] * r
    newXYZ_2 = [cos(lon2)*cos(lat2), sin(lon2)*cos(lat2), sin(lat2)] * r

    # Find theta using cos inverse of the dot product
    tmpDot = np.dot(newXYZ_1,newXYZ_2)
    # print(f"Here's the dotted value... {tmpDot}\nIs it too large?")
    while tmpDot > 1:
        tmpDot -= math.pi
        # if tmpDot > 0:
        #     pass
        # else:
        #     tmpDot -= 2 * math.pi
        # print(f"The current value of tmpDot is {tmpDot}")
    # tmpDot = float(input("Please enter what this in a smaller number of radians if this"))
    theta = np.arccos(tmpDot)
    # print(f"This is theta: {theta}")
    # theta = float(input("Is this correct? If so enter it again"))
    # Line given by the formula Slerp(p0,p1,t)= ( p0*sin((1-t)*theta) + p1*sin(t*theta) ) / sin(theta)
    
    # 3d view of that line? 
    if True == False:
        mpl.rcParams['legend.fontsize'] = 10

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        # theta = np.linspace(-4 * np.pi, 4 * np.pi, 100)
        t = np.linspace(-1,1,100)
        # z = np.linspace(-2, 2, 100)
        # r = z**2 + 1
        # x = r * np.sin(theta)
        # y = r * np.cos(theta)
        x = r*(newXYZ_1[0]*np.sin((1.0-t)*theta) + newXYZ_2[0]*np.sin(t*theta) ) / sin(theta)
        y = r*(newXYZ_1[1]*np.sin((1.0-t)*theta) + newXYZ_2[1]*np.sin(t*theta) ) / sin(theta)
        z = r*(newXYZ_1[2]*np.sin((1.0-t)*theta) + newXYZ_2[2]*np.sin(t*theta) ) / sin(theta)
        ax.plot(x, y, z, label='parametric curve')
        ax.plot(newXYZ_1[0],newXYZ_1[1],newXYZ_1[2])
        
        # # Make data
        # u = np.linspace(0, 2 * np.pi, 100)
        # v = np.linspace(0, np.pi, 100)
        # x = r * np.outer(np.cos(u), np.sin(v))
        # y = r * np.outer(np.sin(u), np.sin(v))
        # z = r * np.outer(np.ones(np.size(u)), np.cos(v))

        # # Plot the surface
        # ax.plot_surface(x, y, y, alpha = 0.5)

        # # Set an equal aspect ratio
        ax.set_aspect('equal')
        ax.legend()

        plt.show()
    
    return 

def lerpLine(index1,index2,cities,printLatLon= False): # Prints the linear line on the plot, calculates the great circle distance
    # Lat/Lon details: 
    lat1 = float(cities[index1][6])
    lat2 = float(cities[index2][6])
    lon1 = float(cities[index1][7])
    lon2 = float(cities[index2][7])
    
    # Slope
    m = (lat2-lat1)/(lon2-lon1)
    x = np.linspace(lon1,lon2,200)
    y = m * (x - lon2) + lat2
    # Distance: 
    # distance = havSinDistance(float(cities[index1][6]),float(cities[index1][7]),float(cities[index2][6]),float(cities[index2][7]))
    distance = havSinDistance_Indexes(index1,index2,cities)
    linDistance = np.sqrt((lat2-lat1)**2 + (lon2-lon1)**2) * 100.111

    # Print them: 
    if printLatLon == True:
        print(f"The lon/lat (1) for {cities[index1][0]} is: {float(cities[index1][7])}//{float(cities[index1][6])} with a population of {int(cities[index1][8])/1000} thousand") 
        print(f"The lon/lat (2) for {cities[index2][0]} is: {float(cities[index2][7])}//{float(cities[index2][6])} with a population of {int(cities[index2][8])/1000} thousand") 
        print(f"The distance between {cities[index1][0]} and {cities[index2][0]} is {distance} km") # Optional print?

    # print(f"The distance of the straight line estimate is {round(linDistance,2)}km") # Poor linear estimate if I have haversine
    # plot
    # plt.figure() # Don't have to do this because it's already plotting in one figure
    plt.plot(x,y,'k')
    
    # plt.show() # same reason as above

    data = [m,lon2,lat2] # use this?
    return distance

def closestLinear(index1,index2,cities,cityIndexClose): # Make it a function of the data which is returned from lerpLine? #########>??????????
    # Lat/Lon details: 
    lat1 = float(cities[index1][6])
    lat2 = float(cities[index2][6])
    lon1 = float(cities[index1][7])
    lon2 = float(cities[index2][7])
    # 
    m = (lat2-lat1)/(lon2-lon1)

    x = np.linspace(lon1,lon2,200)
    y = m * (x - lon2) + lat2

    # new x values for the closest city
    y2 = float(cities[cityIndexClose][6])
    x2 = float(cities[cityIndexClose][7])
    xClosest = (m * (y2 - lat2 - m * lon2) + x2) / (1 - m**2)

    y = m * (xClosest - lon2) + lat2
    print(f"The closest X is {xClosest},{y}")
    plt.plot(xClosest,y,'ko')
    return

def slerpLineClosest(cityIndexStart, cityIndexEnd, cityIndexCheck): #currently doesn't work
    # Do cross products to find the vector which should be on the line... 
    # Convert from lat/lon to x,y,z
    # Cross start and end = New
    # Cross New and to Check = CheckNew
    # Cross CheckNew with Check to get potential point

    # Convert to lat/lon and do distance formula with city check

    # return distance and where the point is in lat/lon

    return 

def latlon_mid(latA,latB,lonA,lonB): # Used to compress other code in potentialRoute_old
    # Take midpoint distance of both lat and lon
    lonDif = abs(lonA - lonB)
    latDif  = abs(latA - latB)
    if latA > latB:
        latMid = latB + 0.5 * latDif
    if latB > latA:
        latMid = latA + 0.5 * latDif
    if lonA > lonB:
        lonMid = lonB + 0.5 * lonDif
    if lonB > lonA:
        lonMid = lonA + 0.5 * lonDif
    # midpoint = [latMid,lonMid]
    return latMid,lonMid

def lerpLine_pointFinder(latA,latB,lonA,lonB, sectionStopNum, totalStops): # Used in Potential route
    # Assign linear stuff
    if lonA > lonB: # Go from East to west always
        lon1 = lonA # lon1 is always less negative
        lon2 = lonB
        lat1 = latA # longitude just goes with the longitude of the initial city
        lat2 = latB 
    else: 
        lon1 = lonB
        lon2 = lonA
        lat1 = latB
        lat2 = latA
    # Slope
    m = (lat2-lat1)/(lon2-lon1)
    x = np.linspace(lon1,lon2,200)
    y = m * (x - lon2) + lat2
    
    lonFraction = float(sectionStopNum) / float(totalStops) * abs(lon2 - lon1)
    lon = lon1 - lonFraction # Fraction along the distance
    lat = m * (lon - lon2) + lat2
    return lat, lon

def potentialRoute_old(cities, cityA, cityB,inBetween = 3): # Not working
    latStart = float(cities[cityA][6])
    latEnd = float(cities[cityB][6])
    lonStart = float(cities[cityA][7])
    lonEnd = float(cities[cityB][7])

    totalDistance = float()

    # Add figure to create a map and cities # Put this in potential route as well... or just there
    fig = plt.figure()
    ax = fig.add_subplot()
    for i in range(len(cities)):
        plt.plot(float(cities[i][7]),float(cities[i][6]),'*')
    ax.set_aspect('equal')
    # show initial distance "as the crow flies"
    havDistance = havSinDistance_Indexes(cityA,cityB,cities)
    print(havDistance)

    indexList_path = [cityA]
    
    # Midpoint stuff 
            # first midpoint 
    latA = latStart
    latB = latEnd
    lonA = lonStart
    lonB = lonEnd
    '''# # Take midpoint distance of both lat and lon
    # lonDif = abs(lonA - lonB)
    # latDif  = abs(latA - latB)
    # if latA > latB:
    #     latMid = latB + 0.5 * latDif
    # if latB > latA:
    #     latMid = latA + 0.5 * latDif
    # if lonA > lonB:
    #     lonMid = lonB + 0.5 * lonDif
    # if lonB > lonA:
    #     lonMid = lonA + 0.5 * lonDif   '''
    latMid,lonMid = latlon_mid(latA,latB,lonA,lonB)
    realLatMid = latMid
    realLonMid = lonMid
    plt.plot(lonMid,latMid,'k*')
    
    # first midpoint done use it in closest list    
    closestList = findClosestCity_latlon(cities,latMid,lonMid)
    closestIndex = closestList[0][1]
    indexList_path.append(closestIndex)
    midpoint = [float(cities[closestIndex][6]),float(cities[closestIndex][7])] # Form lat,lon -> y,x -> 
    plt.plot(midpoint[1],midpoint[0],'k*')
            
            
            # Second midpoint (loop?)
    latA = latStart
    latB = realLatMid
    lonA = lonStart
    lonB = realLonMid
    latMid,lonMid = latlon_mid(latA,latB,lonA,lonB)
    plt.plot(lonMid,latMid,'k*')
    closestList = findClosestCity_latlon(cities,latMid,lonMid)
    closestIndex = closestList[0][1]
    indexList_path.append(closestIndex)
    midpoint = [float(cities[closestIndex][6]),float(cities[closestIndex][7])]
    plt.plot(midpoint[1],midpoint[0],'k*')
            
            
            # Last midpoint
    latA = realLatMid
    latB = latEnd
    lonA = realLonMid
    lonB = lonEnd
    latMid,lonMid = latlon_mid(latA,latB,lonA,lonB)
    plt.plot(lonMid,latMid,'k*')
    closestList = findClosestCity_latlon(cities,latMid,lonMid)
    closestIndex = closestList[0][1]
    indexList_path.append(closestIndex)
    midpoint = [float(cities[closestIndex][6]),float(cities[closestIndex][7])]
    plt.plot(midpoint[1],midpoint[0],'k*')

    # limit of how much longer than the crow distance decided based on stops? decide after... How to divide the route for even numbers of in between stops? do thirds of same process? 
    #??? 
    # start by finding the closest big city to 0.5 along the slerp line? what defines big? 
        # find slerp line
        # enter t = 0.5 and find closest big city around there (do closest but limit by 1 degree of lat/lon)
        # use haversine distance loop between cities within +- one degree of lat and lon
    
    # example thing: just for a quick test before it is in a loop
    # lerpLine(cityA,int(closestList[0][1]),cities)
    # lerpLine(int(closestList[0][1]),cityB,cities)

    indexList_path.append(cityB)
    # Loop through the list of indices
    tmpList = []
    for i in indexList_path:
        lonValue = float(cities[i][7])
        tmp = [lonValue,i]
        tmpList.append(tmp)
    tmpList.sort()
    print(f"Here is the sorted list with longitude values:\n",tmpList)


    for i in range(len(tmpList)-1):
        dist = lerpLine(tmpList[i][1],tmpList[i+1][1],cities)    
        print(f"Line from {tmpList[i][1]} to {tmpList[i+1][1]}")
        if dist < 1000:
            totalDistance += dist
    # dist = lerpLine(tmpList[0],tmpList[-1],cities)   
    # totalDistance += dist
    print(f"Line from {tmpList[0]} to {tmpList[-1]}")
    print(tmpList)
    print(f"The total distance is {totalDistance} km") 
    
def readFile_popSize(outline = False, decision = 'no'): # Reads the file size and creates a list for it
    pop_100 = 0
    popList_100 = []
    popEast_100 = 0
    popEastList_100 = []
    cityNameList = []

    if outline == False:
        # Getting the data from the csv into a list which I can work with:
        # file = list(csv.reader(open("uscities.csv")))
        file = list(csv.reader(open("Optimization/USA_Trains/uscities.csv")))
        citiesList = []
        for row in file:
            citiesList.append(row)
        print(len(file))
        print(len(citiesList))
        # Finding number of cities which are over a certain population, adding them to a list and making a separate list of 100k+ east of the 100th parallel
        doubleLimit = 'no'
        if decision == 'yes':
            while True:
                doubleLimit = input("Would you like to have an upper ` as well as a lower limit? (yes/no) ").lower()
                if doubleLimit == 'yes' or doubleLimit == 'no':
                    break
                else:
                    doubleLimit == input("Please enter yes or no")
        
        if doubleLimit == 'yes':
            populationLimit_Lower = float(input("How large of cities would you like to include? (lower limit in 100k increments) "))
            populationLimit_Lower = populationLimit_Lower * 1e5
            populationLimit_Upper = float(input("How large of cities would you like to exclude? (upper limit in 100k increments) "))
            populationLimit_Upper = populationLimit_Upper * 1e5
        else: 
            populationLimit_Lower = float(input("How large of cities would you like to include? (lower limit in 100k increments) "))
            populationLimit_Lower = populationLimit_Lower * 1e5

        for i in range(1,len(file)):
            if int(citiesList[i][8]) >= populationLimit_Lower:
                pop_100 += 1
                popList_100.append(citiesList[i][:-2])
            if doubleLimit == 'yes':
                if int(citiesList[i][8]) >= populationLimit_Lower and int(citiesList[i][8]) <= populationLimit_Upper and \
                    float(citiesList[i][7]) > -100 and float(citiesList[i][6]) > 20 : # 6 is to eliminate cities below florida...
                    popEast_100 += 1
                    popEastList_100.append(citiesList[i][:-2])
                    cityNameList.append(citiesList[i][1])
            else:
                if int(citiesList[i][8]) >= populationLimit_Lower and float(citiesList[i][7]) > -100 and float(citiesList[i][6]) > 20 : # 6 is to eliminate lower cities.
                    popEast_100 += 1
                    popEastList_100.append(citiesList[i][:-2])
                    cityNameList.append(citiesList[i][1])
        
        print(f"{pop_100} cities with populations greater than {populationLimit_Lower}")
        print(f"{popEast_100} cities with populations greater than {populationLimit_Lower} and east of 100")

        return popEastList_100, cityNameList
    
    else:
        # file = list(csv.reader(open("uscities.csv")))
        file = list(csv.reader(open("Optimization/USA_Trains/uscities.csv")))
        citiesList = []
        for row in file:
            citiesList.append(row)
        populationLimit_Lower = 1e5
        for i in range(1,len(file)):
            if int(citiesList[i][8]) >= populationLimit_Lower:
                popList_100.append(citiesList[i][:-2])
            if int(citiesList[i][8]) >= populationLimit_Lower and float(citiesList[i][7]) > -100 and float(citiesList[i][6]) > 20:
                popEastList_100.append(citiesList[i][:-2])

        return popEastList_100

def plotSilhouette(): # Used to plot the image of the USA using cities as the silhouette, could be better if I wanted to include an actual outline with state lines, etc
    list_outline = readFile_popSize(outline= True)
    for i in range(len(list_outline)):
        plt.plot(float(list_outline[i][7]),float(list_outline[i][6]),'y*')
    return

def potentialRoute_withSetup(): # Gives options for potential route, city size, repeat, loop, etc
    '''
    Basic function which asks for the settings of the route finder function
    - how large of cities (bounds or not --> use another small function to read the data like below)
    - which cities (start/end)
!!! - midpoint cities (eventually) !!!
    - Loop or not (just one time) -->    ,loopBool = True): put the loop in here? 
    - print stuff or not
    - plot for each or all at the end?

    - more! I know there is more, I just can't think of it
    '''
    popEastList_100, cityNameList = readFile_popSize()
    
    # Ask for two cities and search for their indices
    print(f"This will be a route between two cities. ")
    print(f"What are the two cities you would like to search for?")
    cityA_tmp = input("Enter the first city: ")
    cityA_index = cityNameList.index(cityA_tmp.title()) # Search for the index of the city. 
    print(f"The city A index is {cityA_index}")
    cityB_tmp = input("Enter the second city: ")
    cityB_index = cityNameList.index(cityB_tmp.title())
    print(f"The city B index is {cityB_index}")
    # Print distance between

    # Stop number: 
    loopBool = str(input("Would you like to run a loop for different numbers of stops? (Yes/No) ").title())
    printVal = input("Would you like to print details? (True/False) ").title()
    if loopBool == 'Yes': 
        print(f"Consider how much distance there is between these two cities")
        inBetweenNumLow = int(input("Please enter the minimum number of stops you would like between the start and end cities: "))
        inBetweenNumHigh = int(input("Please enter the maximum number of stops you would like between the start and end cities: "))
        print()

        for i in range(inBetweenNumLow,inBetweenNumHigh+1):
            print(f"This is for {i} stops in between:")
            potentialRoute(popEastList_100,cityA= cityA_index,cityB= cityB_index,inBetween= i, printValue= printVal)
            print()
    else:
        inBetweenNum = int(input("Enter how many stops you would like between the two cities: "))
        potentialRoute(popEastList_100, cityA= cityA_index, cityB= cityB_index, inBetween= inBetweenNum,printValue= printVal)    
    
    plt.show()


    return

def potentialRoute(cities, cityA, cityB,inBetween = 3,printValue = False):
    ''' 
    This function shows a route from cityA to cityB with a number of stops inbetween which by default is 3. 
    This will print out the number of stops (total) and the latitude and longitude of each of the stops. 
    The total distance will also be displayed as the sum of the distances between stops. 
    The print functions are found in the LerpLine function 

    This function also displays all cities with 100k+ population and east of 100 degrees longitude
    
    '''

    # # closest cities to a random point can also be used to find closest to a midpoint of line segments

    # Maybe just go through the distance across the line and divide it by however many stops it should be? 
    # Try this again... 
    latStart = float(cities[cityA][6])
    latEnd = float(cities[cityB][6])
    lonStart = float(cities[cityA][7])
    lonEnd = float(cities[cityB][7])
    # Necessary? 
    latA = latStart
    latB = latEnd
    lonA = lonStart
    lonB = lonEnd
    # Plot with points:
    fig = plt.figure()
    ax = fig.add_subplot()
    plotSilhouette()
    ax.set_aspect('equal')
    # find points:
    tmpList = []
    stopNum = inBetween + 1 # The number of stops + start and stop
    print(f"The number of total stops is {stopNum+1} (including starting and stopping stations)")
    for i in range(0,stopNum+1):
        lat,lon = lerpLine_pointFinder(latStart,latEnd,lonStart,lonEnd, i, stopNum)
        tmp = [lon,lat]
        tmpList.append(tmp)
        # print(f"Here's the value of i: {i}")
    # print(f"Here's the list of latitude, longitude:\n",tmpList,"\nOr here's another one:")
    # for i in tmpList:
    #     print(i)

    indicesPath = []
    dist = 0.0

    # Plot the points:
    for i in range(len(tmpList)):
        nearestCities = findClosestCity_latlon(cities,tmpList[i][1],tmpList[i][0])
        counter = 0
        # nearestIndex = int(nearestCities[counter][1])
        while True: # Make sure you don't pick one you've already used. 
            nearestIndex = int(nearestCities[counter][1])
            if len(indicesPath) < 1:
                break
            if nearestIndex == indicesPath[-1]:
                counter += 1
            else: 
                break

        # print(f"Lon:",cities[nearestIndex][7],"Lat:",cities[nearestIndex][6])
        plt.plot(float(cities[nearestIndex][7]),float(cities[nearestIndex][6]),'k*')
        indicesPath.append(nearestIndex)
    # stops = [] # Add in a stops list which will print the stops?
    for i in range(len(tmpList)-1):
        # stops.append()
        dist += lerpLine(indicesPath[i],indicesPath[i+1],cities,printLatLon=printValue)
    print()
    print(f"The route goes from {cities[cityA][1]}, {cities[cityA][2]} to {cities[cityB][1]}, {cities[cityB][2]}")
    distance = havSinDistance_Indexes(cityA,cityB,cities)
    print(f"The distance between these two cities is {distance} km (as the crow flies)")
    print(f"The stops are:")
    for i in range(len(indicesPath)):
        index_tmp = indicesPath[i]
        print(f"- {cities[index_tmp][1]}, {cities[index_tmp][2]} Population: {cities[index_tmp][8]}")
    print(f"Total distance of the path is {dist}")
    # Add lerpline to each of these

    # plt.show()
    return distance

def potentialRoute_Optimization(cities,):
    # First get a rough idea of length from a great circle distance estimate
    cityA_tmp = input("Enter the first city: ")
    cityA = cityNameList.index(cityA_tmp.title()) # Search for the index of the city. 
    # print(f"The city A index is {cityA}")
    cityB_tmp = input("Enter the second city: ")
    cityB = cityNameList.index(cityB_tmp.title())
    # print(f"The city B index is {cityB}")
    distance = havSinDistance_Indexes(cityA,cityB,cities=cities)
    print(f"The direct distance is {distance} km")

    latStart = float(cities[cityA][6])
    latEnd = float(cities[cityB][6])
    lonStart = float(cities[cityA][7])
    lonEnd = float(cities[cityB][7])

    # Take straight line estimate
    # reduce list to cities within a certain distance of the path 
    # or nearest 25-30 cities  along the path (or more if it doesn't include enough of them)
    # optimize route using the reduced list --> updating the list during the route optimization comes later, this is just a first step (!!!)
    
    stopDistance = int(input("Enter the average distance between each stop: "))
    inBetween = int(distance // stopDistance)
    print(f"The number of stops is {inBetween}")
    answer = input("Would you like to change this value? (yes/no) ").lower()
    while True:
        if answer == 'yes':
            inBetween = int(input("Please enter a new number of stops: "))
            break
        elif answer == 'no':
            print(f"The number of stops will be {inBetween}")
            break
        else:
            answer = input("That is not a correct answer. Enter yes or no ").lower()

    # Straight line estimate is going to be with stops every ~200 miles, (How many times +-1 that goes into it)
    # Loop for nearest cities from each stop
    # nearest 5-10 cities from each stop --> Arguments for cities list
    
    # loop for finding lat lon of points along the path: 
    
    
    tmpList = []
    stopNum = inBetween + 1 # The number of stops + start and stop
    print(f"The number of total stops is {stopNum+1} (including starting and stopping stations)")
    for i in range(0,stopNum+1):
        lat,lon = lerpLine_pointFinder(latStart,latEnd,lonStart,lonEnd, i, stopNum)
        tmp = [lon,lat]
        tmpList.append(tmp)
    tmpCitiesList = []
    citiesList = []
    for i in range(len(tmpList)):
        closestCities = findClosestCity_latlon(cities,tmpList[i][1],tmpList[i][0]) 
        for i in range(len(closestCities)):
            if closestCities[i][0] > 80:
                pass
            else: 
                tmpCitiesList.append(closestCities[i])
        # tmpCitiesList = tmpCitiesList[0:4]
        ''' Argument for how many cities to consider is currently 5 change to distance? '''
        print(len(tmpCitiesList))
        citiesList.append(tmpCitiesList)
    indexList = []
    for i in range(len(citiesList)):
        for j in range(len(citiesList[0])):
            if havSinDistance_Indexes(cityA,citiesList[i][j][1],cities) < 80.0 or havSinDistance_Indexes(cityA,citiesList[i][j][1],cities) < 80.0:
                pass
            else: 
                # Add in check to see if it's already in the list
                indexList.append(citiesList[i][j][1])
    
    # Print stuff: cities and indices
    # for i in citiesList:
    #     print(i)
    # for i in indexList:
    #     print(i,end=' / ')
    print(len(indexList))
    
    # Do optimize minimize picking a group of points with the number of stops chosen above from the list
    # Can it be used like that? pick from x integer values? not repeating? 
    
    '''
    MILP can be used to do integers, I just need to have a distance function as well as a value of the stop function

    Need to find some way to fix certain points as well, ie. require them to be in a certain route

    MILP should be okay with relatively large numbers of possibilities, right? 
    
    '''


    myArgs = (cities, cityA, cityB, stopNum, indexList) # Arguments for the cities

    return

def route_OptFunction(indexes, cities, cityA, cityB, stopNum): # is it necessary? or can we start with just adding in the list of populated places we want to include
    
    # Can use int() to convert float indexes to integer indexes... For loop changing each one? 

    # Function to begin:
    #   sum of distance
    #   negative sum of population percentage (1% or a bit more?)
    distanceTotal = 0.0
    popTotal = 0

    for i in range(len(indexes)-1):
        distanceTotal += lerpLine(indexes[i],indexes[i+1],cities,printLatLon=False)
    for j in range(len(indexes)):
        popTotal += cities[indexes[i]][8]
    
    popPercentage = 0.01*popTotal # 1 % ridership is a good goal, but is also an estimate for what we would like to see. 
    
    value = distanceTotal*10 - popPercentage*0.1

    return value

def cityPairs(cityList,startIndex, lowerLimit=80, upperLimit=200):
    '''
    Find the gravity or worth of city pairs based on distance and population, or different qualities. 
    
    T_ij = P_i * P_j / (d_ij)^2
    City_i (P_i) <---- d_i ----> City_j (P_j)

    '''

    # cityList, nameList = readFile_popSize() # Filter cities above a certain value
    # cityList needs to be a different list which has been reduced to the cities which are near the city... Or at least the cities which 
    # are along the route. I have code which does that already, right? 
    gravityList = []

    for i in range(len(cityList)):
        # for j in range(len(cityList)):
        cityA = startIndex
        cityB = i

        cityAlon = cityList[startIndex][7]
        cityBlon = cityList[i][7]
        # Check that we are still moving the correct way... This assumes that we always start east to west...
        # I will need to change this
        '''
        Change this to work any direction!!! Or just make it start east to west each time...?  That may be easier... 
        '''
        if cityAlon < cityBlon: # if more west, keep going
            distance = havSinDistance_Indexes(cityA,cityB,cityList,True)
            if distance < upperLimit and distance > lowerLimit: # If far enough away keep going
                popA = float(cityList[cityA][8])
                popB = float(cityList[i][8])
                gravity = popA * popB / distance**2 / 10000
                gravityList.append([np.round(gravity,2), np.round(distance,0), cityA, cityB, cityList[i][1]])
                # Add more so it becomes based on different values. 
                # Incorporate monetary stuff? 

    gravityList.sort()

    return gravityList

''' This is all from testing... Not completely necessary, but steps were good
start_time = time.time()
# Getting the data from the csv into a list which I can work with:
# file = list(csv.reader(open("uscities.csv")))
file = list(csv.reader(open("Optimization/USA_Trains/uscities.csv")))
citiesList = []
for row in file:
    citiesList.append(row)
print(len(file))
print(len(citiesList))
print(f"{time.time()-start_time} seconds")

# Finding number of cities which are over 100k population, adding them to a list and making a separate list of 100k+ east of the 100th parallel

pop_100 = 0
popList_100 = []
popEast_100 = 0
popEastList_100 = []
cityNameList = []

populationLimit = int(input("How large of cities would you like to include? "))

start_time = time.time()
for i in range(1,len(file)):
    if int(citiesList[i][8]) >= populationLimit:
        pop_100 += 1
        popList_100.append(citiesList[i][:-2])
    if int(citiesList[i][8]) >= populationLimit and float(citiesList[i][7]) > -100 and float(citiesList[i][6]) > 20:
        popEast_100 += 1
        popEastList_100.append(citiesList[i][:-2])
        cityNameList.append(citiesList[i][1])

print(f"{pop_100} cities with populations greater than {populationLimit}")
print(f"{popEast_100} cities with populations greater than {populationLimit} and east of 100")
print(f"{time.time()-start_time} seconds\n")

# find distances from a city index. use new york? It's index 0 so it's easy
# print(f"{popEastList_100[0]}")
# print(f"{citiesList[0][:-2]}\n") # Printing what the row names are, just to see it easier... Not always necessary

# Calculate order of closest cities
# distancesList = findClosestCity_index(popEastList_100,0)

# print(distancesList[37]) # just printing a random number from the list to show it works.
# Print out results - or do this to see closest 10 or any range
# for row in range(10,15):
#     print(distancesList[row])
    # print(popEastList_100[int(distancesList[row][1])])

# Now on to find a route between and check closest cities ever 50-100 miles? 

# Treat it as linear with 1 lat or lon = 111.111 km, just to get a quick estimate. 
# It's honestly possible to do everything with longitude and latitude and then multiply it to kilometers at the end?
# Derivative of distance^2 would be very simple compared to the other one... Definitely quicker to check or assume

# Add figure to create a map and cities # Put this in potential route as well... or just there
# fig = plt.figure()
# ax = fig.add_subplot()
# for i in range(len(popEastList_100)):
#     plt.plot(float(popEastList_100[i][7]),float(popEastList_100[i][6]),'*')
# ax.set_aspect('equal')
# lerpLine(0,15,popEastList_100) # return data to be used with y = m * (x - lon2) + lat2 (data is in this order, [m, lon2, lat2])???

# Actual route

# !!! Add some way to exlude lakes and the ocean and go around stuff? 
# !!! What way to include stuff? 

# Ask for two cities and search for their indices
print(f"This will be a route between two cities. ")
print(f"What are the two cities you would like to search for?")
cityA_tmp = input("Enter the first city: ")
cityA_index = cityNameList.index(cityA_tmp.title()) # Search for the index of the city. 
print(f"The city A index is {cityA_index}")
cityB_tmp = input("Enter the second city: ")
cityB_index = cityNameList.index(cityB_tmp.title())
print(f"The city B index is {cityB_index}")
# Print distance between

# Stop number: 
print(f"Consider how much distance there is between these two cities")
inBetweenNumLow = int(input("Please enter the minimum number of stops you would like between the start and end cities: "))
inBetweenNumHigh = int(input("Please enter the maximum number of stops you would like between the start and end cities: "))
print()
for i in range(inBetweenNumLow,inBetweenNumHigh+1):
    print(f"This is for {i} stops in between:")
    potentialRoute(popEastList_100,cityA= cityA_index,cityB= cityB_index,inBetween= i,printValue=True)
    print()
plt.show()
'''

# potentialRoute_withSetup() # This does not have any optimization, it just has brief options (two cities, step numbers, print options)

popEastList_100, cityNameList = readFile_popSize()
# potentialRoute_Optimization(popEastList_100)

# # Sort of test thing:
# index = 0 # 
# cities = [] # List of possible options
# cityA = index
# cityB = index

# indices = [] # Start with A, end with B

def function(indices,cityA,cityB,cities):
    distance = 0
    pop = 0
    cityValue = 0
    potential = 0
    for i in range(len(indices)-1):
        if i == len(indices[-1]):
            distance += havSinDistance_Indexes(indices[i],indices[i+1],cities) # distance between the cities
        else:
            pass
        cityPop = [0] # List of populations
        eachValue = [0] # list of gravities, 
        cityPotential = [0] # something? 
        pop += cityPop[i] # population from each city
        cityValue += eachValue[i] # Value for each city
        potential += cityPotential[i]

    # Combine results

    # 

    return

# While loop to determine 

def testFunction(cities): # newOptimizationFunction is hopefully what this will be...
    '''
    This function will lead you through the optimization of a route. 
    This will also ask for population limits, distance limits and a bit more
    '''
    
    '''
    Starts with one city, finds a route and narrows down the city list based on the cities near the route
    The first stop consists of a choice of 3-5 based on highest gravity and other ratings... 
    The next stops will be based on these 3-5 first stops. Find highest gravity combinations based on the first and second stop
    keep the highest 3 and go from the second stop to the next stop and keep the top 3 routes
    go from the 3rd stop to next 3-5 cities (from each option) and keep highest 
    repeat until at location. 
    show options based on shortest route and hhighest gravity -> problem is that it currently is just based on population not taking other 
    aspects into account. 
    
    '''
    # Find city A and city B: Just use chihcago and new york for now
    cityA = 0 # New York
    cityB = 1 # Chicago
    # Find latitudes:
    latA = float(cities[cityA][6])
    latB = float(cities[cityB][6])
    lonA = float(cities[cityA][7])
    lonB = float(cities[cityB][7])

    # Find cities along the path to reduce city list length
    inBetween = 50 # Use more  stops in between cities and reduce the distance away from the line -> straighter? 
    tmpList = []
    stopNum = inBetween + 2 # The number of stops + start and stop
    # print(f"The number of total stops is {stopNum} (including starting and stopping stations)")
    for i in range(0,stopNum):
        lat,lon = lerpLine_pointFinder(latA,latB,lonA,lonB, i, stopNum)
        tmp = [lon,lat]
        tmpList.append(tmp) 
    
    # tmpList is a list of the latitude and longitude locations of points along the "direct" route
    
    # Find list of closest cities along the path: within 80 km of the direct path.
    pathList = []
    distLimit = 120
    totalOptions = 0
    for i in range(len(tmpList)):
        potentialCities = newClosestCity_latlon(cities,tmpList[i][1],tmpList[i][0],distLimit=distLimit)[0:10]
        # print(len(potentialCities)) # How many potential options do we add
        totalOptions += len(potentialCities)
        pathList.append(potentialCities)
    # for j in pathList:
    #     print(j)
    

    # Clean up the list so it doesn't have repeats? Or can I just have it in order and limit the longitude based on past city and distance?

    dirtyList = [popEastList_100[cityA]] # Add as part of the algorithm
    for i in range(len(pathList)):
        for j in range(len(pathList[i])):
            tmpDirtyItem = pathList[i][j]
            tmpLon = float(popEastList_100[tmpDirtyItem[1]][7])

            if tmpLon > lonB and tmpLon < lonA:
                dirtyList.append(popEastList_100[tmpDirtyItem[1]]) # +[pathList[i][j][0]] adds distance from the point...  
    dirtyList.append(popEastList_100[cityB]) # Add finding the correct "cityB" as part of the algorithm based on
    print(f"Total non-unique options from stops for limit of {distLimit} km is {len(dirtyList)}")
    
    # Clean up the dirty List: 
    cleanList = []
    topRoutes_Indices = {'1':[],'2':[],'3':[]}
    for i in dirtyList:
        if cleanList.count(i) == 0: 
            cleanList.append(i)
        else: 
            # print(dirtyList.index(i))
            # key = 'no'
            # while key == 'no':
            #     key = str(input("Enter anything to keep going"))
            pass

    print(f"Total unique options from stops for limit of {distLimit} km is {len(cleanList)}")
    # print(f"Clean List:")
    # for i in cleanList:
    #     print(i)    
    # start of Algorithm to create optimal path using gravity: 
    stops = 5
    # first step: # go from starting city to next closest cities:
        # Find closest cities between 100 and 200 miles away
    startPoint= cityA
    upLimit = 200
    lowLimit = 80
    list1 = cityPairs(cleanList,startIndex=startPoint,lowerLimit= lowLimit, upperLimit= upLimit)
    print(f"{len(list1)} gravity pairs between {lowLimit} and {upLimit} kilometers away from cityA")
    print(f"{list1}\n")
    
    
    tempList1 = cleanList.copy()

    # Second step: 
        # Find closest cities between 100 and 200 miles away from the next cities

    # Create the new empty lists -> create new citypairs starting from the 0 to 1
    dictStuff = {('Route ' + str(i)):[] for i in range(len(list1))}

    # Print Route steps for the first one
    for i in range(len(dictStuff)):
        currentKey = 'Route ' + str(i)
        dictStuff[currentKey].append(list1[i])
        print(dictStuff[currentKey])
    
    print()

    for i in range(len(dictStuff)):
        currentKey = 'Route ' + str(i)
        tempCityPairsList = cityPairs(tempList1,dictStuff[currentKey][0][3])
        # dictStuff[currentKey].append(tempCityPairsList[-1])
        print(f"TempCityPairs list for {currentKey}")
        print(tempCityPairsList)
        # print(f"The length of tempCityPairsList is {len(tempCityPairsList)}")
        if len(tempCityPairsList) == 0:
            # print(f"Printing complete list (of none...)")
            # print(tempCityPairsList)
            pass
        else:
            lastNum = int(len(tempCityPairsList)-1)
            # print(f"The value of lastItem is {lastItem}")
            # print(tempCityPairsList[lastNum])
            lastItem = tempCityPairsList[lastNum]
            dictStuff[currentKey].append(lastItem)
        print()
    print(f"Print all routes which are longer:\n")

    for i in range(len(dictStuff)):
        currentKey = 'Route ' + str(i)
        if len(dictStuff[currentKey]) > 1:
            # dictStuff[currentKey].append(list1[i])
            print(f"{currentKey}: {dictStuff[currentKey]}")
            # print(len(dictStuff[currentKey]))

    # Nexxt SteP: 

    for i in range(len(dictStuff)):
        currentKey = 'Route ' + str(i)
        if len(dictStuff[currentKey]) > 1:
            tempCityPairsList = cityPairs(tempList1,dictStuff[currentKey][-1][3])
            # dictStuff[currentKey].append(tempCityPairsList[-1])
            print(f"TempCityPairs list for {currentKey}")
            print(tempCityPairsList)
            # print(f"The length of tempCityPairsList is {len(tempCityPairsList)}")
            if len(tempCityPairsList) == 0:
                # print(f"Printing complete list (of none...)")
                # print(tempCityPairsList)
                pass
            else:
                lastNum = int(len(tempCityPairsList)-1)
                # print(f"The value of lastItem is {lastItem}")
                # print(tempCityPairsList[lastNum])
                lastItem = tempCityPairsList[lastNum]
                dictStuff[currentKey].append(lastItem)
            print()
        
    print(f"Print all routes which are longer:\n")

    for i in range(len(dictStuff)):
        currentKey = 'Route ' + str(i)
        if len(dictStuff[currentKey]) > 2:
            # dictStuff[currentKey].append(list1[i])
            print(f"{currentKey}: {dictStuff[currentKey]}")

    # Nexxt Nexxt steP:
    for i in range(len(dictStuff)):
        currentKey = 'Route ' + str(i)
        if len(dictStuff[currentKey]) > 1:
            tempCityPairsList = cityPairs(tempList1,dictStuff[currentKey][-1][3])
            # dictStuff[currentKey].append(tempCityPairsList[-1])
            print(f"TempCityPairs list for {currentKey}")
            print(tempCityPairsList)
            # print(f"The length of tempCityPairsList is {len(tempCityPairsList)}")
            if len(tempCityPairsList) == 0:
                # print(f"Printing complete list (of none...)")
                # print(tempCityPairsList)
                pass
            else:
                lastNum = int(len(tempCityPairsList)-1)
                # print(f"The value of lastItem is {lastItem}")
                # print(tempCityPairsList[lastNum])
                lastItem = tempCityPairsList[lastNum]
                dictStuff[currentKey].append(lastItem)
            print()
        
    print(f"Print all routes which are longer:\n")

    for i in range(len(dictStuff)):
        currentKey = 'Route ' + str(i)
        if len(dictStuff[currentKey]) > 3:
            # dictStuff[currentKey].append(list1[i])
            print(f"{currentKey}: {dictStuff[currentKey]}")


    '''
    Currently I simply take the largest gravity, but I think I need to start making lists of the options. 
    Can I just append the list with an item that has lots of entries and then pick from them? 
    Yes, but I will need to change how the starting and ending cities are picked. 
    Perhaps the list just creates all the options in the order that they can go, it says the starting and ending cities on it in any case
    Then I would just need to create one big dictionary with stops instead of routes. 
    The routes can come later? Or perhaps there are two different options. 
    This would give me something like: routes = {"stop 1": [[],[],[]], "stop 2" [[],[],[],[],[],[],[]],} and so on. 
    This owuld be a large diamond shaped list because it would contain fewer and fewer options as you progressed. 
    But there would be differnt numbers of stops because the routes are different lengths and some have shorter or longer decisions. 

    I could use the same pattern as before, I would just need to do a loop which lengthened the list from each previous stop's optionss
    
    '''


    '''
    Add in a check to ensure that it's not adding to a completed route
    '''

    # Do the clean list here, and determine the stops, etc or just make it into a function
    print(f"\n\n\n\n\n")
    print(f"Starting a new loop, trying to get this all in one loop to get to Chicago")
    print()
    loopBool = True
    loopCounter = 0
    checkCounter = 0
    loopcheckMax = 0
    loopcheck = 0

    while loopBool == True: 
        print(f"Start of loop {loopCounter}")
        if loopCounter == 0: 
            # CHECK IF THIS IS THE FIRST LOOP, START FROM WHICHEVER ONE, DETERMINE DIRECTION
            # DO BOTH DIRECTIONS? 
            # IF IT"S THE FIRST TIME DO A "GET CLEAN List" functioN? then add the templist1 which is used everywhere
            print(f"In the initial loop check")
            startPoint = cityA
            upLimit = 200
            lowLimit = 80
            list1 = cityPairs(cleanList,startIndex=startPoint,lowerLimit= lowLimit, upperLimit= upLimit)
            # print(f"{len(list1)} gravity pairs between {lowLimit} and {upLimit} kilometers away from cityA")
            # print(f"{list1}\n")
            routeDict = {('Route ' + str(i)):[] for i in range(len(list1))}
            for i in range(len(routeDict)):
                currentKey = 'Route ' + str(i)
                routeDict[currentKey].append(list1[i]) # Add the items to the initial list

            if checkCounter >= 1:
                print(f"Something is wrong")
                break
            checkCounter += 1 # TO ENSURE THIS ONLY HAppens once
        else: 
            for i in range(len(routeDict)):
                currentKey = 'Route ' + str(i)
                itemDimensions = np.size(routeDict[currentKey])
                print(f"dimensions: {itemDimensions}")
                lastIndex = len(routeDict[currentKey]) - 1
                print(f"current Index: {lastIndex}")
                # CHECK IF ALL OF THEM ARE AT the end, if one is not there keep going
                if len(routeDict[currentKey]) != 0:
                    if routeDict[currentKey][lastIndex][3] == len(cleanList): 
                        # I need to change the indices for the check... It's out of range? 
                        print(f"In this random place!!! What to do?")
                        loopBool = False
                        pass
                    else: 
                        tempCityPairsList = cityPairs(cleanList,routeDict[currentKey][lastIndex][3])
                        print(f"TempCityPairs list for {currentKey}")
                        print(tempCityPairsList)
                        # print(f"The length of tempCityPairsList is {len(tempCityPairsList)}")
                        if len(tempCityPairsList) == 0:
                            pass
                        else:
                            lastNum = int(len(tempCityPairsList)-1)
                            lastItem = tempCityPairsList[lastNum]
                            routeDict[currentKey].append(lastItem)
                        print()
        
        # do the loop for each one, if it's already at the end skip it? 
        checks = 0
        for i in routeDict:
            print(routeDict[i])
            if routeDict[i][-1][3] == 25:
                checks += 1
        

        if checks > loopcheckMax:
            loopcheckMax = checks
        elif checks == loopcheckMax:
            loopcheck += 1
        
        if checks == len(routeDict) or loopcheck == 5:
            loopBool = False
            
            # Check if it's at the end, if yes, skip it:
            # if routeDict{"route1"}[-1][-1] == cityB or something like that, then skip it
        
        print(f"Checks = {checks}")
        print(f"At the end of the loop {loopCounter}\n")
        loopCounter += 1 # increment count 
        
        # loopBool = False # CURRENTLY JUST TO EXIT THE LOOP
        # if loopCounter > 5:
        #     loopBool = False
        
    # Plotting stuff: 
    plotBool = True
    if plotBool== True: # Plot silhouette and cities along the route
        fig = plt.figure()
        ax = fig.add_subplot()
        plotSilhouette()
        ax.set_aspect('equal')

        for i in range(len(dirtyList)):
            # nearestIndex = dirtyList[i][]s
            # print(f"Nearest Index = {nearestIndex}")
            plt.plot(float(dirtyList[i][7]),float(dirtyList[i][6]),'k*')
        
        for i in range(len(routeDict)):
            currentKey = 'Route ' + str(i)
            tmpList = routeDict[currentKey]
            print(f"Print Temp list: {tmpList}")
            x = []
            y = []
            for j in range(len(tmpList)-1):
                # dist += lerpLine(indicesPath[i],indicesPath[i+1],cities,printLatLon=printValue) 
                # dist = lerpLine(j,j+1, cleanList)
                ''' Add the indices or lat/lon to x/y list and then display them at the end'''
                print(f"index = {tmpList[j][3]}")
                tmp_index = tmpList[j][3]
                lon_tmp = float(cleanList[tmp_index][7])
                lat_tmp = float(cleanList[tmp_index][6])
                x.append([lon_tmp]) # Longitude
                y.append([lat_tmp]) # Latitude
                
            print(f"x values: {x}")
            print(f"y values: {y}")
            plt.plot(x,y)
            # for i in range(len(tmpList)-1):
            #     # stops.append()
            #     dist += lerpLine(indicesPath[i],indicesPath[i+1],cities,printLatLon=printValue)

        plt.plot(lonB,latB,'b*')
        plt.plot(lonA,latA,'b*')

        # for i in range(len())

        plt.show()

    return

    # # closest cities to a random point can also be used to find closest to a midpoint of line segments

    # Maybe just go through the distance across the line and divide it by however many stops it should be? 
    # # Try this again... 

    # # Plot with points:

    # indicesPath = []
    # dist = 0.0

    # # Plot the points:
    # for i in range(len(tmpList)):
    #     nearestCities = findClosestCity_latlon(cities,tmpList[i][1],tmpList[i][0])
    #     counter = 0
    #     # nearestIndex = int(nearestCities[counter][1])
    #     while True: # Make sure you don't pick one you've already used. 
    #         nearestIndex = int(nearestCities[counter][1])
    #         if len(indicesPath) < 1:
    #             break
    #         if nearestIndex == indicesPath[-1]:
    #             counter += 1
    #         else: 
    #             break

    #     # print(f"Lon:",cities[nearestIndex][7],"Lat:",cities[nearestIndex][6])
    #     plt.plot(float(cities[nearestIndex][7]),float(cities[nearestIndex][6]),'k*')
    #     indicesPath.append(nearestIndex)
    # # stops = [] # Add in a stops list which will print the stops?
    # for i in range(len(tmpList)-1):
    #     # stops.append()
    #     dist += lerpLine(indicesPath[i],indicesPath[i+1],cities,printLatLon=printValue)
    # print()
    # print(f"The route goes from {cities[cityA][1]}, {cities[cityA][2]} to {cities[cityB][1]}, {cities[cityB][2]}")
    # distance = havSinDistance_Indexes(cityA,cityB,cities)
    # print(f"The distance between these two cities is {distance} km (as the crow flies)")
    # print(f"The stops are:")
    # for i in range(len(indicesPath)):
    #     index_tmp = indicesPath[i]
    #     print(f"- {cities[index_tmp][1]}, {cities[index_tmp][2]} Population: {cities[index_tmp][8]}")
    # print(f"Total distance of the path is {dist}")
    # # Add lerpline to each of these

    # # plt.show()
    # return distance


testFunction(popEastList_100)