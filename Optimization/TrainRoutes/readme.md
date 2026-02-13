# Introduction to my Train Project

This is a fun project I made originally to create high-speed rail routes in the United States of America, specifically on the densely populated portion of the continent east of the 100th meridian. This boundary is used because approximately 80% of the US population lives in this area.

With some simple changes this program will be usable in the USA, the UK, and the world, using the data contained in this folder. For the USA I use small cities to create a background map, but I would like to use a different map in the future, both for the USA and the world. This will allow the focus to be on the cities and the routes created.

The idea behind this project was to create train routes which can connect major hubs in the United States - especially shorter routes which are heavy with air traffic. Over certain distances, see the graphic on the website from [The Geography of Transport Systems](https://transportgeography.org/contents/applications/high-speed-rail-systems/breakeven-distances-rail-air-transport/).

## How it works

The program is simple. It will ask you if you want to loop through multiple distances between cities, how many iterations to perform, and how large of cities to include.

The result will be a map, or several, plotting the routes and the cities selected, along with relevant distances and other information.

## Technical Work

As an initial description I will simply say it's a lot of loops to divide up a distance and find good candidates for where things should go. It's an old project, but I want to improve it.

## Credits

I use some data from this paper: <http://www.railway-technical.com/books-papers--articles/high-speed-railway-capacity.pdf>

I use the csv data files for the USA, the UK, and the World file from simplemaps found on [the simplemaps website](https://simplemaps.com/data). I also use their python function for distance on a sphere from their website ([found here](https://simplemaps.com/resources/location-distance)).
