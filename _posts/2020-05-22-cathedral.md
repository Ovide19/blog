---
layout: posts
title: Finding the orientation of the Strasbourg cathedral using OpenCV
published: true
header:
  overlay_image: /assets/images/cathedral.jpg
  overlay_filter: 0.2
tags: [flask, heroku, folium]
date: 2020-05-22
---

Can OpenCV be used to delineate a building from a map, given the building’s coordinates? Let’s find out!

As soon as my buddy Anatolii talked me into checking out [OpenCV](https://opencv.org/), a cross-platform library dedicated to computer vision, I found myself eager to try its nifty contouring tools. 
One of the first applications that came to my mind was to apply those to maps or aerial images in order to gain insight about the properties of historical landmarks, such as their orientation.

Living in Strasbourg, getting started with the cathedral seemed a rather obvious choice, but if you bear with me throughout this post, you should be able to apply a similar strategy to whatever building you are interested into. Granted, it would be easier to determine the nave's orientation using a good old compass, but keep in mind that when I started working on this project, the Covid-19 lockdown was still enforced in my hometown, so going anywhere near the cathedral was not an option.

### 1. Importing libraries

Let's get started! If you haven’t done so already, go ahead and install OpenCV. I strongly suggest that you follow the thorough installation guide provided on Adrian Rosebrock’s [PyImageSearch blog](https://www.pyimagesearch.com/opencv-tutorials-resources-guides/). Once OpenCV is up and running, let’s import it:

```python
import cv2
import numpy as np
import folium
import numpy as np
import selenium.webdriver
import time
```

We will also need a few other libraries: 
- **Folium** was introduced in my recent post about [how to map Covid-19 statistics](https://ovide19.github.io/blog/covidI/). 
- **Selenium WebDriver** is a web framework that will come in handy when trying to convert html pages created using Folium to a static image.


### 2. Creating a map centered on the coordinates of the cathedral

Now that imports are out of the way, let’s write a function creating a map centered on the latitude and longitude of the cathedral:

```python
def create_html_map(name,latitude,longitude):
     custom_map = folium.Map(location=[latitude, longitude],
                   tiles = 'Stamen Terrain',
                   zoom_start=18, control_scale=False)              
     custom_map.save('./'+name+'.html')
     return
```

`create_html_map` is fed the name, latitude and longitude of a building and returns a map centered on the building’s coordinates as a html file. The `zoom_start` parameter is the initial zoom level: a value of 18 basically means that the world map is split into 2^18 tiles, the maximum zoom level allowed by leaflet.

### 3. Converting the html map to a static image

Now let’s convert our html map into a static image:

```python
def create_static_map(name):
     driver = selenium.webdriver.PhantomJS()
     driver.set_window_size(1000, 1000)  # choose a resolution
     driver.get('./'+name+'.html')
     time.sleep(3)
     driver.save_screenshot('./'+name+'.png')
     return
```

This is where Selenium steps in: it allows us to use [PhantomJS](https://phantomjs.org/), a headless browser that opens our html file and takes a screenshot of the map.
To play things safe, `time.sleep(3)` buys PhantomJS 3 seconds to load the html file. This prevents the map from being only partially loaded when the screenshot is taken.

Just be aware that since the development of PhantomJS is currently suspended, Python will throw a `UserWarning` error. No sweat: `create_static_map` will do its job nonetheless, *i.e.* it will turn our html file into a 1000 pixels x 1000 pixels square image of Strasbourg's city center: 

![Contour of Strasbourg cathedral](/blog/assets/images/Strasbourg.png)

### 3. Delineating the contour of the cathedral

Now the fun part begins:

```python
def draw_cathedral_contour(name):
     img = cv2.imread('./'+name+'.png')
     
     # define color ranges
     low_gray = (200, 200, 200)
     high_gray = ( 210, 210, 210 )
     gray_mask = cv2.inRange(img, low_gray, high_gray)
     cv2.imwrite("./gray_mask.jpg",gray_mask)
     # find contours
     contours, hierarchy = cv2.findContours(gray_mask,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

     image_center = np.asarray(img.shape[1::-1]) / 2
     image_center = tuple(image_center.astype('int32'))

     # draw the outline of all contours
     for contour in contours:
          test=cv2.pointPolygonTest(contour, image_center, measureDist=False) 
          if test==1.0:
               cathedral_contour=contour

               background = np.zeros(img.shape).astype(img.dtype)
               cv2.drawContours(background, [cathedral_contour], -1, (255, 255, 255), -1)
               
               #Draw ellipse
               (x,y),(MA,ma),ellipse_angle = cv2.fitEllipse(cathedral_contour)
  
               cv2.ellipse(background, (int(x),int(y)), (np.int(MA),np.int(ma)), ellipse_angle, 
                          np.int(ellipse_angle), np.int(ellipse_angle+360), (255,0,255), 1)

               x1=np.int((int(x) + np.int(MA)*np.sin(ellipse_angle * np.pi / 180.0)))
               y1=np.int((int(y) - np.int(MA)*np.cos(ellipse_angle * np.pi / 180.0)))     
               x2=np.int((int(x) - np.int(MA)*np.sin(ellipse_angle * np.pi / 180.0)))
               y2=np.int((int(y) + np.int(MA)*np.cos(ellipse_angle * np.pi / 180.0))) 
               cv2.line(background, (x1, y1), (x2, y2),(255,0,255),4) 

               cv2.putText(background, name+ ': Orientation ' + str(int(ellipse_angle))+' degrees', (100,50),cv2.FONT_HERSHEY_SIMPLEX , 0.6, (255,0,255),thickness=1)
               cv2.imwrite("./"+name+"_binary.jpg",background)
               return ellipse_angle
```

What's going on here?

Well, `draw_cathedral_contour` first creates a color-based mask. Opening our newly created png image using GIMP or another editor reveals that the dark gray color filling the buildings has a BGR value of (204, 204, 204).  Setting up low_gray to (200, 200, 200) and high_gray to (210, 210, 210) therefore allows us to delineate the buildings. 
Note that if you are using another style than “Stamen Terrain” or working with another mapping tool, you will need to adjust these thresholds accordingly. And if your target is filled with a color other than gray, just keep in mind that OpenCV relies on BGR color instead of RGB!

Our function then delineates all buildings using cv2’s `findContours` built-in method. `findContours` is fed three arguments:

- our `gray_mask`.
- the contour retrieval mode `cv.RETR_TREE`, which not only retrieves all contours but also sorts them hierarchically... Quite usefule when dealing with nested contours!
- the contour approximation method `cv.CHAIN_APPROX_SIMPLE`. This method removes the redundant points in the contour, thus allowing to save memory.

So `findContours` returns the contours of all the buildings in the image… But remember, we are only interested in the cathedral! 

Good news are, thanks to create_html_map, our image is centered on our GPS waypoint, so the cathedral’s contour should encompass the center of the image.
Let’s loop over all contours and find which of those meets this requirements. To do this we use the `pointPolygonTest` built-in method: 

```python
test=cv2.pointPolygonTest(contour, image_center, measureDist=False) 
```

If the contour contains `image_center`, then `test` is equal to 1.

We’re almost there! 

Now that we have nailed down the cathedral's contour, let’s create a black background the size of our original image and let’s display this contour in white:

```python
background = np.zeros(img.shape).astype(img.dtype)
cv2.drawContours(background, [cathedral_contour], -1, (255, 255, 255), -1)
```

We can now find the contour's center, and fit an ellipse around it using openCV’s `fitEllipse` function:

```python
cv2.ellipse(background, (int(x),int(y)), (np.int(MA),np.int(ma)), ellipse_angle, 
np.int(ellipse_angle), np.int(ellipse_angle+360), (255,0,255), 1)
```

This function calculates the ellipse that best fits a set of 2D point in the least-square sense: it returns the center of the ellipse, the semi-minor and semi-major axes and last but not least angle of the rotated rectangle in which the ellipse is inscribed. This angle is what we are interested into: it gives the orientation of the nave. We can simply display it on our final image using OpenCV's `putText` method:

```python
cv2.putText(background, name+ ': Orientation ' + str(int(ellipse_angle))+' degrees', (100,50),cv2.FONT_HERSHEY_SIMPLEX , 0.6, (255,0,255),thickness=1)
```

### 4. Running the program

OK, time for a sanity check!
Let’s first grab the cathedral’s coordinates from [this Wikipedia entry](https://en.wikipedia.org/wiki/List_of_tallest_church_buildings): its latitude is 48.581808 and its longitude is 7.750361. 
We can now feed these coordinates to `create_html_map`, call `create_static_map` and finally compute the angle using `draw_cathedral_contour`:

```python
if __name__ == '__main__':
     name='Strasbourg'
     latitude=48.581808
     longitude=7.750361
     create_html_map(name,latitude,longitude)
     create_static_map(name)
     angle=draw_cathedral_contour(name)
     print(Catheral: 'name')
     print('Orientation:'+str(int(angle))+'°')
```

`draw_cathedral_contour` also saves the binary image of the cathedral's contour as a jpg file:

![Contour of Strasbourg cathedral](/blog/assets/images/Strasbourg_binary.jpg)

Done! Well, before going any further, there's a caveat you should know about: finding the nave's orientation by fitting the cathedral's contour with an ellipse works because the Strasbourg cathedral is roughly symmetrical. If you are interested into a building with an asymmetrical floor plan, you will need to find another way to define its orientation.

Now back to the Strasbourg cathedral. A lot of religious edifices are oriented eastwards, so what's up with the 60°? Is this angle associated with the sun's rising point during a symbolic time of the year, such as an equinox? To clarify the matter, I called my friend Aymeric, who works as a stonemason at the *Fondation de l'Oeuvre de Notre Dame*. It turns out that the solution is much more pragmatic. Strasbourg -or *Argentoratum* as it was called back then- was a former Roman military outpost established in 12 BC:

![Model of Argentoratum](/blog/assets/images/argentoratum.jpg)

Its two main roads were the *via principalis* and the *via praetoria*. The latter is roughly oriented north-northeast (NNE): when the first cathedral was commissioned by the bishop Arbogast by the end of the 7th century, it was naturally aligned along these pre-existing roads!

![Via Principalis and Via Pratoria](/blog/assets/images/argentoratum2.jpg)

 	


























