"""
Adaptation of a script from
  http://stackoverflow.com/questions/7490491/capture-embedded-google-map-image-with-python-without-using-a-browser

to download and stitch static Google maps together.

"""

from __future__ import print_function
from PIL import Image
import urllib
import StringIO
from math import log, log10, exp, tan, atan, pi, ceil, floor
import matplotlib.pyplot as plt
from matplotlib import ticker
import glob


key = 'AIzaSyBJXJaxTH08YYvmcBkt7MZaAvqPqW2G8SM' # belongs to msbandstra@lbl.gov

EARTH_RADIUS = 6378137.
EQUATOR_CIRCUMFERENCE = 2. * pi * EARTH_RADIUS
INITIAL_RESOLUTION = EQUATOR_CIRCUMFERENCE / 256.0
ORIGIN_SHIFT = EQUATOR_CIRCUMFERENCE / 2.0

# Image scale and maximum size in pixels
SCALE = 1
MAXSIZE = 640


def latlon_to_pixels(lat, lon, zoom):
    mx = (lon * ORIGIN_SHIFT) / 180.0
    my = log(tan((90 + lat) * pi/360.0)) / (pi/180.0)
    my = (my * ORIGIN_SHIFT) / 180.0
    res = INITIAL_RESOLUTION / (2**zoom)
    px = (mx + ORIGIN_SHIFT) / res
    py = (my + ORIGIN_SHIFT) / res
    return px, py


def pixels_to_latlon(px, py, zoom):
    res = INITIAL_RESOLUTION / (2**zoom)
    mx = px * res - ORIGIN_SHIFT
    my = py * res - ORIGIN_SHIFT
    lat = (my / ORIGIN_SHIFT) * 180.0
    lat = 180 / pi * (2*atan(exp(lat*pi/180.0)) - pi/2.0)
    lon = (mx / ORIGIN_SHIFT) * 180.0
    return lat, lon


def aspect(lat1, lat2, lon1, lon2, zoom):
    """Aspect for imshow. Script automatically figures out the right ordering
    for the bounds."""
    # automatically figure out correct lat/lon bounds
    ullat = max([lat1, lat2])
    lrlat = min([lat1, lat2])
    ullon = min([lon1, lon2])
    lrlon = max([lon1, lon2])
    ul = latlon_to_pixels(ullat, ullon, zoom)
    lr = latlon_to_pixels(lrlat, lrlon, zoom)
    # return (ullat - lrlat) / (lrlon - ullon) * (lr[1] - ul[1]) / (ul[0] - lr[0])
    return (lrlon - ullon) / (ullat - lrlat) * (lr[1] - ul[1]) / (ul[0] - lr[0])


def image(lat1, lat2, lon1, lon2, zoom):
    """Low-level script to retrieve google map. Script automatically figures
    out the right ordering for the bounds."""
    # automatically figure out correct lat/lon bounds
    ullat = max([lat1, lat2])
    lrlat = min([lat1, lat2])
    ullon = min([lon1, lon2])
    lrlon = max([lon1, lon2])

    # convert all these coordinates to pixels
    ulx, uly = latlon_to_pixels(ullat, ullon, zoom)
    lrx, lry = latlon_to_pixels(lrlat, lrlon, zoom)

    # calculate total pixel dimensions of final image
    dx, dy = lrx - ulx, uly - lry

    # calculate rows and columns
    cols, rows = int(ceil(dx/MAXSIZE)), int(ceil(dy/MAXSIZE))

    # calculate pixel dimensions of each small image
    bottom = 120
    largura = int(ceil(dx/cols))
    altura = int(ceil(dy/rows))
    alturaplus = altura + bottom

    final = Image.new('RGB', (int(dx), int(dy)))
    for x in range(cols):
        for y in range(rows):
            dxn = largura * (0.5 + x)
            dyn = altura * (0.5 + y)
            latn, lonn = pixels_to_latlon(ulx + dxn, uly - dyn - bottom/2, zoom)
            position = ','.join((str(latn), str(lonn)))
            print(x, y, position)
            urlparams = urllib.urlencode({'center': position,
                                          'zoom': str(zoom),
                                          'size': '%dx%d' % (largura, alturaplus),
                                          'maptype': 'satellite',
                                          'sensor': 'false',
                                          'scale': SCALE,
                                          'key':key,
                                          'sensor':'false'})
            url = 'http://maps.google.com/maps/api/staticmap?' + urlparams
            try:
                f = urllib.urlopen(url)
                im = Image.open(StringIO.StringIO(f.read()))
                # try:
                #     imgs = glob.glob('google_map_*.png')
                #     nums = [
                #         int(img.split('google_map_')[-1].split('.png')[0])
                #         for img in imgs]
                #     if len(nums) > 0:
                #         num = max(nums) + 1
                #     else:
                #         num = 0
                #     im.save('google_map_{:03d}.png'.format(num))
                # except:
                #     print('Could not save image')
                final.paste(im, (int(x*largura), int(y*altura)))
            except:
                print('Warning: unable to access Google Maps')
    return final


def plot(lat1, lat2, lon1, lon2, zoom, lat_fmt='auto', lon_fmt='auto'):
    """Plot a map from google maps for the box bounded by the lat and lon
    arguments. Script automatically figures out the right ordering for
    the bounds."""
    # automatically figure out correct lat/lon bounds
    ullat = max([lat1, lat2])
    lrlat = min([lat1, lat2])
    ullon = min([lon1, lon2])
    lrlon = max([lon1, lon2])
    asp = aspect(lat1, lat2, lon1, lon2, zoom)
    final = image(lat1, lat2, lon1, lon2, zoom)
    plt.imshow(final, extent=[ullon, lrlon, lrlat, ullat],
        aspect=asp)
    plt.xlim(ullon, lrlon)
    plt.ylim(lrlat, ullat)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')

    # format the latitude and longitude tick labels
    if lat_fmt == 'auto':
        dlat = abs(lat1 - lat2)
        decimal_places = 1 - int(floor(log10(dlat))) + 3
        if decimal_places < 0: decimal_places = 0
        total_places = decimal_places + 4
        lat_fmt = '{{:+{:d}.{:d}f}}'.format(total_places, decimal_places)
    if lon_fmt == 'auto':
        dlon = abs(lon1 - lon2)
        decimal_places = 1 - int(floor(log10(dlon))) + 3
        if decimal_places < 0: decimal_places = 0 
        total_places = decimal_places + 5
        lon_fmt = '{{:+{:d}.{:d}f}}'.format(total_places, decimal_places)
    def latitude_formatter(x, p):
        return lat_fmt.format(x)
    def longitude_formatter(x, p):
        return lon_fmt.format(x)
    plt.gca().get_xaxis().set_major_formatter(ticker.FuncFormatter(
        longitude_formatter))
    plt.gca().get_yaxis().set_major_formatter(ticker.FuncFormatter(
        latitude_formatter))


def auto_zoom(lat1, lat2, lon1, lon2, tiles=1):
    """Determine a good zoom for the desired level of detail. The detail is
    specified by how many tiles are required to be loaded."""
    assert(tiles >= 1)
    # automatically figure out correct lat/lon bounds
    ullat = max([lat1, lat2])
    lrlat = min([lat1, lat2])
    ullon = min([lon1, lon2])
    lrlon = max([lon1, lon2])

    for zoom in range(20):
        # convert all these coordinates to pixels
        ulx, uly = latlon_to_pixels(ullat, ullon, zoom)
        lrx, lry = latlon_to_pixels(lrlat, lrlon, zoom)

        # calculate total pixel dimensions of final image
        dx, dy = lrx - ulx, uly - lry

        # calculate rows and columns
        cols, rows = int(ceil(dx/MAXSIZE)), int(ceil(dy/MAXSIZE))

        # detail level determined by how many tiles are loaded
        n_tiles = rows * cols
        if n_tiles > tiles:
            break
    return (zoom - 1)


def plot_autozoom(lat1, lat2, lon1, lon2, tiles=1, lat_fmt='auto',
    lon_fmt='auto'):
    """Plot a map from google maps for the box bounded by the lat and lon
    arguments. Script automatically figures out the right ordering for
    the bounds and the correct zoom level."""
    zoom = auto_zoom(lat1, lat2, lon1, lon2, tiles=tiles)
    plot(lat1, lat2, lon1, lon2, zoom, lat_fmt=lat_fmt, lon_fmt=lon_fmt)


if __name__ == '__main__':
    lonlatbox = [-122.29, -122.25, 37.79, 37.81]
    zoom = 15
    asp = aspect(lonlatbox[2], lonlatbox[3], lonlatbox[0], lonlatbox[1], zoom)
    print('aspect: ', asp)
    final = image(lonlatbox[2], lonlatbox[3], lonlatbox[0], lonlatbox[1], zoom)

    plt.figure()
    plt.imshow(final, extent=lonlatbox, aspect=asp)
    plt.plot(lonlatbox[0:2], lonlatbox[2:4], 'bo-')
    plt.xlim(lonlatbox[0:2])
    plt.ylim(lonlatbox[2:4])
    plt.show()

    plt.figure()
    plot(lonlatbox[2], lonlatbox[3], lonlatbox[0], lonlatbox[1], zoom,
        lat_fmt='{:+.3f}', lon_fmt='{:+.3f}')
    plt.plot(lonlatbox[0:2], lonlatbox[2:4], 'bo-')
    plt.show()

    plt.figure()
    LAT = 36.99897
    LON = -109.045167
    plot(LAT + 0.001, LAT - 0.001, LON - 0.001, LON + 0.001, 14)
    plt.plot([LON], [LAT], 'bo-', ms=10)
    plt.show()

    plt.figure()
    LAT = 36.99897
    LON = -109.045167
    plot(LAT + 0.001, LAT - 0.001, LON - 0.001, LON + 0.001, 17)
    plt.plot([LON], [LAT], 'bo-', ms=10)
    plt.show()

    plt.figure()
    LAT = 36.99897
    LON = -109.045167
    plot(LAT + 0.001, LAT - 0.001, LON - 0.001, LON + 0.001, 20)
    plt.plot([LON], [LAT], 'bo-', ms=10)
    plt.show()

    plt.figure()
    LAT = 36.99897
    LON = -109.045167
    plot_autozoom(LAT + 0.010, LAT - 0.010, LON - 0.015, LON + 0.015,
        tiles=1)
    plt.plot([LON], [LAT], 'bo-', ms=10)
    plt.show()

    plt.figure()
    LAT = 36.99897
    LON = -109.045167
    plot_autozoom(LAT + 0.010, LAT - 0.010, LON - 0.015, LON + 0.015,
        tiles=4)
    plt.plot([LON], [LAT], 'bo-', ms=10)
    plt.show()

    plt.figure()
    LAT = 36.99897
    LON = -109.045167
    plot_autozoom(LAT + 0.010, LAT - 0.010, LON - 0.015, LON + 0.015,
        tiles=7)
    plt.plot([LON], [LAT], 'bo-', ms=10)
    plt.show()
