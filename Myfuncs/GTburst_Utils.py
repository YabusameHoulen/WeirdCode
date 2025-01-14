import math
import numpy as np

# Angles of NaI detectors in spacecraft coordinates
# (from Meegan et al.)
# [zenith, azimuth]
DetDir = {}
DetDir["n0"] = [20.58, 45.89]
DetDir["n1"] = [45.31, 45.11]
DetDir["n2"] = [90.21, 58.44]
DetDir["n3"] = [45.24, 314.87]
DetDir["n4"] = [90.27, 303.15]
DetDir["n5"] = [89.79, 3.35]
DetDir["n6"] = [20.43, 224.93]
DetDir["n7"] = [46.18, 224.62]
DetDir["n8"] = [89.97, 236.61]
DetDir["n9"] = [45.55, 135.19]
DetDir["na"] = [90.42, 123.73]
DetDir["nb"] = [90.32, 183.74]
# Angles of BGO detectors, just to mean they are in two opposite directions
DetDir["b0"] = [90.0, 0.00]
DetDir["b1"] = [90.0, 180.00]
# the LAT is at 0,0
# DetDir["LAT-LLE"] = [0.0, 0.0]
# DetDir["LAT"] = [0.0, 0.0]


def getDetectorAngle(ra_scx, dec_scx, ra_scz, dec_scz, sourceRa, sourceDec, detector):
    if detector in list(DetDir.keys()):
        t = DetDir[detector][0]
        p = DetDir[detector][1]
        ra, dec = getRaDec(ra_scx, dec_scx, ra_scz, dec_scz, t, p)
        return getAngularDistance(sourceRa, sourceDec, ra, dec)
    else:
        raise ValueError("Detector %s is not recognized" % (detector))


def getAngularDistance(ra1, dec1, ra2, dec2):
    # Vincenty formula, stable also at antipodes

    lon1 = np.deg2rad(ra1)
    lat1 = np.deg2rad(dec1)
    lon2 = np.deg2rad(ra2)
    lat2 = np.deg2rad(dec2)

    sdlon = np.sin(lon2 - lon1)
    cdlon = np.cos(lon2 - lon1)
    slat1 = np.sin(lat1)
    slat2 = np.sin(lat2)
    clat1 = np.cos(lat1)
    clat2 = np.cos(lat2)

    num1 = clat2 * sdlon
    num2 = clat1 * slat2 - slat1 * clat2 * cdlon
    denominator = slat1 * slat2 + clat1 * clat2 * cdlon

    return np.rad2deg(np.arctan2(np.sqrt(num1**2 + num2**2), denominator))


def getThetaPhi(ra_scx, dec_scx, ra_scz, dec_scz, RA, DEC):
    v0 = getVector(RA, DEC)
    vx = getVector(ra_scx, dec_scx)
    vz = getVector(ra_scz, dec_scz)
    vy = Vector(vz.cross(vx))

    theta = math.degrees(v0.angle(vz))
    phi = math.degrees(math.atan2(vy.dot(v0), vx.dot(v0)))
    if phi < 0:
        phi += 360
    return (theta, phi)


def getRaDec(ra_scx, dec_scx, ra_scz, dec_scz, theta, phi):
    vx = getVector(ra_scx, dec_scx)
    vz = getVector(ra_scz, dec_scz)

    vxx = Vector(vx.rotate(phi, vz))
    vy = Vector(vz.cross(vxx))

    vzz = vz.rotate(theta, vy)
    # print(vzz)
    ra = math.degrees(math.atan2(vzz[1], vzz[0]))
    dec = math.degrees(math.asin(vzz[2]))

    if ra < 0:
        ra += 360.0
    return ra, dec


def getVector(ra, dec):
    ra1 = math.radians(ra)
    dec1 = math.radians(dec)

    cd = math.cos(dec1)

    return Vector([math.cos(ra1) * cd, math.sin(ra1) * cd, math.sin(dec1)])


class Vector(object):
    def __init__(self, array):
        self.vector = np.array(array)

    def rotate(self, angle, axisVector):
        ang = math.radians(angle)
        matrix = self._getRotationMatrix(axisVector.vector, ang)
        # print matrix
        return np.dot(matrix, self.vector)

    def cross(self, vector):
        return np.cross(self.vector, vector.vector)

    def _getRotationMatrix(self, axis, theta):
        axis = axis / np.sqrt(np.dot(axis, axis))
        a = np.cos(theta / 2)
        b, c, d = -axis * np.sin(theta / 2)
        return np.array(
            [
                [
                    a * a + b * b - c * c - d * d,
                    2 * (b * c + a * d),
                    2 * (b * d - a * c),
                ],
                [
                    2 * (b * c - a * d),
                    a * a + c * c - b * b - d * d,
                    2 * (c * d + a * b),
                ],
                [
                    2 * (b * d + a * c),
                    2 * (c * d - a * b),
                    a * a + d * d - b * b - c * c,
                ],
            ]
        )

    def norm(self):
        return np.linalg.norm(self.vector)

    def dot(self, vector):
        return np.dot(self.vector, vector.vector)

    def angle(self, vector):
        return math.acos(
            np.dot(self.vector, vector.vector) / (self.norm() * vector.norm())
        )


def getBoundingCoordinates(lon, lat, radius):
    """
    Finds the smallest "rectangle" which contains the given Region Of Interest.
    It returns lat_min, lat_max, dec_min, dec_max. If a point has latitude
    within lat_min and lat_max, and longitude within dec_min and dec_max,
    it is possibly contained in the ROI. Otherwise, it is certainly NOT
    within the ROI.
    """
    radLat = np.deg2rad(lat)
    radLon = np.deg2rad(lon)

    radDist = np.deg2rad(radius)

    minLat = radLat - radDist
    maxLat = radLat + radDist

    MIN_LAT = np.deg2rad(-90.0)
    MAX_LAT = np.deg2rad(90.0)
    MIN_LON = np.deg2rad(-180.0)
    MAX_LON = np.deg2rad(180.0)

    if minLat > MIN_LAT and maxLat < MAX_LAT:
        pole = False

        deltaLon = np.arcsin(np.sin(radDist) / np.cos(radLat))

        minLon = radLon - deltaLon
        maxLon = radLon + deltaLon

        if minLon < MIN_LON:
            minLon += 2.0 * np.pi
        if maxLon > MAX_LON:
            maxLon -= 2.0 * np.pi

        # In FITS files the convention is to have longitude from 0 to 360, instead of
        # -180,180. Correct this
        if minLon < 0:
            minLon += 2.0 * np.pi
        if maxLon < 0:
            maxLon += 2.0 * np.pi
    else:
        pole = True
        # A pole is within the ROI
        minLat = max(minLat, MIN_LAT)
        maxLat = min(maxLat, MAX_LAT)
        minLon = 0
        maxLon = 2.0 * np.pi

    # Inversion can happen due to boundaries, so make sure min and max are right
    # minLatf,maxLatf             = min(minLat,maxLat),max(minLat,maxLat)
    # minLonf,maxLonf             = min(minLon,maxLon),max(minLon,maxLon)

    return (
        np.rad2deg(minLon),
        np.rad2deg(maxLon),
        np.rad2deg(minLat),
        np.rad2deg(maxLat),
        pole,
    )


################################ simple extended utils #####################################
from astropy.io import fits  # type: ignore


def get_ra_dec(rsp_dir: str):
    rsp_hdu = fits.open(rsp_dir)
    RA_OBJ = rsp_hdu[0].header["RA_OBJ"]
    DEC_OBJ = rsp_hdu[0].header["DEC_OBJ"]
    return RA_OBJ, DEC_OBJ


def get_dets_angles(trigdat_dir: str, one_rsp_dir: str, chosen_angle=361, nai_num=3)->dict[str,float]:
    """
    calculate the dets pointing angle

    choose angles less than chosen_angle and

    keep the nai_num dets with the nearest angle separation
    (with at least one bgo det)
    """
    
    trigdat_hdu = fits.open(trigdat_dir)
    GRB_NAME = trigdat_hdu[0].header["OBJECT"]
    DET_MASK = trigdat_hdu[0].header["DET_MASK"]
    RA_SCX = trigdat_hdu[0].header["RA_SCX"]
    DEC_SCX = trigdat_hdu[0].header["DEC_SCX"]
    RA_SCZ = trigdat_hdu[0].header["RA_SCZ"]
    DEC_SCZ = trigdat_hdu[0].header["DEC_SCZ"]
    # show the det_masks provided in the file
    # print(GRB_NAME, "Det_Mask in trigdet file: ", DET_MASK)
    RA_OBJ, DEC_OBJ = get_ra_dec(one_rsp_dir)
    ### choose one bgo detector with the nearest angle separation
    bgo_nearest = "b0"
    bgo_angle = 360
    nai_dets_angle = dict()
    for det in DetDir.keys():
        angle = getDetectorAngle(RA_SCX, DEC_SCX, RA_SCZ, DEC_SCZ, RA_OBJ, DEC_OBJ, det)

        if det.startswith("b"):
            if angle < bgo_angle:
                bgo_angle = angle
                bgo_nearest = det
        else:
            if angle < chosen_angle:
                nai_dets_angle[det] = angle

    nai_dets = sorted(nai_dets_angle, key=nai_dets_angle.get)
    if not nai_dets:
        print(f"\033[31mthe chosen_angle {chosen_angle} is too small, no nai dets chosen !!!\033[0m")
    nai_dets = nai_dets[0:nai_num]
    chosen_dets_angle = {
        det: nai_dets_angle[det] for det in nai_dets if det in nai_dets_angle
    }

    if bgo_nearest not in chosen_dets_angle:
        chosen_dets_angle[bgo_nearest] = bgo_angle

    return chosen_dets_angle


if __name__ == "__main__":
    ### 对字典排序的testcase
    my_dict = {"a": 10, "b": 20, "c": 15, "d": 6}
    print(sorted(my_dict, key=my_dict.get))
    my_new_dict = {v: k for k, v in my_dict.items()}
    print(sorted(my_new_dict))
