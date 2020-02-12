import utils
import numpy

class BGRFuncFilter(object):
    def __init__(self, vFunc = None, bFunc = None, gFunc = None, rFunc = None,dtype = numpy.uint8) :
        length = numpy.iinfo(dtype).max + 1
        self._bLookupArray = utils.createLookupArray(utils.createCompositeFunc(bFunc, vFunc), length)
        self._gLookupArray = utils.createLookupArray(utils.createCompositeFunc(gFunc, vFunc), length)
        self._rLookupArray = utils.createLookupArray(utils.createCompositeFunc(rFunc, vFunc), length)
        
    def apply(self, src, dst) :
        """Apply the filter with a BGR source/destination."""
        b, g, r = cv2.split(src)
        utils.applyLookupArray(self._bLookupArray, b, b)
        utils.applyLookupArray(self._gLookupArray, g, g)
        utils.applyLookupArray(self._rLookupArray, r, r)
        cv2.merge([ b, g, r ], dst)

class BGRCurveFilter(BGRFuncFilter):
    def __init__(self, vPoints = None, bPoints = None,gPoints = None, rPoints = None, dtype = numpy.uint8):
        BGRFuncFilter.__init__(self, utils.createCurveFunc(vPoints), utils.createCurveFunc(bPoints), utils.createCurveFunc(gPoints), utils.createCurveFunc(rPoints), dtype)

class BGRPortraCurveFilter(BGRCurveFilter):
    def __init__(self, dtype = numpy.uint8):
        BGRCurveFilter.__init__(self,
        vPoints = [ (0, 0), (23, 20), (157, 173), (255, 255) ],
        bPoints = [ (0, 0), (41, 46), (231, 228), (255, 255) ],
        gPoints = [ (0, 0), (52, 47), (189, 196), (255, 255) ],
        rPoints = [ (0, 0), (69, 69), (213, 218), (255, 255) ],
        dtype = dtype)