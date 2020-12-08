# coding: utf-8

from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import cv2
import numpy

from .config import get_config
from .imgutils import (crop, _frame_repr, pixel_bounding_box, _validate_region)
from .logging import ddebug, ImageLogger
from .types import Region


class MotionResult(object):
    """The result from `detect_motion` and `wait_for_motion`.

    :ivar float time: The time at which the video-frame was captured, in
        seconds since 1970-01-01T00:00Z. This timestamp can be compared with
        system time (``time.time()``).

    :ivar bool motion: True if motion was found. This is the same as evaluating
        ``MotionResult`` as a bool. That is, ``if result:`` will behave the
        same as ``if result.motion:``.

    :ivar Region region: Bounding box where the motion was found, or ``None``
        if no motion was found.

    :ivar Frame frame: The video frame in which motion was (or wasn't) found.
    """
    _fields = ("time", "motion", "region", "frame")

    def __init__(self, time, motion, region, frame):
        self.time = time
        self.motion = motion
        self.region = region
        self.frame = frame

    def __bool__(self):
        return self.motion

    def __nonzero__(self):
        return self.__bool__()

    def __repr__(self):
        return (
            "MotionResult(time=%s, motion=%r, region=%r, frame=%s)" % (
                "None" if self.time is None else "%.3f" % self.time,
                self.motion, self.region, _frame_repr(self.frame)))


class FrameDiffer(object):
    """Interface for different algorithms for diffing frames in a sequence.

    Say you have a sequence of frames A, B, C. Typically you will compare frame
    A against B, and then frame B against C. This is a class (not a function)
    so that you can remember work you've done on frame B, so that you don't
    repeat that work when you need to compare against frame C.
    """

    def __init__(self, initial_frame, region=Region.ALL, mask=None,
                 min_size=None):
        self.prev_frame = initial_frame
        self.region = _validate_region(self.prev_frame, region)
        self.mask = mask
        self.min_size = min_size
        if (self.mask is not None and
                self.mask.shape[:2] != (self.region.height, self.region.width)):
            raise ValueError(
                "The dimensions of the mask %s don't match the %s <%ix%i>"
                % (_frame_repr(self.mask),
                   "frame" if region == Region.ALL else "region",
                   self.region.width, self.region.height))

    def diff(self, frame):
        raise NotImplementedError(
            "%s.diff is not implemented" % self.__class__.__name__)


class MotionDiff(FrameDiffer):
    """Compares 2 frames by calculating the color-distance between them.

    The algorithm is a simple color distance calculation. Conceptually, for
    each pair of corresponding pixels in the 2 frames we perform the following
    calculations:

    - Calculate the squared difference separately for each color channel.
    - Sum the 3 results (for the 3 colour channels).
    - Normalise back to the original range (0-255) by dividing by 3 and taking
      the square root.
    - Compare against the specified threshold. Pixel values greater than the
      threshold are counted as differences (that is, motion).

    Then, an "erode" operation removes differences that are only 1 pixel wide
    or high. If any differences remain, the 2 frames are considered different.

    This is the default diffing algorithm for `wait_for_motion`,
    `press_and_wait`, and `find_selection_from_background`.
    """

    def __init__(self, initial_frame, region=Region.ALL, mask=None,
                 min_size=None, threshold=25, erode=True):
        super(MotionDiff, self).__init__(initial_frame, region, mask, min_size)

        self.threshold = threshold

        if isinstance(erode, numpy.ndarray):  # For power users
            kernel = erode
        elif erode:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        else:
            kernel = None
        self.kernel = kernel

    def diff(self, frame):
        imglog = ImageLogger("MotionDiff", region=self.region,
                             min_size=self.min_size,
                             threshold=self.threshold)
        imglog.imwrite("source", frame)
        imglog.imwrite("previous_frame", self.prev_frame)

        mask_pixels = self.mask

        cframe = crop(frame, self.region)
        cprev = crop(self.prev_frame, self.region)

        sqd = numpy.subtract(cframe, cprev, dtype=numpy.int32)  # pylint:disable=assignment-from-no-return
        sqd = (sqd[:, :, 0] ** 2 +
               sqd[:, :, 1] ** 2 +
               sqd[:, :, 2] ** 2)

        if imglog.enabled:
            normalised = numpy.sqrt(sqd / 3)
            if mask_pixels is None:
                imglog.imwrite("sqd", normalised)
            else:
                imglog.imwrite("sqd",
                               normalised.astype(numpy.uint8) & mask_pixels)

        d = sqd >= (self.threshold ** 2) * 3
        if mask_pixels is not None:
            d = numpy.logical_and(d, mask_pixels)  # pylint:disable=assignment-from-no-return
        d = d.astype(numpy.uint8)
        if imglog.enabled:
            imglog.imwrite("thresholded", d * 255)

        if self.kernel is not None:
            d = cv2.morphologyEx(d, cv2.MORPH_OPEN, self.kernel)
            if imglog.enabled:
                imglog.imwrite("eroded", d * 255)

        out_region = pixel_bounding_box(d)
        if out_region:
            # Undo crop:
            out_region = out_region.translate(self.region)

        motion = bool(out_region and (
            self.min_size is None or
            (out_region.width >= self.min_size[0] and
             out_region.height >= self.min_size[1])))

        if motion:
            # Only update the comparison frame if it's different to the previous
            # one.  This makes `detect_motion` more sensitive to slow motion
            # because the differences between frames 1 and 2 might be small and
            # the differences between frames 2 and 3 might be small but we'd see
            # the difference by looking between 1 and 3.
            self.prev_frame = frame

        result = MotionResult(getattr(frame, "time", None), motion,
                              out_region, frame)
        ddebug(str(result))
        imglog.html(MOTION_HTML, result=result)
        return result


MOTION_HTML = u"""\
    <h4>
      MotionDiff:
      {{ "Found" if result.motion else "Didn't find" }} differences
    </h4>

    {{ annotated_image(result) }}

    <h5>Result:</h5>
    <pre><code>{{ result | escape }}</code></pre>

    {% if result.region and not result.motion %}
    <p>Found motion <code>{{ result.region | escape }}</code> smaller than
    min_size <code>{{ min_size | escape }}</code>.</p>
    {% endif %}

    <h5>Previous frame:</h5>
    <img src="previous_frame.png" />

    <h5>Differences:</h5>
    <img src="sqd.png" />

    <h5>Differences above threshold ({{threshold}}):</h5>
    <img src="thresholded.png" />

    <h5>After opening:</h5>
    <img src="eroded.png" />
"""


class GrayscaleDiff(FrameDiffer):
    """Compares 2 frames by converting them to grayscale and calculating their
    absolute difference.

    This was the default `wait_for_motion` diffing algorithm before v33.
    """

    def __init__(self, initial_frame, region=Region.ALL, mask=None,
                 min_size=None, threshold=None, erode=True):
        super(GrayscaleDiff, self).__init__(initial_frame, region, mask,
                                            min_size)

        if threshold is None:
            threshold = get_config('motion', 'threshold', type_=float)
        self.threshold = threshold

        if isinstance(erode, numpy.ndarray):  # For power users
            kernel = erode
        elif erode:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        else:
            kernel = None
        self.kernel = kernel

        self.prev_frame_gray = self.gray(initial_frame)

    def gray(self, frame):
        return cv2.cvtColor(crop(frame, self.region), cv2.COLOR_BGR2GRAY)

    def diff(self, frame):
        frame_gray = self.gray(frame)

        imglog = ImageLogger("GrayscaleDiff", region=self.region,
                             min_size=self.min_size,
                             threshold=self.threshold)
        imglog.imwrite("source", frame)
        imglog.imwrite("gray", frame_gray)
        imglog.imwrite("previous_frame_gray", self.prev_frame_gray)

        absdiff = cv2.absdiff(self.prev_frame_gray, frame_gray)
        imglog.imwrite("absdiff", absdiff)

        if self.mask is not None:
            absdiff = cv2.bitwise_and(absdiff, self.mask)
            imglog.imwrite("mask", self.mask)
            imglog.imwrite("absdiff_masked", absdiff)

        _, thresholded = cv2.threshold(
            absdiff, int((1 - self.threshold) * 255), 255,
            cv2.THRESH_BINARY)
        imglog.imwrite("absdiff_threshold", thresholded)
        if self.kernel is not None:
            thresholded = cv2.morphologyEx(
                thresholded, cv2.MORPH_OPEN,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
            imglog.imwrite("absdiff_threshold_erode", thresholded)

        out_region = pixel_bounding_box(thresholded)
        if out_region:
            # Undo crop:
            out_region = out_region.translate(self.region)

        motion = bool(out_region and (
            self.min_size is None or
            (out_region.width >= self.min_size[0] and
             out_region.height >= self.min_size[1])))

        if motion:
            # Only update the comparison frame if it's different to the previous
            # one.  This makes `detect_motion` more sensitive to slow motion
            # because the differences between frames 1 and 2 might be small and
            # the differences between frames 2 and 3 might be small but we'd see
            # the difference by looking between 1 and 3.
            self.prev_frame = frame
            self.prev_frame_gray = frame_gray

        result = MotionResult(getattr(frame, "time", None), motion,
                              out_region, frame)
        ddebug(str(result))
        imglog.html(GRAYSCALEDIFF_HTML, result=result)
        return result


GRAYSCALEDIFF_HTML = u"""\
    <h4>
      GrayscaleDiff:
      {{ "Found" if result.motion else "Didn't find" }} differences
    </h4>

    {{ annotated_image(result) }}

    <h5>Result:</h5>
    <pre><code>{{ result | escape }}</code></pre>

    {% if result.region and not result.motion %}
    <p>Found motion <code>{{ result.region | escape }}</code> smaller than
    min_size <code>{{ min_size | escape }}</code>.</p>
    {% endif %}

    <h5>ROI Gray:</h5>
    <img src="gray.png" />

    <h5>Previous frame ROI Gray:</h5>
    <img src="previous_frame_gray.png" />

    <h5>Absolute difference:</h5>
    <img src="absdiff.png" />

    {% if "mask" in images %}
    <h5>Mask:</h5>
    <img src="mask.png" />
    <h5>Absolute difference – masked:</h5>
    <img src="absdiff_masked.png" />
    {% endif %}

    <h5>Thresholding (threshold={{threshold}}):</h5>
    <img src="absdiff_threshold.png" />

    <h5>Eroded:</h5>
    <img src="absdiff_threshold_erode.png" />
"""
