# Look at animal detection as well:
# https://towardsdatascience.com/detecting-animals-in-the-backyard-practical-application-of-deep-learning-c030d3263ba8
# Basic algo from: https://www.pyimagesearch.com/2015/11/09/pedestrian-detection-opencv/

# This script uses:
# * Background subtraction to separate foreground from background (fast)
# * Object tracking to track a previously identified object (fast)
# * People detection to identify new objects of interest when we have untracked foreground data (slow)
# * Face detection to identify the best frames in a sequence to take a screenshot (slow)
#
# We will always do background subtraction which is fairly fast and gives 
# a really good idea of the bounding boxes where we likely will see objects
# of interest. These bounding boxes are then also used in other steps to help determine
# the likelihood of any given detection.
#
# When we see something in the foreground then we will try and
# detect/track people/faces in it. The detection is slow, so 
# we use the detection at first to identify a person, and then we
# use tracking on that blob afterwards to follow them through the
# following frames more cheaply without having to do more detection
#
# Finally we use face detection to get an idea of the "quality" of the
# image for taking a still when we see faces.

PRODUCTION = False
DEBUG_INPUT = False
DEBUG_TRACKER_REPLAY = True
DEBUG_TRACKER_DECISIONS = True

import numpy as np
import cv2
import time
import copy
import imutils.object_detection
import glob
import os
import re
import datetime
import os.path
import sys
import json
import argparse
import logging
logger = logging.getLogger(__name__)


class Tracker(object):
    def __init__(self, id, tracker):
        self.id = id
        self.tracker = tracker
        self.failed_start_time = None
        
        self.start_time = None
        self.end_time = None
        
        self.last_bbox = None
        self.best_match = None
        self.best_match_bbox = None
        self.best_match_debug = None
        
        self.best_match_face_area = None
        self.best_match_large_bbox = None
        self.best_match_small_bbox = None

        self.best_match_frame_count = None
        self.face_weight = None
        
        self.tracked_video = None
        self.small_tracked_video = None


class VideoTracking:
    def __init__(self, channel, input_dir, output_dir):
        self.monitor_completed_files = []

        self.channel = '{:02d}'.format(int(channel))
        self.input_dir = input_dir
        self.output_dir = output_dir
        try: os.makedirs(self.output_dir)
        except FileExistsError: pass
    
        self.FRAME_RATE_DIVISOR = 3
        self.SCALE = 3
        self.trackers = []
        self.trackers_seen = 0
        self.background_remover = None

        # Lets also maintain an image of what we think are false positive foreground objects
        # Usually these will be things like trees moving in the wind, with this we can
        # automatically construct our own movement mask map.
        self.noisy_spots, self.noisy_spots_count = self.LoadProbability('noisy')
        self.tracked_spots, self.tracked_spots_count = self.LoadProbability('tracked')
        
        self.frame_count = 0

        self.people_detector = cv2.HOGDescriptor()
        self.people_detector.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        # Downloaded manually from: https://github.com/opencv/opencv/tree/master/data/haarcascades
        self.face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    def GetFileDetails(self, input_file):
        # Two variants of the file name:
        # DVR_02_20210114195959_20210114210001.mp4'
        # 02_20210114195959.mp4'

        # Expect the date in the file name. Take the first one that matches
        start_time_re = re.compile(r'^[^0-9]*(?P<channel>\d{2})_(?P<datetime>\d{14}).*\.[a-zA-Z0-9]*$')

        basename = os.path.basename(input_file)
        m = start_time_re.match(basename)
        if m is None:
            raise Exception('Not a valid DVR recording file name format: ' + str(basename))
            
        channel = m.group('channel')
        start_time = datetime.datetime.strptime(m.group('datetime'), '%Y%m%d%H%M%S')

        return (channel, start_time)

    def SaveBackground(self, background_remover, file_name):
        #background_remover.save(self.DebugFile(file_name + '.debug.txt'))
        params = {}
        
        params['BackgroundRatio'] = background_remover.getBackgroundRatio()
        params['ComplexityReductionThreshold'] = background_remover.getComplexityReductionThreshold()
        params['DetectShadows'] = background_remover.getDetectShadows()
        params['History'] = background_remover.getHistory()
        params['NMixtures'] = background_remover.getNMixtures()
        params['ShadowThreshold'] = background_remover.getShadowThreshold()
        #params['ShadowValue'] = background_remover.getShadowValue()
        params['VarInit'] = background_remover.getVarInit()
        params['VarMax'] = background_remover.getVarMax()
        params['VarMin'] = background_remover.getVarMin()
        params['VarThreshold'] = background_remover.getVarThreshold()
        params['VarThresholdGen'] = background_remover.getVarThresholdGen()
        with open(file_name + '.json', 'w') as file:
            json.dump(params, file, sort_keys=True, indent=4)
        
        im = background_remover.getBackgroundImage()
        cv2.imwrite(file_name + '.png', im)

    def LoadBackground(self, file_name, raise_on_error=False):
        background_remover = cv2.createBackgroundSubtractorMOG2(history = 200, varThreshold = 16, detectShadows = False)
        background_remover.setNMixtures(5)
        background_remover.setBackgroundRatio(0.69999999999999996)
        
        # Assuming this is variance with is standard division squared. I think noiseSigma is == standard deviation
        # Running the cv2.bgsegm.createBackgroundSubtractorMOG() I saw a noiseSigma == 15
        background_remover.setVarInit(15 * 15)

        try:
            with open(file_name + '.json', 'r') as file:
                params = json.load(file)
        except FileNotFoundError:
            if raise_on_error: raise
            return background_remover

        background_remover.setBackgroundRatio(params['BackgroundRatio'])
        background_remover.setComplexityReductionThreshold(params['ComplexityReductionThreshold'])
        background_remover.setDetectShadows(params['DetectShadows'])
        background_remover.setHistory(params['History'])
        background_remover.setNMixtures(params['NMixtures'])
        background_remover.setShadowThreshold(params['ShadowThreshold'])
        #background_remover.setShadowValue(params['ShadowValue'])
        background_remover.setVarInit(params['VarInit'])
        background_remover.setVarMax(params['VarMax'])
        background_remover.setVarMin(params['VarMin'])
        background_remover.setVarThreshold(params['VarThreshold'])
        background_remover.setVarThresholdGen(params['VarThresholdGen'])

        im = cv2.imread(file_name + '.png', cv2.IMREAD_UNCHANGED)
        background_remover.apply(im, learningRate=1)

        return background_remover

    def DrawBoxes(self, frame, boxes, colour, width, scale=1.0):
        for (x, y, w, h) in boxes:
            cv2.rectangle(frame, (int(x * scale), int(y * scale)), (int((x + w) * scale), int((y + h) * scale)), colour, width)

    def NonMaxSuppression(self, boxes):
        dual_points_array = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])
        merged_dual_points_array = imutils.object_detection.non_max_suppression(dual_points_array, probs=None, overlapThresh=0.65)
        return np.array([[x1, y1, x2 - x1, y2 - y1] for (x1, y1, x2, y2) in merged_dual_points_array])

    def Area(self, box): 
        return box[2] * box[3]
        
    def Union(self, a,b):
      x = min(a[0], b[0])
      y = min(a[1], b[1])
      w = max(a[0]+a[2], b[0]+b[2]) - x
      h = max(a[1]+a[3], b[1]+b[3]) - y
      return (x, y, w, h)

    def UnionAll(self, boxes):
        u = boxes[0]
        for b in boxes[1:]: u = self.Union(u, b)
        return u

    def Intersection(self, a,b):
      x = max(a[0], b[0])
      y = max(a[1], b[1])
      w = min(a[0]+a[2], b[0]+b[2]) - x
      h = min(a[1]+a[3], b[1]+b[3]) - y
      if w<0 or h<0: return (x,y,0,0)
      return (x, y, w, h)

    def ExpandBox(self, box, shape, multiplier):
        
        x_diff = int(((box[2] * multiplier) - box[2]) / 2)
        y_diff = int(((box[3] * multiplier) - box[3]) / 2)
        
        x1 = box[0] - x_diff
        y1 = box[1] - y_diff
        x2 = box[0] + box[2] + x_diff
        y2 = box[1] + box[3] + y_diff

        # Now limit to within the specified boundary of shape
        x1 = min(shape[1], max(0, x1))
        x2 = min(shape[1], max(0, x2))

        y1 = min(shape[0], max(0, y1))
        y2 = min(shape[0], max(0, y2))

        return (x1,y1,x2-x1,y2-y1)
        
    def IoU(self, a, b):
        intersection = self.Intersection(a, b)
        intersection_area = self.Area(intersection)

        a_area = self.Area(a)
        b_area = self.Area(b)
        union_area = a_area + b_area - intersection_area
        
        iou = intersection_area / float(union_area)
        return iou

    def FindMatchingObjectIndexes(self, tracked_bbox, foreground_bboxes, required_overlap=0.7):
        matching = []
        for i in range(0, len(foreground_bboxes)):
            foreground_bbox = foreground_bboxes[i]
            
            intersection = self.Intersection(tracked_bbox, foreground_bbox)
            area_i = self.Area(intersection)

            area_a = self.Area(tracked_bbox)
            area_b = self.Area(foreground_bbox)

            perc = area_i / min(area_a, area_b)
            
            if perc > required_overlap: 
                matching.append(i)

        return matching

    def ExtractForeground(self, small_gray_frame):
        small_gray_frame_blurred = cv2.GaussianBlur(small_gray_frame, (19, 19), cv2.BORDER_DEFAULT)
        small_gray_foreground_frame = self.background_remover.apply(small_gray_frame_blurred)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
        small_gray_foreground_frame = cv2.morphologyEx(small_gray_foreground_frame, cv2.MORPH_DILATE, kernel)
        #small_gray_foreground_frame = cv2.morphologyEx(small_gray_foreground_frame, cv2.MORPH_CLOSE, kernel)
    
        return small_gray_foreground_frame

    def FindForegroundObjects(self, small_gray_foreground_frame):
        # Identify the foreground_bboxes from the foreground images used in many following steps
        foreground_contours, hierarchy = cv2.findContours(small_gray_foreground_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        foreground_contours = [c for c in foreground_contours if cv2.contourArea(c) > 250]
        foreground_bboxes = [cv2.boundingRect(c) for c in foreground_contours]
        #foreground_bboxes = self.NonMaxSuppression(foreground_bboxes)
        return (foreground_contours, foreground_bboxes)

    def DebugFile(self, file_name):
        dirname, basename = os.path.split(file_name)

        try: os.makedirs(os.path.join(dirname, 'debug'))
        except FileExistsError: pass
        
        return os.path.join(dirname, 'debug', basename)

    def OnStartTracker(self, tracker):
        # If debugging is enabled, we will dump state right now
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        
        if DEBUG_TRACKER_REPLAY:
            self.SaveBackground(self.background_remover, self.DebugFile(tracker.file_name + '_background'))
            tracker.tracked_video = cv2.VideoWriter(self.DebugFile(tracker.file_name + '.avi'), fourcc, 12.0, (1920,1080))

        if DEBUG_TRACKER_DECISIONS:
            tracker.small_tracked_video = cv2.VideoWriter(self.DebugFile(tracker.file_name + '.small_colour_frame.avi'), fourcc, 12.0, (int(1920/self.SCALE),int(1080/self.SCALE)))

    def OnTickTracker(self, tracker, colour_frame, small_colour_frame):
        if tracker.tracked_video is not None:
            tracker.tracked_video.write(colour_frame)
        
        if tracker.small_tracked_video is not None:
            tracker.small_tracked_video.write(small_colour_frame)
    
    def OnFinishTracker(self, tracker):
        if tracker.tracked_video is not None:
            tracker.tracked_video.release()
            tracker.tracked_video = None

        if tracker.small_tracked_video is not None:
            tracker.small_tracked_video.release()
            tracker.small_tracked_video = None

        print (str(self.frame_count) + ' : Destroy tracker: ' + str(tracker) + ' Now: ' + str(len(self.trackers)) + ' active self.trackers')

        tracker.file_name = tracker.file_name + '_' + tracker.end_time.strftime('%Y%m%d%H%M%S')
        
        # When we finish tracking an object. Create a still image of the best 
        # frame we think we saw for that object
        if tracker.best_match is not None:
            self.DrawBoxes(tracker.best_match, [tracker.best_match_bbox], (255, 0, 0), 2)
            cv2.imwrite(tracker.file_name + '.png', tracker.best_match)
            cv2.imwrite(self.DebugFile(tracker.file_name + '_debug.png'), tracker.best_match_debug)


    def CategorizeForegroundObjects(self, trackers, foreground_contours, foreground_bboxes):
        tracked_foreground_index_to_tracker_map = {}
        
        for tracker in trackers:
            matching_foreground_bbox_indexes = self.FindMatchingObjectIndexes(tracker.last_bbox, foreground_bboxes)
            for foreground_bbox_index in matching_foreground_bbox_indexes:
                try: tracked_foreground_index_to_tracker_map[foreground_bbox_index].append(tracker)
                except KeyError: tracked_foreground_index_to_tracker_map[foreground_bbox_index] = [tracker]

        tracked_bboxes = [tracker.last_bbox for tracker in trackers]
        
        # If there are any objects left-over that are not being tracked by active trackers
        # then we want to run the people detector to see if there are new
        # people to track in this frame
        tracked_foreground_indexes  = [index                      for index in tracked_foreground_index_to_tracker_map.keys()]
        tracked_foreground_contours = [foreground_contours[index] for index in tracked_foreground_index_to_tracker_map.keys()]
        tracked_foreground_bboxes   = [foreground_bboxes[index]   for index in tracked_foreground_index_to_tracker_map.keys()]
        
        untracked_foreground_indexes  = [index                      for index in range(0, len(foreground_bboxes))   if index not in tracked_foreground_index_to_tracker_map]
        untracked_foreground_contours = [foreground_contours[index] for index in range(0, len(foreground_contours)) if index not in tracked_foreground_index_to_tracker_map]
        untracked_foreground_bboxes   = [foreground_bboxes[index]   for index in range(0, len(foreground_bboxes))   if index not in tracked_foreground_index_to_tracker_map]

        return (
            tracked_bboxes,
            tracked_foreground_indexes, 
            tracked_foreground_contours, 
            tracked_foreground_bboxes,
            
            untracked_foreground_indexes,
            untracked_foreground_contours,
            untracked_foreground_bboxes)


    def UpdateActiveTrackers(self, small_colour_frame, foreground_contours, foreground_bboxes):
        # List of trackers that just stopped as the object "disappeared" from the image so we can
        # delete them
        inactive_trackers = []
        
        for tracker in self.trackers:
            matching_foreground_bboxes = []

            # Update the tracker with the new image
            (success, tracked_bbox) = tracker.tracker.update(small_colour_frame)
            if success:
                tracker.last_bbox = tracked_bbox

                # Find any foreground_bboxes that match this tracked bbox
                matching_foreground_bbox_indexes = self.FindMatchingObjectIndexes(tracked_bbox, foreground_bboxes)
                matching_foreground_bboxes = [foreground_bboxes[foreground_bbox_index] for foreground_bbox_index in matching_foreground_bbox_indexes]
            
            # If we can no longer find the object in this image frame then we will
            # deactivate the tracker
            # 
            # The second part of the conditional is because some trackers will 
            # continue to match forever after the bbox walks off the screen
            # So we will identify this when it doesnt match any foreground changes 
            # and force a failed match
            if not success or len(matching_foreground_bboxes) == 0:
                tracker.end_time = self.current_time

                # Note: Don't stop right away, give it a few seconds
                if tracker.failed_start_time is None:
                    tracker.failed_start_time = self.current_time
                    
                if (tracker.failed_start_time + datetime.timedelta(seconds=3)) <= self.current_time:
                    inactive_trackers.append(tracker)
            else:
                tracker.failed_start_time = None
        
        # Clean up old/unused self.trackers
        for tracker in inactive_trackers:
            self.trackers.remove(tracker)
            self.OnFinishTracker(tracker)


    def CreateTracker(self, colour_frame, small_colour_frame, tracked_bbox):
        tracker = Tracker(self.trackers_seen, cv2.TrackerCSRT_create())
        self.trackers_seen += 1
        
        tracker.tracker.init(small_colour_frame, tracked_bbox)
        tracker.last_bbox = tracked_bbox

        tracker.best_match = copy.copy(colour_frame)
        tracker.best_match_bbox = (int(tracked_bbox[0] * self.SCALE), int(tracked_bbox[1] * self.SCALE), int(tracked_bbox[2] * self.SCALE), int(tracked_bbox[3] * self.SCALE))
        tracker.best_match_debug = small_colour_frame
        tracker.best_match_frame_count = self.frame_count

        tracker.start_time = self.current_time
        # + '_tracker' + str(tracker.id)
        tracker.file_name = os.path.join(self.output_dir, 'channel_' + str(self.channel) + '_' + tracker.start_time.strftime('%Y%m%d%H%M%S'))
        print (str(self.frame_count) + ' : Create tracker: ' + str(tracker) + ' box: ' + str(tracked_bbox) + ' Now: ' + str(len(self.trackers)) + ' active self.trackers')
        
        return tracker

    def ReInitTracker(self, tracker, colour_frame, small_colour_frame, tracked_bbox):
        tracker.tracker = cv2.TrackerCSRT_create()
        tracker.tracker.init(small_colour_frame, tracked_bbox)
        tracker.last_bbox = tracked_bbox
    
    def CheckObjectToTrack(self,
        colour_frame,
        small_colour_frame,
        foreground_bboxes, 
        person_bbox):

        # Find any foreground objects associated with this object of interest and make sure we didn't find
        # an object of interest in the "background". If we did then we will just ignore it as a false positive
        matching_foreground_bbox_indexes = self.FindMatchingObjectIndexes(person_bbox, foreground_bboxes)
        if len(matching_foreground_bbox_indexes) == 0:
            print (str(self.frame_count) + ' : No foreground objects match the found person, likely a false positive found in the background image')
            return None

        
        # First we need to "adjust" the person_bbox. The person detector often 
        # reports larger images than expected. We can trim it down based on the
        # matching_foreground_bboxes to include only area that we saw foreground
        # changes in. Otherwise we would just use "tracked_bbox = tuple(person_bbox)"
        matching_foreground_bboxes = [foreground_bboxes[i] for i in matching_foreground_bbox_indexes]
        merged_foreground_box = self.UnionAll(matching_foreground_bboxes)
        merged_foreground_box = self.ExpandBox(merged_foreground_box, small_colour_frame.shape, 1.2)
        tracked_bbox = self.Intersection(person_bbox, merged_foreground_box)

        # Now we may already have a tracker for this person, lets check and if so we want to
        # skip them and just keep using the existing tracker.
        existing_tracked_bboxes = [tracker.last_bbox for tracker in self.trackers]
        person_matching_tracker_bbox_indexes = self.FindMatchingObjectIndexes(person_bbox, existing_tracked_bboxes, required_overlap=0.4)
        if len(person_matching_tracker_bbox_indexes) > 0:
            # We should find the "best" tracker match. Probably the tracker with the most
            # overlap OR assuming equal overlap with the smallest area?
            # @todo for now just pick the first match
            tracker = self.trackers[person_matching_tracker_bbox_indexes[0]]
            
            # We may want to re-size/re-init the tracker though
            print (str(self.frame_count) + ' : skipping object: ' + str(person_bbox) + ' as we already have at least one tracker covering its area: ' + str(person_matching_tracker_bbox_indexes))
            self.ReInitTracker(tracker, colour_frame, small_colour_frame, tracked_bbox)
            return None


        # Lets create the tracker and update the state
        tracker = self.CreateTracker(colour_frame, small_colour_frame, tracked_bbox)
        return tracker
    
    def ActivateNewTrackers(self,
            colour_frame,
            small_colour_frame,
            small_gray_frame,
            foreground_contours, 
            foreground_bboxes):
    
        new_objects_tracked = []
        new_tracked_bboxes = []
        
        # Lets add any trackers for people detected in the scene
        #people_bboxes, people_weights = self.people_detector.detectMultiScale(small_gray_frame, winStride=(4,4), padding=(8, 8), self.SCALE=1.05)
        people_bboxes, people_weights = self.people_detector.detectMultiScale(small_gray_frame)
        #people_bboxes = self.NonMaxSuppression(people_bboxes)
        if len(people_bboxes) > 0:
            print (str(self.frame_count) + ' : found people: ' + str(people_bboxes) + ' weights: '+ str(people_weights))
            for person_bbox in people_bboxes:
                print (str(self.frame_count) + ' : found person: ' + str(person_bbox))
                tracker = self.CheckObjectToTrack(
                    colour_frame,
                    small_colour_frame,
                    foreground_bboxes, 
                    person_bbox)
                
                if tracker is not None:
                    self.trackers.append(tracker)
                    new_objects_tracked.append(tracker)
                    new_tracked_bboxes.append(tracker.last_bbox)
                    self.OnStartTracker(tracker)
        return (new_objects_tracked, list(people_bboxes), new_tracked_bboxes)


    def GetForegroundObjects(self, tracker, foreground_contours, foreground_bboxes):
        tracked_foreground_indexes  = []
        tracked_foreground_contours = []
        tracked_foreground_bboxes   = []

        matching_foreground_bbox_indexes = self.FindMatchingObjectIndexes(tracker.last_bbox, foreground_bboxes)
        for foreground_bbox_index in matching_foreground_bbox_indexes:
            tracked_foreground_indexes.append(foreground_bbox_index)
            tracked_foreground_contours.append(foreground_contours[foreground_bbox_index])
            tracked_foreground_bboxes.append(foreground_bboxes[foreground_bbox_index])
        return (
            tracked_foreground_indexes, 
            tracked_foreground_contours, 
            tracked_foreground_bboxes)

    def CheckForFacesInTrackedObjects(self, colour_frame, small_colour_frame, foreground_contours, foreground_bboxes):
        faces_bboxes = []
        for tracker in self.trackers:
            person_bbox = tracker.last_bbox
            person_upper_bbox = (person_bbox[0], person_bbox[1], int(person_bbox[2]/2.5), int(person_bbox[3]/2.5))
            
            # We will crop the tracked bounding box to search for a face in it instead of the entire image
            # this is faster.
            x = int(person_upper_bbox[0] * self.SCALE)
            y = int(person_upper_bbox[1] * self.SCALE)
            w = int(person_upper_bbox[2] * self.SCALE)
            h = int(person_upper_bbox[3] * self.SCALE)
            crop_img = colour_frame[y:y+h, x:x+w]
            faces_bboxes_big, face_weights = self.face_detector.detectMultiScale2(crop_img, scaleFactor=1.05, minNeighbors=5, minSize=(40,40), maxSize=(300,300))
            
            # Convert the higher resolution but cropped bboxes to fit the 
            # small_colour_frame/small_gray_frame images to be consistent 
            # with other bboxes
            faces_bboxes_small = []
            for fb in faces_bboxes_big:
                faces_bboxes_small.append((int((fb[0] + x) / self.SCALE), int((fb[1] + y) / self.SCALE), int((fb[2]) / self.SCALE), int((fb[3]) / self.SCALE)))
            
            # Filter out faces that don't match our person upper-body bbox
            faces_bboxes_small_indexes = self.FindMatchingObjectIndexes(person_upper_bbox, faces_bboxes_small)
            faces_bboxes_small = [faces_bboxes_small[i] for i in faces_bboxes_small_indexes]

            # Filter out faces that don't intersect with the actual foreground object contours for this tracker
            # I.e. They are inside the bbox, but it often has a lot of background image in it and we
            # dont care about matches that are in the background part of the image.
            faces_bboxes_small_tmp = []
            for face_bbox in faces_bboxes_small:
                x1 = face_bbox[0]
                y1 = face_bbox[1]
                x2 = x1 + face_bbox[2]
                y2 = y1 + face_bbox[3]
                face_contour = np.array([[[x1,y1]], [[x2,y1]], [[x2,y2]], [[x1,y2]]], dtype=np.int32)

                face_mask = np.zeros(small_colour_frame.shape, np.uint8)
                cv2.drawContours(face_mask, [face_contour], 0, 1, cv2.FILLED)
                
                matched = False
                (
                    tracked_foreground_indexes, 
                    tracked_foreground_contours, 
                    tracked_foreground_bboxes
                ) = self.GetForegroundObjects(tracker, foreground_contours, foreground_bboxes)
                
                for tracked_foreground_contour in tracked_foreground_contours:
                    object_mask = np.zeros(small_colour_frame.shape, np.uint8)
                    cv2.drawContours(object_mask, [tracked_foreground_contour], 0, 2, cv2.FILLED)
                
                    combined = face_mask + object_mask
                    intersect_area = np.sum(np.greater(combined, 2))
                    if intersect_area > 0:
                        matched = True
                        break

                if matched: faces_bboxes_small_tmp.append(face_bbox)
            faces_bboxes_small = faces_bboxes_small_tmp
            
            # After filtering good face matches, we want to now find the best face match out of what 
            # we have seen
            this_frame_best_face_bbox = None
            this_frame_best_face_bbox_area = None

            for face_bbox in faces_bboxes_small:

                # Largest area of intersection with its bbox will give a good match criteria
                # As will hopefully give a bigger facial shot
                area = self.Area(self.Intersection(person_upper_bbox, face_bbox))

                if this_frame_best_face_bbox is None or area > this_frame_best_face_bbox_area:
                    this_frame_best_face_bbox = face_bbox
                    this_frame_best_face_bbox_area = area
                
                if tracker.best_match_face_area is None or area > tracker.best_match_face_area:
                    tracker.best_match_face_area = area

                    tracker.best_match_large_bbox = (int(person_bbox[0] * self.SCALE), int(person_bbox[1] * self.SCALE), int(person_bbox[2] * self.SCALE), int(person_bbox[3] * self.SCALE))
                    tracker.best_match_small_bbox = (int(person_bbox[0]), int(person_bbox[1]), int(person_bbox[2]), int(person_bbox[3]))

                    tracker.best_match_bbox = tracker.best_match_large_bbox
                    tracker.best_match = copy.copy(colour_frame)
                    
                    # Note: Later will be updated to include the debug outlines etc
                    tracker.best_match_debug = small_colour_frame
                    
                    tracker.best_match_frame_count = self.frame_count

            if this_frame_best_face_bbox is not None:
                faces_bboxes.append(this_frame_best_face_bbox)
                
        return faces_bboxes

    def TrackObjects(self,
        colour_frame,
        small_colour_frame,
        small_gray_frame,
        small_gray_foreground_frame,
        foreground_contours, 
        foreground_bboxes,
        
        # @todo This is an in/out param and is a bad style to use this way
        objects_tracked):

        tracked_bboxes = []
        people_bboxes = []

        # Skip this frame if too much has changed
        area_foreground = np.sum(np.greater(small_gray_foreground_frame, 200))
        total_area = small_gray_foreground_frame.shape[0] * small_gray_foreground_frame.shape[1]
        percentage_changed = 100.0 * area_foreground / total_area
        if len(self.trackers) == 0 and percentage_changed > 40:
            print ('Too much of the image has changed, likely transition from day to night camera so skipping this frame. area canged: ' + str(area_foreground) + ' total area: ' + str(total_area) + ' percentage_changed: ' + str(percentage_changed))
            return (tracked_bboxes, people_bboxes)
        
        # Tick all the active trackers which will update the existing self.trackers list based on the current frame
        self.UpdateActiveTrackers(
            small_colour_frame, 
            foreground_contours, 
            foreground_bboxes)

        # Handle special case where we start this file with some trackers enabled
        # that were active at the end of the last file and are considered still
        # active at the start of this file.
        #
        # objects_tracked will be empty as we haven't been able to create new ones yet
        # as that happens after this code and self.trackers will be non-empty.
        #
        # NOTE: Is deliberately after the UpdateActiveTrackers() handling to be sure we don't
        # add it if it existed in previous video but not in this one
        if len(self.trackers) > 0 and len(objects_tracked) == 0:
            objects_tracked += [t for t in self.trackers if t.failed_start_time is None]
        
        # Give the updated list of trackers from last frame we will categorize the foreground objects
        # into tracked and untracked objects
        (
            tracked_bboxes,
            tracked_foreground_indexes, 
            tracked_foreground_contours, 
            tracked_foreground_bboxes, 
            untracked_foreground_indexes, 
            untracked_foreground_contours, 
            untracked_foreground_bboxes
        ) = self.CategorizeForegroundObjects(self.trackers, foreground_contours, foreground_bboxes)
        assert(len(untracked_foreground_bboxes) == (len(foreground_bboxes) - len(tracked_foreground_bboxes)))
        
        # Any untracked objects are candidates for new potential new people to be detected and tracked
        if len(untracked_foreground_bboxes) > 0:
            ot, pb, tb = self.ActivateNewTrackers(
                colour_frame,
                small_colour_frame,
                small_gray_frame,
                foreground_contours, 
                foreground_bboxes)
            objects_tracked += ot
            people_bboxes += pb
            tracked_bboxes += tb

        # @todo We no longer properly maintain tracked_bboxes as we can modify an existing tracker. Maybe re-init it instead?
        return (tracked_bboxes, people_bboxes)

    def Load(self, file_name, raise_on_error=False):
        self.background_remover = self.LoadBackground(file_name + '_background', raise_on_error=False)
    
    def Save(self, file_name):
        self.SaveBackground(self.background_remover, file_name + '_background')
    
    # @todo Pass in file name and dont use DebugFile in here
    def SaveProbability(self, spots, spots_count, name):
        spots_copy = spots.copy()
        spots_max = spots_copy.max()
        spots_copy *= 255.0 / spots_max
        
        file_name_base = self.DebugFile(os.path.join(self.output_dir, 'channel_' + str(self.channel) + '.' + name))
        cv2.imwrite(file_name_base + '.png', spots_copy)
        
        with open(file_name_base + '.json', 'w') as file:
            print('''{
count:'''+str(spots_count)+''',
max:'''+str(spots_max)+''',
}''', file=file)


    def LoadProbability(self, name):
        # @todo Add file loading support
        spots = np.zeros(( int(1080/self.SCALE), int(1920/self.SCALE) ), dtype=np.float32)
        spots_count = 0
        return (spots, spots_count)
    
    def UpdateTrackingPositionProbabilities(self, trackers, foreground_contours, foreground_bboxes):
        (
            tracked_bboxes,
            tracked_foreground_indexes, 
            tracked_foreground_contours, 
            tracked_foreground_bboxes, 
            untracked_foreground_indexes, 
            untracked_foreground_contours, 
            untracked_foreground_bboxes
        ) = self.CategorizeForegroundObjects(trackers, foreground_contours, foreground_bboxes)
    
        this_frames_noisy_spots = np.zeros(self.noisy_spots.shape, np.uint8)
        cv2.drawContours(this_frames_noisy_spots, untracked_foreground_contours, 0, 1, cv2.FILLED)
        if this_frames_noisy_spots.max() > 0:
            self.noisy_spots = self.noisy_spots + this_frames_noisy_spots
            self.noisy_spots_count += 1
        
        this_frames_tracked_spots = np.zeros(self.noisy_spots.shape, np.uint8)
        cv2.drawContours(this_frames_tracked_spots, tracked_foreground_contours, 0, 1, cv2.FILLED)
        if this_frames_tracked_spots.max() > 0:
            self.tracked_spots = self.tracked_spots + this_frames_tracked_spots
            self.tracked_spots_count += 1
    
    def GetContourIntersection(self, small_colour_frame, contours, bboxes, colour):
        bbox_mask = np.zeros(small_colour_frame.shape, np.uint8)
        for bbox in bboxes:
            x1 = bbox[0]
            y1 = bbox[1]
            x2 = x1 + bbox[2]
            y2 = y1 + bbox[3]
            bbox_contour = np.array([[[x1,y1]], [[x2,y1]], [[x2,y2]], [[x1,y2]]], dtype=np.int32)
            cv2.drawContours(bbox_mask, [bbox_contour], -1, (1,1,1), cv2.FILLED)

        contour_mask = np.zeros(small_colour_frame.shape, np.uint8)
        cv2.drawContours(contour_mask, contours, -1, (2,2,2), cv2.FILLED)
    
        #import code
        #code.interact(local=dict(globals(), **locals()))
        combined = bbox_mask + contour_mask
        
        return np.where(np.greater(combined, 2), colour, (0,0,0)).astype(np.uint8)

    def OverlayMask(self, background, foreground, alpha=0.3):
        # From: https://learnopencv.com/alpha-blending-using-opencv-cpp-python/
        
        fg = foreground * alpha
        bg = background * (1.0 - alpha)
        blended = bg + fg
        blended = blended.astype(np.uint8)
        
        # Wont return a copy instead we will modify the background so it works for debug of
        # trackers where drawing is done at the end
        background[:] = blended[:]
        
        #import code
        #code.interact(local=dict(globals(), **locals()))
    
    def ProcessVideo(self, input_file, initial_state=None):
        print ('Processing input file: ' + str(input_file))
        cap = cv2.VideoCapture(input_file)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        channel, start_time = self.GetFileDetails(input_file)
        assert(channel == self.channel)
        print ('Start time: ' + str(start_time))
        print ('Channel: ' + str(self.channel))
        print ('Video FPS: ' + str(fps))

        raise_on_error = False
        if initial_state is None:
            initial_state = os.path.join(self.output_dir, 'channel_' + str(self.channel))
            raise_on_error = True
        self.Load(initial_state, raise_on_error=raise_on_error)

        out_colour_frame = None
        out_small_colour_frame = None
        out_small_gray_frame = None
        out_small_gray_foreground_frame = None
        if DEBUG_INPUT:
            output_base = os.path.join(self.output_dir, os.path.basename(input_file))
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            #out_colour_frame = cv2.VideoWriter(self.DebugFile(output_base + '.output.avi'), fourcc, 12.0, (1920,1080))
            out_small_colour_frame = cv2.VideoWriter(self.DebugFile(output_base + '.small_colour_frame.avi'), fourcc, 12.0, (int(1920/self.SCALE),int(1080/self.SCALE)))
            #out_small_gray_frame = cv2.VideoWriter(self.DebugFile(output_base + '.small_grey.avi'), fourcc, 12.0, (int(1920/self.SCALE),int(1080/self.SCALE)))
            #out_small_gray_foreground_frame = cv2.VideoWriter(self.DebugFile(output_base + '.small_gray_foreground_frame.avi'), fourcc, 12.0, (int(1920/self.SCALE),int(1080/self.SCALE)))

        self.frame_count = 0
        frames_last_count = 0
        frames_last_time = time.time()
        objects_tracked = []

        while(cap.isOpened()):
            ret, colour_frame = cap.read()
            if not ret:
                break

            self.current_time = start_time + datetime.timedelta(seconds=(self.frame_count / fps))
            self.frame_count += 1
            if (frames_last_time + 1.0) <= time.time():
                fps = (self.frame_count - frames_last_count) / (time.time() - frames_last_time)
                print ('FPS: ' + str(fps) + ' frame: ' + str(self.frame_count) + ' / ' + str(total_frame_count) + ' : ' + str(int(100 * self.frame_count / total_frame_count)) + '%')
                
                frames_last_time += 1.0
                frames_last_count = self.frame_count

            # We can increase the "fps" by skipping a bunch of frames
            # we don't really need to run at the full framerate (12 fps)
            #
            # However once we have a tracker we will process every frame and
            # not skip any
            if len(self.trackers) == 0 and (self.frame_count % self.FRAME_RATE_DIVISOR) != 0:
                continue

            # Different processing segments require different image sizes and depths
            # for accuracy and speed
            small_colour_frame = cv2.resize(colour_frame, (int(1920/self.SCALE), int(1080/self.SCALE)))
            small_gray_frame = cv2.cvtColor(small_colour_frame, cv2.COLOR_RGB2GRAY)

            # Extract the foreground image
            small_gray_foreground_frame = self.ExtractForeground(small_gray_frame)

            # Extract the objects/contours from the foreground
            foreground_contours, foreground_bboxes = self.FindForegroundObjects(small_gray_foreground_frame)

            tracked_bboxes, people_bboxes = self.TrackObjects(
                colour_frame,
                small_colour_frame,
                small_gray_frame,
                small_gray_foreground_frame,
                foreground_contours, 
                foreground_bboxes,
                
                # Note: Is an out parameter
                objects_tracked
                )

            # Finally lets look inside any tracked bboxes (new or previously active) and 
            # see if we can find a face. We will keep a copy of the best facial image we
            # see while we are tracking this object.
            faces_bboxes = self.CheckForFacesInTrackedObjects(colour_frame, small_colour_frame, foreground_contours, foreground_bboxes)

            self.UpdateTrackingPositionProbabilities(self.trackers, foreground_contours, foreground_bboxes)
            
            if DEBUG_TRACKER_DECISIONS or DEBUG_INPUT or PRODUCTION:
                face_mask = self.GetContourIntersection(small_colour_frame, foreground_contours, faces_bboxes, (255, 255, 0))
                person_mask = self.GetContourIntersection(small_colour_frame, foreground_contours, people_bboxes, (255, 0, 0))
                tracker_mask = self.GetContourIntersection(small_colour_frame, foreground_contours, tracked_bboxes, (0, 255, 0))

                # @todo Break into tracking/noisy probable masks
                foreground_mask = self.GetContourIntersection(small_colour_frame, foreground_contours, foreground_bboxes, (0, 0, 255))

                merged = np.zeros(small_colour_frame.shape, np.uint8)
                merged = np.where(np.greater(foreground_mask, 0), foreground_mask, merged).astype(np.uint8)
                merged = np.where(np.greater(tracker_mask, 0), tracker_mask, merged).astype(np.uint8)
                merged = np.where(np.greater(person_mask, 0), person_mask, merged).astype(np.uint8)
                merged = np.where(np.greater(face_mask, 0), face_mask, merged).astype(np.uint8)
                self.OverlayMask(small_colour_frame, merged)
                
                # Draw all debug onto the small_colour_frame
                self.DrawBoxes(small_colour_frame, faces_bboxes, (255, 255, 0), 1)
                self.DrawBoxes(small_colour_frame, people_bboxes, (255, 0, 0), 3)
                self.DrawBoxes(small_colour_frame, tracked_bboxes, (0, 255, 0), 2)
                self.DrawBoxes(small_colour_frame, foreground_bboxes, (0, 0, 255), 1)

            for tracker in self.trackers:
                self.OnTickTracker(tracker, colour_frame, small_colour_frame)

            if not PRODUCTION:
                #cv2.imshow('frame', colour_frame)
                cv2.imshow('frame', small_colour_frame)
                #cv2.imshow('frame', small_gray_frame)
                #cv2.imshow('frame', small_gray_foreground_frame)
                pass

            if out_colour_frame is not None:
                out_colour_frame.write(colour_frame)

            if out_small_colour_frame is not None:
                out_small_colour_frame.write(small_colour_frame)

            if out_small_gray_frame is not None:
                out_small_gray_frame.write(cv2.cvtColor(small_gray_frame, cv2.COLOR_GRAY2RGB))

            if out_small_gray_foreground_frame is not None:
                out_small_gray_foreground_frame.write(cv2.cvtColor(small_gray_foreground_frame, cv2.COLOR_GRAY2RGB))
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.SaveProbability(self.noisy_spots, self.noisy_spots_count, 'noisy')
        self.SaveProbability(self.tracked_spots, self.tracked_spots_count, 'tracked')
        self.Save(initial_state)
        
        cap.release()

        if out_colour_frame is not None:
            out_colour_frame.release()

        if out_small_colour_frame is not None:
            out_small_colour_frame.release()

        if out_small_gray_frame is not None:
            out_small_gray_frame.release()

        if out_small_gray_foreground_frame is not None:
            out_small_gray_foreground_frame.release()

        if not PRODUCTION:
            cv2.destroyAllWindows()
        
        return objects_tracked
        
    def ProcessFiles(self):
        available_files = []
        for f in os.listdir(self.input_dir):
            input_file = os.path.join(self.input_dir, f)
            
            if not os.path.isfile(input_file):
                continue
            
            if input_file in self.monitor_completed_files:
                continue
            
            channel = '00'
            start_time = None
            try: channel, start_time = self.GetFileDetails(input_file)
            except Exception as e: 
                print ('Skipping file: ' + str(input_file) + ', reason: ' + str(e))
                self.monitor_completed_files.append(input_file)
                continue
            
            if int(channel) != int(self.channel):
                print ('Skipping file: ' + str(input_file) + ', as it doesnt belong to our monitored channel: ' + str(self.channel))
                self.monitor_completed_files.append(input_file)
                continue
            
            available_files.append((start_time, input_file))

        sorted_files = sorted(available_files, key=lambda tup: tup[0])
        for start_time, input_file in sorted_files:
            objects_tracked = video_tracking.ProcessVideo(input_file)
            if PRODUCTION :
                try: os.remove(input_file)
                except: print("Error while deleting file : ", input_file)
            print ('Finished processing file: ' + str(input_file) + ' and found ' + str(len(objects_tracked)) + ' unique objects in the video feed')
            self.monitor_completed_files.append(input_file)

def ParseCmdArgs():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--channel', dest='channel', metavar='str', default='1', help= 'The channel to monitor')
    parser.add_argument('--input-dir', dest='input_dir', metavar='FILE', help= 'Input directory to monitor for new video files to process')
    parser.add_argument('--output-dir', dest='output_dir', metavar='FILE', help= 'Output directory to place generated data into')
    
    parser.add_argument('--input-debug', dest='input_debug', metavar='FILE', help= 'File to replay')
    return parser.parse_args()

if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    args = ParseCmdArgs()

    video_tracking = VideoTracking(args.channel, args.input_dir, args.output_dir)
    
    if args.input_debug:
        # Special scenario, we will load a initial state and debug
        PRODUCTION = False
        DEBUG_INPUT = False
        DEBUG_TRACKER_REPLAY = True
        DEBUG_TRACKER_DECISIONS = True

        # channel_01_20210113112609.avi
        # channel_01_20210113112609_background.json
        initial_state = os.path.splitext(args.input_debug)[0] + '_background'
        objects_tracked = video_tracking.ProcessVideo(args.input_debug, initial_state=initial_state)

    else:
        print ('Waiting for mp4 video stream files in folder: ' + str(args.input_dir) + ' on channel: ' + str(args.channel))
        while True:
            video_tracking.ProcessFiles()
            time.sleep(1)
