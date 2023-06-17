class DetectResultsMaster:
    def __init__(self,postions,confidence, class_id, tracker_id):
        self._postions=postions
        self._confidence=confidence 
        self._class_id=class_id 
        self._tracker_id=tracker_id
    
    def getPostions(self):
        return self._postions
    
    def __str__(self):
        return f"""
        ---------------------------------------
        postions {self._postions},
        confidence {self._confidence},
        class_id {self._class_id},
        tracker_id {self._tracker_id}
        ---------------------------------------
        """
    def getConfidence(self):
        return float("{:.2f}".format(self._confidence))
    
    def getClassId(self):
        return self._class_id
    
    def getTrackerId(self):
        return self._tracker_id