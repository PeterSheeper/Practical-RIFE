from scenedetect import detect, AdaptiveDetector


class SceneCutDetector:
    def __init__(self, video_path):
        self.video_path = video_path
        self.scene_cut_frames = []

    def detect_scene_cuts(self, score_scene_value=3.0):
        if self.video_path is None:
            return
        scene_list = detect(self.video_path, AdaptiveDetector(score_scene_value))
        cuts_list = [scene[0] for scene in scene_list[1:]]
        self.scene_cut_frames = [cut.get_frames() for cut in cuts_list]  # Skipping the first cut

        # Print detected scene cuts
        print("Detected Scene Cuts:")
        for cut in cuts_list:
            print(f"Cut {cut.get_timecode()} Frame {cut.get_frames()}")

    def user_correction(self):
        remove_frames = []
        add_frames = []

        repeat_ask = True
        while repeat_ask:
            user_changes = input("Type -000 to remove cut or +000 to add cut (separate by spaces):\n")
            repeat_ask = False
            remove_frames.clear()
            add_frames.clear()
            if len(user_changes) == 0:
                break
            for change in user_changes.split():
                if change.startswith('-'):
                    remove_frames.append(int(change.replace('-', '')))
                elif change.startswith('+'):
                    add_frames.append(int(change.replace('+', '')))
                else:
                    print("Error! Wrong input")
                    repeat_ask = True

        frames = add_frames
        for cut_frame in self.scene_cut_frames:
            if cut_frame not in remove_frames and cut_frame not in frames:
                frames.append(cut_frame)

        frames.sort()
        self.scene_cut_frames = frames

    def get_scene_cut_frames(self):
        return self.scene_cut_frames
