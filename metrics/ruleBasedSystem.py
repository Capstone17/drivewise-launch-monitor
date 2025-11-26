from .metric_output import return_metrics


# Helper function for unsuccessful club detection
def ball_only_message(side_angle):
    if side_angle is None:
        return ("Ignore", "Error: Side angle data unavailable.")
    
    if side_angle < -2.0:
        return ("Pull", "Pull: Both your face and path are left, causing a pull. Try weakening your grip on the club and keeping the face more square.")
    elif side_angle > 2.0:
        return ("Push", "Push: Both your face and path are right, causing a push. Try strengthening your grip on the club.")
    
    return ("Ideal", "Straight: Nice shot!")


# Helper function for unsuccessful ball detection
def club_only_message(swing_path):
    if swing_path is None:
        return ("Ignore", "Error: Swing path data unavailable.")
    
    if swing_path < -3.0:
        return ("Slice", "Fade: Your shot is curving right. If your shot is landing right of the target, try slightly closing your stance.")
    elif swing_path > 3.0:
        return ("Hook", "Draw: Your shot is curving left. If your shot is landing left of the target, try slightly opening your stance.")
    
    return ("Ideal", "Straight: Nice shot!")


# -------------------------------
# Output as a function (for bluetooth)
# -------------------------------
def rule_based_system(club_selection):

    # This line MUST be called inside the function, not globally, or else it messes up the bluetooth
    raw_data = return_metrics()  # face_angle, swing_path, attack_angle, side_angle, face_to_path
    
    # -------------------------------
    # FACT BASE
    # - Note that parameters have been relaxed slightly to allow for the error in our detection
    # - Ideal numbers for face angle are from Trackman: https://www.trackman.com/blog/golf/face-to-path, and GolfWRX: https://www.golfwrx.com/342864/how-to-hit-a-push-draw-and-a-pull-fade/1000/ 
    # - Face Angle: 
    #     Face to Path – The angle difference between FACE ANGLE and CLUB PATH as defined (FACE ANGLE minus CLUB PATH).
    #     A positive Face to Path means the face is pointed to the right of the club path regardless of dexterity.
    #     A negative Face to Path means the face is pointed to the left of the club path regardless of dexterity.
    # - Ideal numbers for club path, attack angle, and side angle (aka start direction) are from Rapsodo: https://rapsodo.ca/blogs/golf/understanding-club-path-and-attack-angle-for-your-golf-launch-monitor?shpxid=cd932132-fcb7-46ff-aff3-dc4ccb5e24f9
    # - Club Path:
    #     Positive (In-to-Out) - The clubhead is moving to the right of the target line at impact (for a right-handed golfer).
    #     Negative (Out-to-In) - The clubhead is moving to the left of the target line at impact (for a right-handed golfer).
    #     Zero (Straight) - The clubhead is moving directly down the target line at impact.
    # - Attack Angle:
    #     Positive Attack Angle: The clubhead is traveling upward (commonly seen with a driver).
    #     Negative Attack Angle: The clubhead is descending (typical for irons and wedges).
    #     Zero Attack Angle: Level with the ground at impact (rare, but possible).
    # - Ideal numbers for face-to-path are from GolfWRX: https://www.golfwrx.com/342864/how-to-hit-a-push-draw-and-a-pull-fade/1000/
    # -------------------------------
    facts = {
        # Error case: club and ball were not detected
        "club_and_ball_detection_error": ( raw_data['face_angle'] is None and 
                                        raw_data['swing_path'] is None and 
                                        raw_data['attack_angle'] is None and
                                        raw_data['side_angle'] is None and
                                        raw_data['face_to_path'] is None ),

        # Error case: club was not detected
        "club_detection_error": (   raw_data['swing_path'] is None and 
                                    raw_data['attack_angle'] is None    ),

        # Error case: ball was not detected
        "ball_detection_error": (raw_data['side_angle'] is None),

        # Face angle facts - with None checks
        "face_extreme_left": raw_data['face_angle'] is not None and raw_data['face_angle'] < -6.0,
        "face_slight_left": raw_data['face_angle'] is not None and -6.0 <= raw_data['face_angle'] < -3.0,
        "face_straight": raw_data['face_angle'] is not None and -3.0 <= raw_data['face_angle'] < 3.0,
        "face_slight_right": raw_data['face_angle'] is not None and 3.0 <= raw_data['face_angle'] < 6.0,
        "face_extreme_right": raw_data['face_angle'] is not None and 6.0 <= raw_data['face_angle'],

        # Path facts - with None checks
        "path_extreme_left": raw_data['swing_path'] is not None and raw_data['swing_path'] < -6.0,
        "path_slight_left": raw_data['swing_path'] is not None and -6.0 <= raw_data['swing_path'] < -2.0,
        "path_straight": raw_data['swing_path'] is not None and -2.0 <= raw_data['swing_path'] < 2.0,
        "path_slight_right": raw_data['swing_path'] is not None and 2.0 <= raw_data['swing_path'] < 6.0,
        "path_extreme_right": raw_data['swing_path'] is not None and 6.0 <= raw_data['swing_path'],

        # Attack angle facts - with None checks
        "attack_extreme_up": raw_data['attack_angle'] is not None and 5.0 < raw_data['attack_angle'],
        "attack_up": raw_data['attack_angle'] is not None and 3.0 < raw_data['attack_angle'] <= 5.0,
        "attack_slight_up": raw_data['attack_angle'] is not None and 1.0 < raw_data['attack_angle'] <= 3.0,
        "attack_neutral": raw_data['attack_angle'] is not None and -2.0 < raw_data['attack_angle'] <= 1.0,
        "attack_1.5to3_down": raw_data['attack_angle'] is not None and -1.5 < raw_data['attack_angle'] <= 3.0,
        "attack_slight_down": raw_data['attack_angle'] is not None and -5.0 < raw_data['attack_angle'] <= -2.0,
        "attack_very_down": raw_data['attack_angle'] is not None and -10.0 < raw_data['attack_angle'] <= -5.0,
        "attack_extreme_down": raw_data['attack_angle'] is not None and raw_data['attack_angle'] <= -10.0,

        # Side angle facts - with None checks
        "side_extreme_left": raw_data['side_angle'] is not None and raw_data['side_angle'] < -6.0,
        "side_slight_left": raw_data['side_angle'] is not None and -6.0 <= raw_data['side_angle'] < -2.0,
        "side_straight": raw_data['side_angle'] is not None and -2.0 <= raw_data['side_angle'] < 2.0,
        "side_slight_right": raw_data['side_angle'] is not None and 2.0 <= raw_data['side_angle'] < 6.0,
        "side_extreme_right": raw_data['side_angle'] is not None and 6.0 <= raw_data['side_angle'],

        # Face-to-path facts - with None checks
        "face_to_path_extreme_left": raw_data['face_to_path'] is not None and raw_data['face_to_path'] < -6.0,
        "face_to_path_slight_left": raw_data['face_to_path'] is not None and -6.0 <= raw_data['face_to_path'] < -2.0,
        "face_to_path_straight": raw_data['face_to_path'] is not None and -2.0 <= raw_data['face_to_path'] < 2.0,
        "face_to_path_slight_right": raw_data['face_to_path'] is not None and 2.0 <= raw_data['face_to_path'] < 6.0,
        "face_to_path_extreme_right": raw_data['face_to_path'] is not None and 6.0 <= raw_data['face_to_path']
    }


    # -------------------------------
    # Rules
    # - The shot shaping diagnosis was taken from this Reddit post: https://www.reddit.com/r/GolfSwing/comments/1546tus/i_think_everyone_should_save_this_diagram_to_help/ 
    # -------------------------------
    rules = [
        # -------------------------------
        # Error Cases 
        # - Severity is higher than any other result (7, 6)
        # - Club detection error, fall back to "Ball Only Detected"
        # - Ball detection error, fall back to "Club Only Detected"
        # -------------------------------
        {
            "group": "Ignore",
            "name": "Club and Ball Detection Error",
            "category": "all",
            "severity": 7,
            "condition": lambda f: (f["club_and_ball_detection_error"]),
            "action": lambda: "Error: Shot was not detected. Make sure your device is calibrated and placed on a flat surface, and aim for minimal movement in the background."
        },
        {
            "group": None,  # Will be determined dynamically
            "name": "Ball Only Detected",
            "category": "all",
            "severity": 6,
            "condition": lambda f: (f["club_detection_error"]),
            "action": lambda: ball_only_message(raw_data['side_angle'])
        },
        {
            "group": None,  # Will be determined dynamically
            "name": "Club Only Detected",
            "category": "all",
            "severity": 6,
            "condition": lambda f: (f["ball_detection_error"]),
            "action": lambda: club_only_message(raw_data['swing_path'])
        },
        # -------------------------------
        # Shot shaping
        # - Severity levels: 5=WORST, 1=BEST
        # -------------------------------
        {
            # Worst-case
            "group": "Hook",
            "name": "Pull Hook",
            "category": "all",
            "severity": 5,
            "condition": lambda f: ((f["face_slight_left"] and f["path_extreme_right"]) or
                                    (f["face_extreme_left"] and (f["path_slight_right"] or f["path_extreme_right"]))
            ),
            "action": lambda: "Pull hook: You're closing the clubface too much and swinging in-to-out. If your shot is landing left of the target, try slightly opening your stance."
        },
        {
            "group": "Pull",
            "name": "Pull Draw",
            "category": "all",
            "severity": 4,
            "condition": lambda f: (f["face_slight_left"] and f["path_straight"]) or 
                                    (f["face_extreme_left"] and (f["path_straight"] or f["path_slight_left"])),
            "action": lambda: "Pull draw: Your clubface is closed with a neutral-to-left path. Try weakening your grip and keeping the face more square."
        },
        {
            "group": "Pull",
            "name": "Pull",
            "category": "all",
            "severity": 3,
            "condition": lambda f: (f["face_slight_left"] and f["path_slight_left"] and not f["face_to_path_slight_left"])   or   (f["face_extreme_left"] and (f["path_extreme_left"])),
            "action": lambda: "Pull: Both your face and path are left, causing a pull. Try weakening your grip and keeping the face more square."
        },
        {
            # Ideal fade
            "group": "Ideal",
            "name": "Pull Fade",
            "category": "all",
            "severity": 1,
            "condition": lambda f: f["face_slight_left"] and f["path_slight_left"] and f["face_to_path_slight_left"],
            "action": lambda: "Fade: Pure fade, was that Tiger Woods?"  # Encouraging message
        },
        {
            "group": "Pull",
            "name": "Pull Slice",
            "category": "all",
            "severity": 2,
            "condition": lambda f: f["face_slight_left"] and f["path_extreme_left"],
            "action": lambda: "Pull slice: Your club face is aiming left and your swing path is out-to-in. Try weakening your grip and keeping the face more square."
        },
        {
            "group": "Hook",
            "name": "Straight Hook",
            "category": "all",
            "severity": 4,
            "condition": lambda f: f["face_straight"] and f["path_extreme_right"],
            "action": lambda: "Straight hook: Your path is in-to-out while the face stays square. If your shot is landing left of the target, try slightly opening your stance."
        },
        {
            "group": "Hook",
            "name": "Straight Draw",
            "category": "all",
            "severity": 3,
            "condition": lambda f:  (f["face_straight"] and f["path_slight_right"]) or
                                    (f["face_slight_left"] and f["path_slight_right"]),
            "action": lambda: "Straight draw: A gentle rightward path with a square face is causing a draw. If your shot is landing left of the target, try slightly opening your stance."
        },
        {
            # Ideal straight
            "group": "Ideal",
            "name": "Straight",
            "category": "all",
            "severity": 1,
            "condition": lambda f: f["face_straight"] and f["path_straight"],
            "action": lambda: "Straight: Now that is a strike!"  # Encouraging message
        },
        {
            "group": "Slice",
            "name": "Straight Fade",
            "category": "all",
            "severity": 3,
            "condition": lambda f:  (f["face_straight"] and f["path_slight_left"]) or
                                    (f["face_slight_right"] and f["path_slight_left"]),
            "action": lambda: "Straight fade: A mild leftward path with a square face produces this fade. If your shot is landing right of the target, try slightly closing your stance."
        },
        {
            "group": "Slice",
            "name": "Straight Slice",
            "category": "all",
            "severity": 4,
            "condition": lambda f: f["face_straight"] and f["path_extreme_left"],
            "action": lambda: "Straight slice: The face is square, but your path is far left, causing a slice. If your shot is landing right of the target, try slightly closing your stance."
        },
        {
            "group": "Push",
            "name": "Push Hook",
            "category": "all",
            "severity": 2,
            "condition": lambda f: f["face_slight_right"] and f["path_extreme_right"],
            "action": lambda: "Push hook: Your path is in-to-out and the face is slightly open. Try strengthening your grip on the club."
        },
        {
            # Ideal draw
            "group": "Ideal",
            "name": "Push Draw",
            "category": "all",
            "severity": 1,
            "condition": lambda f: f["face_slight_right"] and f["path_slight_right"] and f["face_to_path_slight_right"],
            "action": lambda: "Draw: Buttery draw, was that Rory McIlroy?"
        },
        {
            "group": "Push",
            "name": "Push",
            "category": "all",
            "severity": 3,
            "condition": lambda f: (f["face_slight_right"] and f["path_slight_right"] and not f["face_to_path_slight_right"])   or   (f["face_extreme_right"] and f["path_extreme_right"]),
            "action": lambda: "Push: A rightward path and open face are sending shots directly right. Try strengthening your grip on the club."
        },
        {
            "group": "Push",
            "name": "Push Fade",
            "category": "all",
            "severity": 4,
            "condition": lambda f: (f["face_extreme_right"] and f["path_slight_right"])   or   (f["face_slight_right"] and f["path_straight"]),
            "action": lambda: "Push fade: You're swinging slightly right with an open face, causing a fading shot that starts right. Try strengthening your grip on the club."
        },
        {
            # Worst-case
            "group": "Slice",
            "name": "Push Slice",
            "category": "all",
            "severity": 5,
            "condition": lambda f: (f["face_extreme_right"] and (f["path_straight"] or f["path_slight_left"] or f["path_extreme_left"]))   or   (f["face_slight_right"] and f["path_extreme_left"])   or   (f["face_slight_right"] and f["path_slight_left"]),
            "action": lambda: "Push slice: Your face is open and your path is too far left, exaggerating spin. If your shot is landing right of the target, try slightly closing your stance."
        }
        # -------------------------------
        # Ball flight
        # - Heavily depends on club selection; currently we assume 7-iron
        # - Our optimal conditions come from the LPGA tour averages here: https://www.trackman.com/blog/golf/introducing-updated-tour-averages 
        #    - We are currently using LPGA because their numbers are closer to our target market of novice to intermediate players
        # - We will consider drivers, woods, long irons (2-4), mid-irons (5-7), long-irons (8-PW), and wedges
        # -------------------------------
    ]


    # Filter rules based on user category selection
    selected_rules = [rule for rule in rules if (rule["category"] == club_selection or rule["category"] == "all")]

    triggered_rules = []

    # Only check conditions within the selected rule set
    for rule in selected_rules:
        if rule["condition"](facts):
            triggered_rules.append(rule)

    if triggered_rules:
        # Pick rule with highest severity
        best_rule = max(triggered_rules, key=lambda r: r["severity"])
        result = best_rule["action"]()
        
        # Check if result is a tuple (group, message) or just message
        if isinstance(result, tuple):
            group, feedback = result
        else:
            group = best_rule["group"]
            feedback = result
    else:
        group = "Ideal"
        feedback = "No swing issues detected."

    # -----------------------
    # Handle the Error Cases
    # - If nothing was detected, return all zeroes
    # - If club was not detected, return only ball metrics, with swing_path, attack_angle, and face_angle set to zero
    # - If ball was not detected, return only club metrics, with side_angle, and face angle set to zero
    # -----------------------
    if facts["club_and_ball_detection_error"]:
        # Nothing detected - return all zeros
        raw_data["face_angle"] = 0.0
        raw_data["swing_path"] = 0.0
        raw_data["attack_angle"] = 0.0
        raw_data["side_angle"] = 0.0
    elif facts["club_detection_error"]:
        # Club not detected - zero out club metrics, keep ball metrics
        raw_data["face_angle"] = 0.0
        raw_data["swing_path"] = 0.0
        raw_data["attack_angle"] = 0.0
        # side_angle remains as detected
    elif facts["ball_detection_error"]:
        # Ball not detected - zero out ball metrics, keep club metrics
        raw_data["side_angle"] = 0.0
        raw_data["face_angle"] = 0.0
        # swing_path, attack_angle remain as detected

    print(f"{feedback}")

    return {
        "metrics": {
            "type": "metrics",
            "face angle": round(raw_data["face_angle"], 2),
            "swing path": round(raw_data["swing_path"], 2),
            "attack angle": round(raw_data["attack_angle"], 2),
            "side angle": round(raw_data["side_angle"], 2),
            "group": group
        },
        "feedback":  feedback
    }


# -------------------------------
# Main function for testing
# -------------------------------
# if __name__ == "__main__":
#     result = rule_based_system("mid-iron")
