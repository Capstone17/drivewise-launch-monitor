
from metric_output import return_metrics

# -------------------------------
# Raw input data
# -------------------------------
raw_data = return_metrics()  # face_angle, swing_path, attack_angle, side_angle, face_to_path

# -------------------------------
# Fact base
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
    "face_extreme_left": raw_data['face_angle'] < -4.0,
    "face_slight_left": -4.0 <= raw_data['face_angle'] < -2.0,
    "face_straight": -2.0 <= raw_data['face_angle'] < 2.0,
    "face_slight_right": 2.0 <= raw_data['face_angle'] < 4.0,
    "face_extreme_right": 4.0 <= raw_data['face_angle'],

    "path_extreme_left": raw_data['swing_path'] < -5.0,  # Slice
    "path_slight_left": -5.0 <= raw_data['swing_path'] < -2.0,  # Fade-bias
    "path_straight": -2.0 <= raw_data['swing_path'] < 2.0,  # Neutral
    "path_slight_right": 2.0 <= raw_data['swing_path'] < 5.0,  # Draw-bias
    "path_extreme_right": 5.0 <= raw_data,  # Hook

    "attack_extreme_up": 5.0 < raw_data['attack_angle'],
    "attack_up": 3.0 < raw_data['attack_angle'] <= 5.0,  # Ideal for driver distance
    "attack_slight_up": 1.0 < raw_data['attack_angle'] <= 3.0,
    "attack_neutral": -2.0 < raw_data['attack_angle'] <= 1.0,
    "attack_slight_down": -5.0 < raw_data['attack_angle'] <= -2.0,  # Ideal for iron shots
    "attack_very_down": -10.0 < raw_data['attack_angle'] <= -5.0,  # Ideal for wedge shots
    "attack_extreme_down": raw_data['attack_angle'] <= -10.0,

    "side_extreme_left": raw_data['side_angle'] < -4.0,
    "side_slight_left": -4.0 <= raw_data['side_angle'] < -1.0,  # Ideal for a fade
    "side_straight": -1.0 <= raw_data['side_angle'] < 1.0,  # Ideal for a straight shot
    "side_slight_right": 1.0 <= raw_data['side_angle'] < 4.0,  # Ideal for hitting a draw
    "side_extreme_right": 4.0 <= raw_data['side_angle'],

    "face_to_path_extreme_left": raw_data['side_angle'] < -4.0,
    "face_to_path_slight_left": -4.0 <= raw_data['side_angle'] < -2.0,  # Ideal for a fade
    "face_to_path_straight": -2.0 <= raw_data['side_angle'] < 2.0,  # Ideal for a straight shot
    "face_to_path_slight_right": 2.0 <= raw_data['side_angle'] < 4.0,  # Ideal for hitting a draw
    "face_to_path_extreme_right": 4.0 <= raw_data['side_angle']
}

# -------------------------------
# Rules
# - The shot shaping diagnosis was taken from this Reddit post: https://www.reddit.com/r/GolfSwing/comments/1546tus/i_think_everyone_should_save_this_diagram_to_help/ 
# -------------------------------
rules = [
    # -------------------------------
    # Shot shaping
    # -------------------------------
    {
        # Worst-case
        "name": "Pull Hook",
        "condition": lambda f: (f["face_slight_left"] and f["path_extreme_right"])   or   (f["face_extreme_left"] and (f["path_slight_right"] or f["path_extreme_right"])),
        "action": lambda: print("Pull hook: You're closing the clubface too much and swinging aggressively out-to-in. Try keeping your clubface more neutral and reducing how far left you're swinging.")
    },
    {
        "name": "Pull Draw",
        "condition": lambda f: (f["face_slight_left"] and f["path_straight"])   or   (f["face_extreme_left"] and (f["path_straight"] or f["path_slight_left"])),
        "action": lambda: print("Pull draw: Your clubface is slightly closed with a neutral-to-left path. Focus on squaring the face and aiming your swing path a bit more to the right.")
    },
    {
        "name": "Pull",
        "condition": lambda f: (f["face_slight_left"] and f["path_slight_left"] and not f["face_to_path_slight_left"])   or   (f["face_extreme_left"] and (f["path_extreme_left"])),
        "action": lambda: print("Pull: Both your face and path are left, causing a pull. Try aligning your stance and path more rightward and ensure the face matches the path.")
    },
    {
        # Ideal fade
        "name": "Pull Fade",
        "condition": lambda f: f["face_slight_left"] and f["path_slight_left"] and f["face_to_path_slight_left"],
        "action": lambda: print("Fade: ")
    },
    {
        "name": "Pull Slice",
        "condition": lambda f: f["face_slight_left"] and f["path_extreme_left"],
        "action": lambda: print("Pull slice: Your club face is aiming left and your out-to-in path are producing sidespin. Work on neutralizing your swing path and straightening the face angle to prevent excessive spin.")
    },
    {
        "name": "Straight Hook",
        "condition": lambda f: f["face_straight"] and f["path_extreme_right"],
        "action": lambda: print("Straight hook: Your path is far right while the face stays square. Aim to reduce the in-to-out path and allow your face to release away from your body to match it.")
    },
    {
        "name": "Straight Draw",
        "condition": lambda f: f["face_straight"] and f["path_slight_right"],
        "action": lambda: print("Straight draw: ")
    },
    {
        # Ideal straight
        "name": "Straight",
        "condition": lambda f: f["face_straight"] and f["path_straight"],
        "action": lambda: print("Straight: ")
    },
    {
        "name": "Straight Fade",
        "condition": lambda f: f["face_straight"] and f["path_slight_left"],
        "action": lambda: print("Straight fade: ")
    },
    {
        "name": "Straight Slice",
        "condition": lambda f: f["face_straight"] and f["path_extreme_left"],
        "action": lambda: print("Straight slice: ")
    },
    {
        "name": "Push Hook",
        "condition": lambda f: f["face_slight_right"] and f["path_extreme_right"],
        "action": lambda: print("Push hook: ")
    },
    {
        # Ideal draw
        "name": "Push Draw",
        "condition": lambda f: f["face_slight_right"] and f["path_slight_right"] and f["face_to_path_slight_right"],
        "action": lambda: print("Draw: ")
    },
    {
        "name": "Push",
        "condition": lambda f: (f["face_slight_right"] and f["path_slight_right"] and not f["face_to_path_slight_right"])   or   (f["face_extreme_right"] and f["path_extreme_right"]),
        "action": lambda: print("Push: ")
    },
    {
        "name": "Push Fade",
        "condition": lambda f: (f["face_extreme_right"] and f["path_slight_right"])   or   (f["face_slight_right"] and f["path_straight"]),
        "action": lambda: print("Push fade: ")
    },
    {
        # Worst-case
        "name": "Push Slice",
        "condition": lambda f: (f["face_extreme_right"] and (f["path_straight"] or f["path_slight_left"] or f["path_extreme_left"]))   or   (f["face_slight_right"] and f["path_extreme_left"])   or   (f["face_slight_right"] and f["path_slight_left"]),
        "action": lambda: print("Push slice: ")
    },
    # -------------------------------
    # Ball flight
    # - Heavily depends on club selection
    # -------------------------------
    {
        "name": "Very Upward",
        "condition": lambda f: f["attack_extreme_up"],
        "action": lambda: print("Very upward: ")
    }
]

# -------------------------------
# Inference engine
# -------------------------------
def run_inference(facts, rules):
    print("Running rule-based inference...\n")
    triggered = 0
    for rule in rules:
        if rule["condition"](facts):
            rule["action"]()
            triggered += 1
    if triggered == 0:
        print("No rules were triggered.")

# -------------------------------
# Run
# -------------------------------
if __name__ == "__main__":
    run_inference(facts, rules)
