
# from metric_output import return_metrics
from .metric_output_3_metrics import return_metrics  # Use this until face angle can be accurately detected

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
    "path_extreme_right": 5.0 <= raw_data['swing_path'],  # Hook

    "attack_extreme_up": 5.0 < raw_data['attack_angle'],
    "attack_up": 3.0 < raw_data['attack_angle'] <= 5.0,  # Ideal for driver distance
    "attack_slight_up": 1.0 < raw_data['attack_angle'] <= 3.0,
    "attack_neutral": -2.0 < raw_data['attack_angle'] <= 1.0,
    "attack_1.5to3_down": -1.5 < raw_data['attack_angle'] <= 3.0,  # Ideal for mid-irons
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
    # - Severity levels: 5=WORST, 1=BEST
    # -------------------------------
    {
        # Worst-case
        "name": "Pull Hook",
        "category": "all",
        "severity": 5,
        "condition": lambda f: (f["face_slight_left"] and f["path_extreme_right"])   or   (f["face_extreme_left"] and (f["path_slight_right"] or f["path_extreme_right"])),
        "action": lambda: "Pull hook: You're closing the clubface too much and swinging aggressively in-to-out. Try keeping your clubface more neutral and reducing how far right you're swinging."
    },
    {
        "name": "Pull Draw",
        "category": "all",
        "severity": 4,
        "condition": lambda f: (f["face_slight_left"] and f["path_straight"])   or   (f["face_extreme_left"] and (f["path_straight"] or f["path_slight_left"])),
        "action": lambda: "Pull draw: Your clubface is slightly closed with a neutral-to-left path. Focus on squaring the face and aiming your swing path a bit more to the right."
    },
    {
        "name": "Pull",
        "category": "all",
        "severity": 3,
        "condition": lambda f: (f["face_slight_left"] and f["path_slight_left"] and not f["face_to_path_slight_left"])   or   (f["face_extreme_left"] and (f["path_extreme_left"])),
        "action": lambda: "Pull: Both your face and path are left, causing a pull. Try aligning your stance and path more rightward and ensure the face matches the path."
    },
    {
        # Ideal fade
        "name": "Pull Fade",
        "category": "all",
        "severity": 1,
        "condition": lambda f: f["face_slight_left"] and f["path_slight_left"] and f["face_to_path_slight_left"],
        "action": lambda: "Fade: Nice fade! You're pure."  # Encouraging message
    },
    {
        "name": "Pull Slice",
        "category": "all",
        "severity": 2,
        "condition": lambda f: f["face_slight_left"] and f["path_extreme_left"],
        "action": lambda: "Pull slice: Your club face is aiming left and your out-to-in path are producing sidespin. Work on neutralizing your swing path and straightening the face angle to prevent excessive spin."
    },
    {
        "name": "Straight Hook",
        "category": "all",
        "severity": 4,
        "condition": lambda f: f["face_straight"] and f["path_extreme_right"],
        "action": lambda: "Straight hook: Your path is far right while the face stays square. Aim to reduce the in-to-out path and allow your face to release away from your body to match it."
    },
    {
        "name": "Straight Draw",
        "category": "all",
        "severity": 3,
        "condition": lambda f: f["face_straight"] and f["path_slight_right"],
        "action": lambda: "Straight draw: A gentle rightward path with a square face is causing a draw. If your shot is landing too far left of the target, try slightly weakening your grip or evening out your path."
    },
    {
        # Ideal straight
        "name": "Straight",
        "category": "all",
        "severity": 1,
        "condition": lambda f: f["face_straight"] and f["path_straight"],
        "action": lambda: "Straight: Now that is a strike!"  # Encouraging message
    },
    {
        "name": "Straight Fade",
        "category": "all",
        "severity": 3,
        "condition": lambda f: f["face_straight"] and f["path_slight_left"],
        "action": lambda: "Straight fade: A mild leftward path with a square face produces this fade. To straighten the shot, shift your path slightly more right."
    },
    {
        "name": "Straight Slice",
        "category": "all",
        "severity": 4,
        "condition": lambda f: f["face_straight"] and f["path_extreme_left"],
        "action": lambda: "Straight slice: The face is square, but your path is far left, causing a slice. Try to shallow your path and swing more inside-to-out."
    },
    {
        "name": "Push Hook",
        "category": "all",
        "severity": 2,
        "condition": lambda f: f["face_slight_right"] and f["path_extreme_right"],
        "action": lambda: "Push hook: Your path is in-to-out and the face is slightly open. Reduce the in-to-out path and check for overly strong grip or early release."
    },
    {
        # Ideal draw
        "name": "Push Draw",
        "category": "all",
        "severity": 1,
        "condition": lambda f: f["face_slight_right"] and f["path_slight_right"] and f["face_to_path_slight_right"],
        "action": lambda: "Draw: Buttery draw, was that Rory?"
    },
    {
        "name": "Push",
        "category": "all",
        "severity": 3,
        "condition": lambda f: (f["face_slight_right"] and f["path_slight_right"] and not f["face_to_path_slight_right"])   or   (f["face_extreme_right"] and f["path_extreme_right"]),
        "action": lambda: "Push: A rightward path and open face are sending shots directly right. Work on squaring the clubface and adjusting alignment toward the target."
    },
    {
        "name": "Push Fade",
        "category": "all",
        "severity": 4,
        "condition": lambda f: (f["face_extreme_right"] and f["path_slight_right"])   or   (f["face_slight_right"] and f["path_straight"]),
        "action": lambda: "Push fade: You're swinging slightly right with an open face, causing a fading shot that starts right. Square the face more at impact or realign the path to reduce curvature."
    },
    {
        # Worst-case
        "name": "Push Slice",
        "category": "all",
        "severity": 5,
        "condition": lambda f: (f["face_extreme_right"] and (f["path_straight"] or f["path_slight_left"] or f["path_extreme_left"]))   or   (f["face_slight_right"] and f["path_extreme_left"])   or   (f["face_slight_right"] and f["path_slight_left"]),
        "action": lambda: "Push slice: Your face is open and your path is too far left, exaggerating spin. Square the face earlier in the downswing and reduce your out-to-in motion."
    },
    # -------------------------------
    # Ball flight
    # - Heavily depends on club selection; currently we assume 7-iron
    # - Our optimal conditions come from the LPGA tour averages here: https://www.trackman.com/blog/golf/introducing-updated-tour-averages 
    #    - We are currently using LPGA because their numbers are closer to our target market of novice to intermediate players
    # - We will consider drivers, woods, long irons (2-4), mid-irons (5-7), long-irons (8-PW), and wedges
    # -------------------------------

    ### MID-IRONS
    {
        "name": "Good Attack",
        "category": "mid-iron",
        "severity": 1,
        "condition": lambda f: f["attack_1.5to3_down"],
        "action": lambda: "Good attack angle: An extremely upward attack can lead to topped shots or high spin. Try to swing down on the ball and keep your weight more centered through impact."
    }
]

# -------------------------------
# Output as a function (for bluetooth)
# -------------------------------
def rule_based_system(club_selection):
    
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
        feedback = best_rule["action"]()
    else:
        feedback = "No swing issues detected."

    print(f"{feedback}")

    return {
        "metrics": {
            "face angle": raw_data["face_angle"],
            "swing path": raw_data["swing_path"],
            "attack angle": raw_data["attack_angle"],
            "side angle": raw_data["side_angle"]
        },
        "feedback": feedback
    }


# -------------------------------
# Main function for testing
# -------------------------------
# if __name__ == "__main__":
#     result = rule_based_system("mid-iron")