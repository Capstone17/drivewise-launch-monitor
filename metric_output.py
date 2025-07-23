from metric_calculation import face_angle_calc, swing_path_calc, attack_angle_calc, side_angle_calc 

face_angle = face_angle_calc(yaw_current, yaw_ideal)
swing_path = swing_path_calc(pose1, pose2, reference_vector)
attack_angle = attack_angle_calc(pose1, pose2)
side_angle = side_angle_calc(ball_pos1, ball_pos2, reference_vector)