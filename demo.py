from Ippo_ik.Ippo_ik import Ippo

# Define the initial poses for the object and obstacles
obj_pose = [0, 0.5, 0.3, 0, 1.57, 1.57]  # Example object pose [x, y, z]
obs1_pos = [0.5, 0.0, 0.2]  # Example obstacle 1 position [x, y, z]
obs2_pos = [0.0, 0.5, 0.2]  # Example obstacle 2 position [x, y, z]
obs3_pos = [0.5, 0.5, 0.2]  # Example obstacle 3 position [x, y, z]

# Create an instance of the Ippo class
ippo = Ippo(obj_pose, obs1_pos, obs2_pos, obs3_pos)

# Call the ik function to perform inverse kinematics
action = ippo.ik(obj_pose, obs1_pos, obs2_pos, obs3_pos)

print("Chosen action:", action)
