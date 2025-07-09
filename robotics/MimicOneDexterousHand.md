# mimic-one: a Scalable Model Recipe for General Purpose Robot Dexterity

Date: 18th June 2025. Fancied reading a robotics paper today!

[arXiv link](https://arxiv.org/abs/2506.11916)

## Key Points
- Paper was focused around a new 16 DOF freedom hand they had just developed
- **U-Net diffusion based planner** trained on **imitation learning**.
- Data was collected for IL using Apple Vision Pro hand tracking.
- Imitation learning works by trying to minimise the total Euclidean distance between the target and the hand..
- Identify failed modes and then collect more data to make up for mistakes

## Key Methods
- Predict actions chunks rather than single actions to maintain temporal consistency 
- Input is images through vision transformer encoder + low dimensional info about arm projected into U-Net embedded space
- Tips they gave:
  1. Use pose relative to last position
  2. Add distractor objects 50% of the time
  3. Vary lighting conditions