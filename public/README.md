# IK Demo with Humanoid Model

This demo showcases inverse kinematics using the `CCDIKSolver` from three.js. To run this demo, you need a humanoid model in GLTF format. Below are the instructions to obtain and prepare a model for use in this demo.

## Requirements

1. **Model Format**: The model should be in GLTF or GLB format.
2. **Bone Structure**: The model should have a skeleton with named bones. The following bone names are expected for the IK setup:
   - `Hand`: The target bone for the IK solver.
   - `Forearm`: The effector bone for the IK solver.
   - `UpperArm` or `Shoulder`: The link bones for the IK solver.

## Obtaining a Model

1. **Download from Online Repositories**:
   - Websites like [Sketchfab](https://sketchfab.com) or [TurboSquid](https://www.turbosquid.com) offer a variety of 3D models. Ensure the model is rigged and available in GLTF/GLB format.

2. **Create Your Own Model**:
   - Use 3D modeling software like Blender to create and rig a humanoid model. Export the model in GLTF/GLB format.

3. **Convert Existing Models**:
   - If you have a model in another format, you can use Blender or online converters to convert it to GLTF/GLB.

## Setting Up the Model

1. **Place the Model**:
   - Save your model file in the appropriate directory and update the path in `ik-demo.js`:
     ```javascript
     loader.load('path/to/your/model.glb', function (gltf) {
     ```

2. **Verify Bone Names**:
   - Ensure the bone names in your model match those expected in the `ik-demo.js` file. Adjust the bone name filters in the script if necessary.

3. **Run the Demo**:
   - Open `ik-demo.html` in a web browser to see the IK demo in action.
