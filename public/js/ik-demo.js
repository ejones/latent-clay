import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { SkeletonHelper } from 'three/examples/jsm/helpers/SkeletonHelper.js';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

// Initialize scene, camera, and renderer
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
const renderer = new THREE.WebGLRenderer();
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

// Add orbit controls
const controls = new OrbitControls(camera, renderer.domElement);

// Load humanoid model
const loader = new GLTFLoader();
loader.load('path/to/humanoid/model.glb', function (gltf) {
    const model = gltf.scene;
    scene.add(model);

    // Add skeleton helper
    const skeleton = new SkeletonHelper(model);
    scene.add(skeleton);

    // Set up IK (this is a simple example, real IK would be more complex)
    const bones = skeleton.bones;
    const targetPosition = new THREE.Vector3(0, 1, 0);

    function updateIK() {
        // Simple IK logic to move the end effector towards the target
        const endEffector = bones[bones.length - 1];
        endEffector.position.lerp(targetPosition, 0.1);
    }

    // Animation loop
    function animate() {
        requestAnimationFrame(animate);
        updateIK();
        controls.update();
        renderer.render(scene, camera);
    }

    animate();
}, undefined, function (error) {
    console.error(error);
});

// Set camera position
camera.position.z = 5;
