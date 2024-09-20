import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { SkeletonHelper } from 'three';
import { CCDIKSolver } from 'three/addons/animation/CCDIKSolver.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

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
loader.load('models/gltf/Xbot.glb', function (gltf) {
    const model = gltf.scene;
    scene.add(model);

    // Add skeleton helper
    const skeleton = new SkeletonHelper(model);
    scene.add(skeleton);

    // Find bones by name
    const bones = skeleton.bones;
    const targetBone = bones.find(bone => bone.name.includes('Hand'));
    const effectorBone = bones.find(bone => bone.name.includes('Forearm'));
    const linkBones = bones.filter(bone => bone.name.includes('UpperArm') || bone.name.includes('Shoulder'));

    if (targetBone && effectorBone && linkBones.length > 0) {
        // Set up CCDIKSolver
        const iks = [
            {
                target: bones.indexOf(targetBone),
                effector: bones.indexOf(effectorBone),
                links: linkBones.map(bone => ({ index: bones.indexOf(bone) })),
                iteration: 10,
                minAngle: 0.0,
                maxAngle: Math.PI
            }
        ];

        const ccdikSolver = new CCDIKSolver(model, iks);

    function updateIK() {
        if (ccdikSolver) {
            ccdikSolver.update();
        }
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
