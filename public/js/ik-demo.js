import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { IK, IKChain, IKJoint } from 'three-ik';
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

    // Set up IK
    const ik = new IK();
    const chain = new IKChain();

    // Assuming the model has a bone structure with names like 'Arm' and 'Leg'
    const armBone = skeleton.getBoneByName('Arm');
    const handBone = skeleton.getBoneByName('Hand');

    if (armBone && handBone) {
        chain.add(new IKJoint(armBone, { constraints: [] }));
        chain.add(new IKJoint(handBone, { constraints: [] }));
        ik.add(chain);

        const target = new THREE.Object3D();
        target.position.set(0, 1, 0);
        scene.add(target);

        chain.setTarget(target);

        function updateIK() {
            ik.solve();
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
