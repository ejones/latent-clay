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

const ambientLight = new THREE.AmbientLight(0x404040); // soft white light
scene.add(ambientLight);

const directionalLight = new THREE.DirectionalLight(0xffffff, 0.5);
directionalLight.position.set(1, 1, 1).normalize();
scene.add(directionalLight);

// Add orbit controls
const controls = new OrbitControls(camera, renderer.domElement);

const clock = new THREE.Clock();

// Load humanoid model
const loader = new GLTFLoader();
loader.load('models/gltf/Xbot.glb', function (gltf) {
    const group = gltf.scene;
    scene.add(group);

    let model;
    group.traverse(node => {
        if (!model && node.isSkinnedMesh) {
            model = node;
        }
    });

    // Add skeleton helper

    // Find bones by name
    let bones = model.skeleton.bones;
    const effectorBone = bones.find(bone => bone.name.includes('Hand'));
    console.log('bones', bones.length);

    const linkBones = [];

    let link = effectorBone.parent;
    while (link && !link.name?.includes('Shoulder')) {
        linkBones.push(link);
        link = link.parent;
    }
    if (!link?.name?.includes('Shoulder')) {
        console.warn('Couldnt find a path to shoulder', linkBones, link);
        return;
    }
    const baseBone = link;
    const targetBone = new THREE.Bone({name: `target_${effectorBone.name}`});
    baseBone.add(targetBone);
    bones = [...bones, targetBone];
    model.skeleton = new THREE.Skeleton(bones);

    const skeleton = new SkeletonHelper(group);
    scene.add(skeleton);

    let ccdikSolver;

    if (targetBone && effectorBone && linkBones.length > 0) {
        // Set up CCDIKSolver
        const iks = [
            {
                target: bones.indexOf(targetBone),
                effector: bones.indexOf(effectorBone),
                links: linkBones.map(bone => ({
                    index: bones.indexOf(bone),
                    limitation: undefined, // Optional: Set specific limitations if needed
                    rotationMin: new THREE.Vector3(-Math.PI / 2, -Math.PI / 2, -Math.PI / 2),
                    rotationMax: new THREE.Vector3(Math.PI / 2, Math.PI / 2, Math.PI / 2),
                    enabled: true
                })),
                iteration: 10,
                minAngle: 0.0,
                maxAngle: Math.PI
            }
        ];

        ccdikSolver = new CCDIKSolver(model, iks);
    }

    function updateIK() {
        console.log('updateIK called');
        if (ccdikSolver) {
            console.log('ccdikSolver is updating');
            ccdikSolver.update();
        }
    }

    function updateTarget(deltaTime) {
        const waveAmplitude = 5; // Amplitude of the wave motion
        const waveFrequency = 1; // Frequency of the wave motion

        // Calculate the new Y position using a sine wave
        const newY = 20 + waveAmplitude * Math.sin(waveFrequency * clock.getElapsedTime());

        // Update the targetBone position
        targetBone.position.set(0, newY, -2);
    }

    // Animation loop
    function animate() {
        const deltaTime = clock.getDelta();
        requestAnimationFrame(animate);
        updateTarget(deltaTime);
        updateIK();
        controls.update();
        renderer.render(scene, camera);
        
    }

    animate();
}, undefined, function (error) {
    console.error(error);
});

// Set camera position
camera.position.set(0, 1, 5);
