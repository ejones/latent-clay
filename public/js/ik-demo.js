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
    const skeleton = new SkeletonHelper(group);
    scene.add(skeleton);

    // Find bones by name
    const bones = skeleton.bones;
    const targetBone = bones.find(bone => bone.name.includes('Hand'));
    const effectorBone = bones.find(bone => 
        bone.name.includes('LowerArm') || 
        bone.name.includes('Forearm') || 
        bone.name.includes('Elbow') || 
        bone.name.includes('Arm')
    );
    console.log('Effector Bone:', effectorBone ? effectorBone.name : 'Not found');
    const linkBones = bones.filter(bone => bone.name.includes('UpperArm') || bone.name.includes('Shoulder'));

    console.log('Target Bone:', targetBone ? targetBone.name : 'Not found');
    console.log('Effector Bone:', effectorBone ? effectorBone.name : 'Not found');
    console.log('Link Bones:', linkBones.map(bone => bone.name));

    let ccdikSolver;

    console.log(bones);
    console.log(model.skeleton.bones);

    if (targetBone && effectorBone && linkBones.length > 0) {
        // Set up CCDIKSolver
        const iks = [
            {
                target: bones.indexOf(targetBone),
                effector: bones.indexOf(effectorBone),
                links: linkBones.reverse().map(bone => ({
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

    // Animation loop
    function animate() {
        requestAnimationFrame(animate);
        //updateIK();
        controls.update();
        renderer.render(scene, camera);
        
    }

    animate();
}, undefined, function (error) {
    console.error(error);
});

// Set camera position
camera.position.set(0, 1, 5);
