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

class BoneTarget {
      constructor(model, bone, effector) {
          this.model = model;
          this.bone = bone;
          this.effector = effector;

          this._positionMap = new Map(
              model.skeleton.bones.map(bone => {
                  const vec = new THREE.Vector3();
                  for (let p = bone; p && p !== this.model; p = p.parent) {
                      vec.add(p.position);
                  }
                  return [bone.name, vec];
              })
          );

          this._vec = new THREE.Vector3();
      }

      setModelRelative(x, y, z) {
          if (y === undefined) {
              this._vec.copy(x); // x is Vector3
          } else {
              if (typeof y === 'string') {
                  for (const [k, v] of this._positionMap) {
                      if (k.includes(y)) {
                          y = v.y;
                          break;
                      }
                  }
              }
              this._vec.set(x, y, z);
          }
          for (let p = this.bone.parent; p && p !== this.model; p = p.parent) {
              this._vec.sub(p.position);
          }
          this.bone.position.copy(this._vec);
      }
}

const wireUpForIk = (model, specs) => {
    const { bones } = model.skeleton;

    const ikPairs = Object.entries(specs).map(([name, spec]) => {
        const effectorBone = bones.find(bone => bone.name.includes(spec.effector));
        const linkBones = [];
        const targetPos = new THREE.Vector3();
        targetPos.copy(effectorBone.position);

        let link = effectorBone.parent;
        while (link && !link.name?.includes(spec.base)) {
            linkBones.push(link);
            targetPos.add(link.position);
            console.log('add', link.name, link.position.x, link.position.y, link.position.z);
            link = link.parent;
        }
        if (!link?.name?.includes(spec.base)) {
            console.warn('Couldnt find a path from effector to base', linkBones, link);
            return { /* ... ?? */ };
        }

        const baseBone = link;
        const targetBone = new THREE.Bone({name: `target_${effectorBone.name}`});
        targetBone.position.copy(targetPos);
        baseBone.add(targetBone);

        const makeIk = bones => ({
            target: bones.indexOf(targetBone),
            effector: bones.indexOf(effectorBone),
            links: linkBones.map(bone => ({
                index: bones.indexOf(bone),
                rotationMin: new THREE.Vector3(-Math.PI / 2, -Math.PI / 2, -Math.PI / 2),
                rotationMax: new THREE.Vector3(Math.PI / 2, Math.PI / 2, Math.PI / 2),
                enabled: true
            })),
            iteration: 10,
            minAngle: 0.0,
            maxAngle: Math.PI
        });

        return [name, { targetBone, makeIk, effectorBone }];
    });
    
    const newBones = [...bones, ...ikPairs.map(([, { targetBone }]) => targetBone)];
    model.skeleton = new THREE.Skeleton(newBones);

    const targetBones = Object.fromEntries(
        ikPairs.map(([name, { targetBone, effectorBone }]) =>
            [name, new BoneTarget(model, targetBone, effectorBone)]
        )
    );

    const iks = ikPairs.map(([, { makeIk }]) => makeIk(newBones));

    return [targetBones, iks];
};

const updateWave = (targetBone) => {
    const waveAmplitudeX = 20;
    const waveAmplitudeY = 10;
    const waveFrequency = 10;

    const elapsedTime = clock.getElapsedTime();
    const newX = 20 + waveAmplitudeX * Math.cos(waveFrequency * elapsedTime);
    const newY = 40 + waveAmplitudeY * Math.sin(waveFrequency * elapsedTime);

    targetBone.position.set(newX, newY, 0);
};

function animateClap({leftHand, rightHand}) {
    const waveFrequency = 10;
    const elapsedTime = clock.getElapsedTime();
    const xPos = 15 * (1.1 + Math.sin(waveFrequency * elapsedTime));
    leftHand.setModelRelative(xPos, 'Spine1', 50);
    rightHand.setModelRelative(-xPos, 'Spine1', 50);
}

/*
TODO adapt to new BoneTarget
function animateHead(targetBones) {
    const headBone = targetBones.head;
    if (headBone) {
        const nodFrequency = 5;
        const elapsedTime = clock.getElapsedTime();
        headBone.position.z = 5 + 5 * Math.sin(nodFrequency * elapsedTime);
    }
}


function animateLegs(targetBones) {
    const leftFootBone = targetBones.leftFoot;
    const rightFootBone = targetBones.rightFoot;
    if (leftFootBone && rightFootBone) {
        const walkFrequency = 10;
        const elapsedTime = clock.getElapsedTime();
        leftFootBone.position.z = 18 * Math.sin(walkFrequency * elapsedTime);
        rightFootBone.position.z = -18 * Math.sin(walkFrequency * elapsedTime);
    }
}
*/

// Load humanoid model
const loader = new GLTFLoader();
loader.load('models/gltf/Xbot.glb', function (gltf) {
    const group = gltf.scene;
    scene.add(group);

    group.position.y -= 1;

    let model;
    group.traverse(node => {
        if (!model && node.isSkinnedMesh) {
            model = node;
        }
    });

    const [targetBones, iks] = wireUpForIk(model, {
        leftHand: {base: 'Shoulder', effector: 'LeftHand'},
        rightHand: {base: 'Shoulder', effector: 'RightHand'},
        head: {base: 'Spine', effector: 'Head'},
        leftFoot: {base: 'Hips', effector: 'LeftFoot'},
        rightFoot: {base: 'Hips', effector: 'RightFoot'},
    });

    const skeleton = new SkeletonHelper(group);
    scene.add(skeleton);

    targetBones.leftHand.effector.rotateX(Math.PI);
    targetBones.rightHand.effector.rotateX(Math.PI);

    const ccdikSolver = new CCDIKSolver(model, iks);

    function updateTargetBones() {
        //updateWave(targetBones.leftHand);
        //animateHead(targetBones);
        animateClap(targetBones);
        //animateLegs(targetBones);
    }

    // Animation loop
    function animate() {
        requestAnimationFrame(animate);
        updateTargetBones();
        ccdikSolver.update();
        controls.update();
        renderer.render(scene, camera);
        
    }

    animate();
}, undefined, function (error) {
    console.error(error);
});

// Set camera position
camera.position.set(0, 0, 5);
